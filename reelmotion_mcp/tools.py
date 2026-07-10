import os
import httpx
import json
import logging
import re
from typing import Optional, Tuple

from request_context import get_api_token, get_conversation_uuid
from session_manager import get_session_manager
from logging_config import setup_logging
from moderation import is_disallowed_content, get_refusal_message
from pricing import (
    IMAGE_COSTS,
    QUANTITY_IMAGE_MODELS,
    SEEDANCE2_MODELS,
    SEEDANCE2_TOKEN_RATES,
    SEEDANCE2_VALID_ASPECT_RATIOS,
    VIDEO_DURATION_RULES,
    KLING_MODELS,
    KLING_ROUTE_BASE,
    KLING_ROUTE_EDIT,
    KLING_ROUTE_MOTION,
    KLING_ROUTE_REFERENCE,
    KLING_ROUTE_TURBO,
    _normalize_image_model,
    normalize_seedance_resolution,
    normalize_seedance_duration,
    normalize_kling_quality,
    compute_seedance2_cost,
    compute_kling_cost,
    speech_cost,
)
from generation_errors import (
    CATEGORY_BACKEND_UNAVAILABLE,
    CATEGORY_TIMEOUT,
    CATEGORY_UNKNOWN,
    format_generation_error,
    format_generation_processing,
    parse_backend_error,
)

setup_logging()
logger = logging.getLogger(__name__)


def is_blob_url(url: str) -> bool:
    """Check if a URL is a browser-only blob: URL that cannot be fetched server-side."""
    return isinstance(url, str) and url.startswith("blob:")


def is_data_uri(url: str) -> bool:
    """Check if a URL is an inline data: URI (base64-encoded media)."""
    return isinstance(url, str) and url.startswith("data:")


_VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".m4v", ".avi", ".mkv")


def _looks_like_video_url(url: Optional[str]) -> bool:
    """Best-effort: does this media URL point to a video (by file extension)?"""
    if not isinstance(url, str) or not url:
        return False
    path = url.split("?", 1)[0].split("#", 1)[0].lower()
    return path.endswith(_VIDEO_EXTENSIONS)


def normalize_image_model(model: str) -> Optional[str]:
    """
    Resolve a loose image model name to its canonical, case-sensitive backend
    name, or None if it isn't a known image model. There is NO Freepik model.
    Thin wrapper over pricing's normalizer (single source of truth).
    """
    return _normalize_image_model(model)


def _extract_image_urls(result: dict) -> list:
    """
    Pull every generated image URL out of a backend response, tolerant of the
    several shapes the endpoint returns:
      - Seedream/Midjourney hybrid: {"status": "completed", "image_url": "..."}
        (Midjourney also returns "all_result_urls": [4 variants]).
      - GPT/Nano Banana: {"images": [{"url": ...}]} or {"data": [...]}.
    The PRIMARY image (image_url) comes first; Midjourney variants are appended
    after it so callers can show only the primary by default.
    """
    urls: list = []

    def _add(value):
        if isinstance(value, str) and value and value not in urls:
            urls.append(value)
        elif isinstance(value, dict):
            _add(value.get("url"))

    _add(result.get("image_url") or result.get("result_url") or result.get("url"))

    images_data = result.get("images") or result.get("data")
    if isinstance(images_data, list):
        for img in images_data:
            _add(img)
    elif images_data is not None:
        _add(images_data)

    for variant in result.get("all_result_urls") or []:
        _add(variant)

    return urls


def clean_prompt_from_model_mentions(prompt: str) -> str:
    """
    Remove model name mentions from the prompt before sending to the backend.

    Examples:
        "anima este video con veo 3.1" -> "anima este video"
        "genera una imagen con GPT" -> "genera una imagen"
        "create a video using runway aleph" -> "create a video"
    """
    if not prompt:
        return prompt

    # JSON prompts (Veo 3 style) must reach the backend byte-identical — any
    # rewrite would corrupt the structure. If the prompt contains a JSON
    # object at all, never touch it.
    from workflow_state import detect_json_prompt
    if detect_json_prompt(prompt) is not None:
        logger.debug("Prompt contains JSON (prompt_is_json=True); passing through verbatim")
        return prompt

    # Shared model-name alternation (kept in sync with the current catalog).
    _models = (
        r"runway(?:[-\s]?(?:aleph|4\.?5))?|"
        r"veo[-\s]?3\.?1(?:[-\s]?(?:flash|ultra))?|"
        r"seedream|midjourney|nano[-\s]?banana(?:\s?2)?|gpt|"
        r"seedance[-\s]?2(?:\.0)?(?:[-\s]?fast)?|"
        r"kling[-\s]?(?:v?3[-\s]?turbo|v?3|o3)"
    )
    patterns = [
        # English: "with veo 3.1", "using kling v3", etc.
        r"\s+(?:with|using|via|through|by)\s+(?:" + _models + r")\s*$",
        # Spanish: "con veo 3.1", "usando kling o3", etc.
        r"\s+(?:con|usando|mediante|por)\s+(?:" + _models + r")\s*$",
        # Bare model name at end without preposition
        r"\s+(?:runway[-\s]?(?:aleph|4\.?5)|veo[-\s]?3\.?1[-\s]?(?:flash|ultra)|"
        r"kling[-\s]?(?:v?3[-\s]?turbo|v?3|o3))\s*$",
    ]

    cleaned = prompt
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = cleaned.rstrip(" ,.;:")
    logger.debug("Cleaned prompt from '%s' to '%s'", prompt, cleaned)
    return cleaned if cleaned else prompt


def _collect_image_reference_urls(
    model: str,
    context_files: Optional[list],
    reference_image: Optional[str],
    reference_images: Optional[list],
) -> Tuple[list, bool]:
    """
    Resolve the reference image URLs to send, honoring the backend's per-model
    rules. Session reference files are primary; explicit args are a fallback
    used only when the session has none.

    Returns (urls, only_blobs_dropped) where only_blobs_dropped is True when the
    ONLY references available were unusable browser blob: URLs (caller surfaces
    a friendly error in that case).
    """
    session_images = [
        f.get("url") for f in (context_files or [])
        if f.get("type") == "image" and f.get("url")
    ]
    explicit = list(reference_images or [])
    if reference_image:
        explicit.append(reference_image)

    candidates = session_images or explicit
    non_blob = [u for u in candidates if u and not is_blob_url(u)]
    dropped_blobs = [u for u in candidates if is_blob_url(u)]
    if dropped_blobs:
        logger.warning(
            "Filtered out %d blob: URLs that cannot be processed server-side",
            len(dropped_blobs),
        )

    usable = list(non_blob)
    # Midjourney img2img ignores inline data: URIs — only public URLs work.
    if model == "Midjourney":
        data_uris = [u for u in usable if is_data_uri(u)]
        if data_uris:
            logger.warning(
                "Dropped %d data: URI reference(s) — Midjourney requires public URLs",
                len(data_uris),
            )
        usable = [u for u in usable if not is_data_uri(u)]

    # Surface the blob error only when blobs were the ONLY references available.
    only_blobs = bool(dropped_blobs) and not non_blob
    return usable, only_blobs


async def generate_image(
    prompt: str,
    model: str = "Seedream",
    image_type: int = 1,
    quantity: int = 1,
    reference_image: Optional[str] = None,
    reference_images: Optional[list] = None,
    aspect_ratio: str = "16:9",
    quality: str = "2K",
) -> str:
    """
    Generate or edit an image using the reelmotion backend.
    Supports text-to-image (type 1), image-to-image (type 2), and multi-image reference (type 3).

    COST: Seedream = 4, GPT = 6, Nano Banana 2 = 7, Midjourney = 9 tokens per image.
    (Source of truth: pricing.py IMAGE_COSTS. There is NO Freepik model.)

    Seedream and Midjourney deliver asynchronously (hybrid): the call may return
    the finished image (200) or, for slow jobs, "still processing" (202) — in
    which case the user is notified when it's ready and we never retry. A failed
    job (422) is auto-refunded by the backend; insufficient balance is 402.
    """
    logger.debug("Tool 'generate_image' called with prompt='%s', model='%s'", prompt, model)

    # Defense-in-depth: refuse disallowed content even if it slipped past
    # the chatbot-layer moderation (e.g., a stale pending action).
    if is_disallowed_content(prompt):
        logger.warning("generate_image refused: disallowed prompt content")
        return get_refusal_message(prompt)

    # Clean model names passed by the LLM via reference args (we use session files instead)
    prompt = clean_prompt_from_model_mentions(prompt)

    # Validate and normalise model name
    canonical = normalize_image_model(model)
    if canonical is None:
        return f"Error: Invalid model '{model}'. Allowed models are: {', '.join(IMAGE_COSTS)}"
    model = canonical
    logger.debug("Normalised model to '%s'", model)

    backend_url = os.getenv("BACKEND_URL")
    endpoint = os.getenv("IMAGE_CREATION_ENDPOINT")
    api_token = get_api_token() or os.getenv("API_TOKEN")

    # Get reference files from session (no circular chatbot import)
    conversation_uuid = get_conversation_uuid() or "default"
    session_manager = get_session_manager()
    context_files = await session_manager.get_reference_files(conversation_uuid)
    logger.debug(
        "Retrieved %d reference files for uuid='%s'",
        len(context_files) if context_files else 0,
        conversation_uuid,
    )

    if not backend_url or not endpoint:
        return "Error: Backend URL or Image Creation Endpoint not configured."

    url = f"{backend_url}{endpoint}"
    logger.debug("Calling URL: %s", url)

    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    ref_urls, only_blobs = _collect_image_reference_urls(
        model, context_files, reference_image, reference_images
    )
    if only_blobs:
        return (
            "Error: The reference image could not be processed because it uses a temporary "
            "browser URL (blob:). Please try uploading the image again or use a direct image URL."
        )

    payload = {
        "prompt": prompt,
        "model": model,
        "aspect_ratio": aspect_ratio,
    }
    # `quality` (2K/3K) only applies to Seedream; ignored by other models.
    if model == "Seedream":
        payload["quality"] = quality
    # `type`/`quantity` only apply to the models that honor multi-image; Seedream
    # and Midjourney always generate exactly one image per call.
    if model in QUANTITY_IMAGE_MODELS:
        payload["type"] = image_type
        payload["quantity"] = max(1, quantity)

    if len(ref_urls) == 1:
        payload["reference_image"] = ref_urls[0]
    elif len(ref_urls) > 1:
        payload["reference_images"] = ref_urls

    logger.debug("Sending image request with %d reference URL(s)", len(ref_urls))

    try:
        timeout = httpx.Timeout(180.0, connect=10.0)
        # AsyncClient: a sync client here would block the event loop for every
        # other conversation while the backend generates.
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            logger.debug("Response status: %d", response.status_code)

            # Branch on the HTTP STATUS, never the model name (mirrors video):
            #   202 -> accepted but not finished in the sync window; the user is
            #          notified when ready — we do NOT poll and do NOT retry.
            #   200 -> the finished image(s) are in the response body.
            #   else (402/422/401/5xx/...) -> raise for the typed handlers below.
            if response.status_code == 202:
                logger.debug("Backend returned 202: image will be delivered asynchronously")
                return format_generation_processing("image")

            response.raise_for_status()
            result = response.json()

        logger.debug("Backend response: %s", result)

        # Save only the PRIMARY image; Midjourney's extra variants
        # (all_result_urls) are intentionally not shown unless asked for.
        image_urls = _extract_image_urls(result)
        if not image_urls:
            return "Image generation initiated but URL not immediately available. Check status later."

        await session_manager.save_generated_file(conversation_uuid, image_urls[0], "image")
        await session_manager.clear_reference_files(conversation_uuid)
        logger.debug("Reference files cleared after image generation")
        return f"Images generated successfully with {model}."

    except httpx.HTTPStatusError as e:
        parsed = parse_backend_error(e.response.status_code, e.response.text)
        detail = parsed["detail"]
        # A 422 means the provider failed AFTER accepting the job; the backend
        # auto-refunds (refunded: true), so reassure the user they weren't charged.
        if e.response.status_code == 422:
            try:
                body = e.response.json()
                if isinstance(body, dict) and body.get("refunded"):
                    detail += " Your tokens were automatically refunded, so you were not charged."
            except (ValueError, TypeError):
                pass
        logger.error("Error generating image (HTTP %d): %s", parsed["status"], detail)
        return format_generation_error("image", parsed["category"], parsed["status"], detail)
    except httpx.TimeoutException:
        logger.error("Error generating image: timeout after 180s")
        return format_generation_error(
            "image", CATEGORY_TIMEOUT, 0,
            "The image service did not respond within 180 seconds.",
        )
    except httpx.HTTPError as e:
        logger.error("Error generating image (network): %s", e)
        return format_generation_error(
            "image", CATEGORY_BACKEND_UNAVAILABLE, 0,
            f"Could not reach the image service: {e}",
        )
    except Exception as e:
        logger.error("Error generating image: %s", e)
        return format_generation_error("image", CATEGORY_UNKNOWN, 0, str(e))


# ---------------------------------------------------------------------------
# Seedance 2.0 helpers
# ---------------------------------------------------------------------------
# Pricing tables and cost helpers live in pricing.py (single source of truth).


def _build_seedance_media(
    payload: dict,
    context_files: Optional[list],
    reference_images: Optional[list],
    reference_videos: Optional[list],
    reference_audios: Optional[list],
    media_url: Optional[str],
    end_frame: Optional[str],
) -> bool:
    """
    Populate a Seedance payload with the right media fields and let the backend
    autodetect the sub-mode:
      - reference_images / reference_videos / reference_audios -> reference mode
      - media_url -> image mode
      - none -> text mode
    Explicit args take precedence over session reference files.
    Returns True if reference_videos were attached (discounted rate applies).
    """
    session_images: list = []
    session_videos: list = []
    if context_files:
        valid = [f for f in context_files if not is_blob_url(f.get("url", ""))]
        session_images = [f["url"] for f in valid if f.get("type") == "image"]
        session_videos = [f["url"] for f in valid if f.get("type") == "video"]

    video_urls = list(reference_videos) if reference_videos else session_videos
    image_urls = list(reference_images) if reference_images else session_images
    if media_url:
        image_urls = [media_url] + [u for u in image_urls if u != media_url]

    if video_urls:
        # Reference mode (discounted): @Image dance like @Video, style transfer, etc.
        payload["reference_videos"] = video_urls[:3]
        if image_urls:
            payload["reference_images"] = image_urls[:9]
        if reference_audios:
            payload["reference_audios"] = list(reference_audios)[:3]
        return True

    if image_urls:
        # Image mode: animate a single frame (end_frame optional).
        payload["media_url"] = image_urls[0]
        if end_frame:
            payload["end_frame"] = end_frame
        return False

    # Text mode: prompt only.
    return False


def _build_kling_media(
    model: str,
    mode: Optional[str],
    media_url: Optional[str],
    reference_image: Optional[str],
    reference_images: Optional[list],
    reference_video: Optional[str],
    reference_videos: Optional[list],
    motion_video: Optional[str],
    edit_video: Optional[str],
    context_files: Optional[list],
) -> Tuple[str, dict]:
    """
    Resolve the route and media fields for a Kling v3 / v3-turbo / o3 request,
    following the Evolink routing rules:
      - text->video: prompt only (no media)
      - image->video: media_url = an image
      - motion-control (kling-v3): motion_video = a guide VIDEO; appearance
        images go in reference_images/media_url (NEVER the guide video in media_url)
      - reference-to-video (kling-o3): reference_images for character/style consistency
      - video-edit (kling-o3): edit_video = the source video

    Explicit args win; session reference files are the fallback. A single
    attached image is image-to-video (cheaper base route), NOT reference mode —
    reference mode is only entered via an explicit mode/reference_images.
    Returns (route, media_payload).
    """
    session_images, session_videos = [], []
    for f in (context_files or []):
        u = f.get("url", "")
        if not u or is_blob_url(u):
            continue
        if f.get("type") == "image":
            session_images.append(u)
        elif f.get("type") == "video":
            session_videos.append(u)

    media_is_video = _looks_like_video_url(media_url)

    # Explicit "consistency" reference images (reference route).
    explicit_ref_images = [u for u in (reference_images or []) if u and not is_blob_url(u)]

    # The single input frame used for image-to-video (base route).
    i2v_candidates = []
    if reference_image and not is_blob_url(reference_image):
        i2v_candidates.append(reference_image)
    if media_url and not media_is_video:
        i2v_candidates.append(media_url)
    i2v_candidates.extend(session_images)
    i2v_image = i2v_candidates[0] if i2v_candidates else None

    # The input video used for motion/edit routes.
    video_candidates = []
    if reference_video and not is_blob_url(reference_video):
        video_candidates.append(reference_video)
    video_candidates.extend([u for u in (reference_videos or []) if u and not is_blob_url(u)])
    if media_url and media_is_video:
        video_candidates.append(media_url)
    video_candidates.extend(session_videos)
    input_video = video_candidates[0] if video_candidates else None

    mode = (mode or "").lower().strip() or None
    payload: dict = {}

    if model == "kling-v3-turbo":
        # text/image-to-video only — a video input has no route here.
        if i2v_image:
            payload["media_url"] = i2v_image
        return KLING_ROUTE_TURBO, payload

    if model == "kling-o3":
        edit_src = edit_video if (edit_video and not is_blob_url(edit_video)) else input_video
        if mode == "edit" or (mode is None and edit_src):
            if edit_src:
                payload["edit_video"] = edit_src
            return KLING_ROUTE_EDIT, payload
        if mode == "reference" or explicit_ref_images:
            if explicit_ref_images:
                payload["reference_images"] = explicit_ref_images[:7]
            elif i2v_image:
                payload["media_url"] = i2v_image
            return KLING_ROUTE_REFERENCE, payload
        if i2v_image:
            payload["media_url"] = i2v_image
        return KLING_ROUTE_BASE, payload

    # kling-v3
    motion_src = motion_video if (motion_video and not is_blob_url(motion_video)) else input_video
    if mode == "motion" or (mode is None and motion_src):
        if motion_src:
            payload["motion_video"] = motion_src
            appearance = explicit_ref_images or ([i2v_image] if i2v_image else [])
            if appearance:
                payload["reference_images"] = appearance[:7]
            return KLING_ROUTE_MOTION, payload
    if i2v_image:
        payload["media_url"] = i2v_image
    return KLING_ROUTE_BASE, payload


async def generate_video(
    prompt: str,
    model: str,
    duration: int,
    aspect_ratio: str = "16:9",
    reference_image: Optional[str] = None,
    reference_video: Optional[str] = None,
    resolution: str = "720p",
    generate_audio: bool = True,
    seed: Optional[int] = None,
    media_url: Optional[str] = None,
    end_frame: Optional[str] = None,
    reference_images: Optional[list] = None,
    reference_videos: Optional[list] = None,
    reference_audios: Optional[list] = None,
    quality: Optional[str] = None,
    sound: Optional[str] = None,
    mode: Optional[str] = None,
    keep_sound: bool = False,
    motion_video: Optional[str] = None,
    edit_video: Optional[str] = None,
) -> str:
    """
    Generate or edit a video using AI based on a text prompt.
    Supports text-to-video, image-to-video, and video-to-video editing.
    Sends `provider` + params to the backend (/api/ai/mcp-video-generation).

    Token costs per second (source of truth: pricing.py VIDEO_TOKEN_RATES):
    - runway-aleph: 17 tokens/sec (5-10s)
    - runway-4.5: 13 tokens/sec (5, 8, or 10s) — hybrid: backend waits synchronously
      (up to ~590s) and returns the finished video (200); only very long jobs spill
      to 202 (still processing, async notification)
    - veo-3.1: 44 tokens/sec (8s only)
    - veo-3.1-flash: 17 tokens/sec (8s only)
    - veo-3.1-ultra: 65 tokens/sec (8s only)

    Kling v3 / o3 (Evolink) — resolution + route + audio based pricing, 3-15s
    (reference route 3-10s), tokens/sec:
    - kling-v3 / kling-o3 text or image: 720p=9, 1080p=12, 4k=42 (+audio 720p=12, 1080p=14)
    - kling-v3-turbo: 720p=12, 1080p=14 (max 1080p, no audio)
    - kling-o3 reference / video-edit: 720p=13, 1080p=17 (max 1080p)
    - kling-v3 motion-control: 720p=13, 1080p=17 (provisional)
    Kling routing: motion_video -> motion (kling-v3); edit_video / a video media ->
    video-edit (kling-o3); reference_images / mode='reference' -> reference (kling-o3);
    otherwise text/image-to-video. `quality` (720p/1080p/4k, default 720p) and
    `sound` ('on'/'off', base route only) are clamped by the backend.

    Seedance 2.0 (resolution-based pricing, 4-15s, default 5s):
    - seedance-2.0: 480p=15, 720p=32, 1080p=72 tokens/sec
    - seedance-2.0-fast: 480p=12, 720p=26 tokens/sec (no 1080p -> downgraded to 720p)
    - Reference-video discount (reference_videos sent): seedance-2.0 480p=9/720p=20/1080p=43;
      seedance-2.0-fast 480p=7/720p=16.
    Seedance-only params: resolution, generate_audio, seed, media_url + end_frame
    (image mode), reference_images/reference_videos/reference_audios (reference mode).

    Kling-only params: quality, sound, mode ('motion'/'reference'/'edit'),
    keep_sound, motion_video, edit_video.
    """
    logger.debug("Tool 'generate_video' called with prompt='%s', model='%s', duration=%d", prompt, model, duration)

    # Defense-in-depth: refuse disallowed content even if it slipped past
    # the chatbot-layer moderation (e.g., a stale pending action).
    if is_disallowed_content(prompt):
        logger.warning("generate_video refused: disallowed prompt content")
        return get_refusal_message(prompt)

    prompt = clean_prompt_from_model_mentions(prompt)
    is_seedance = model in SEEDANCE2_MODELS
    is_kling = model in KLING_MODELS
    # Seedance/Kling accept an empty/"auto" duration (normalized to a default);
    # other models expect an explicit integer.
    if is_seedance:
        duration = normalize_seedance_duration(duration)
    elif is_kling:
        duration = int(duration) if str(duration).strip().isdigit() else 5
    else:
        duration = int(duration)

    # Valid providers the backend accepts. The old kling-v1 / kling-v3-omni-*
    # keys no longer exist (they return 400).
    allowed_models = [
        "runway-aleph", "runway-4.5",
        "veo-3.1", "veo-3.1-flash", "veo-3.1-ultra",
        "kling-v3", "kling-v3-turbo", "kling-o3",
        "seedance-2.0", "seedance-2.0-fast",
    ]

    if model not in allowed_models:
        return f"Error: Invalid model '{model}'. Allowed models are: {', '.join(allowed_models)}"

    # Valid durations per model live in pricing.py (single source of truth).
    if model in VIDEO_DURATION_RULES and duration not in VIDEO_DURATION_RULES[model]:
        return (
            f"Error: Duration {duration}s is not valid for model '{model}'. "
            f"Allowed durations: {VIDEO_DURATION_RULES[model]} seconds."
        )

    logger.debug("Duration %ds validated for model %s", duration, model)

    backend_url = os.getenv("BACKEND_URL")
    endpoint = os.getenv("VIDEO_CREATION_ENDPOINT")
    api_token = get_api_token() or os.getenv("API_TOKEN")

    logger.debug("backend_url=%s, endpoint=%s", backend_url, endpoint)

    if not backend_url or not endpoint:
        return f"Error: Backend configuration missing. BACKEND_URL={backend_url}, VIDEO_CREATION_ENDPOINT={endpoint}"

    backend_url = backend_url.rstrip("/")
    endpoint = endpoint.lstrip("/")
    url = f"{backend_url}/{endpoint}"
    logger.debug("Full URL: %s", url)

    # Get reference files from session (no circular chatbot import)
    conversation_uuid = get_conversation_uuid() or "default"
    session_manager = get_session_manager()
    context_files = await session_manager.get_reference_files(conversation_uuid)
    logger.debug(
        "Retrieved %d reference files for uuid='%s'",
        len(context_files) if context_files else 0,
        conversation_uuid,
    )

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    payload = {
        "prompt": prompt,
        "provider": model,
        "video_duration": duration,
        "aspect_ratio": aspect_ratio,
    }

    if is_seedance:
        # Resolution-based tier; the "fast" sub-tier has no 1080p.
        res = normalize_seedance_resolution(model, resolution)
        if (resolution or "").lower().strip() == "1080p" and res != "1080p":
            logger.debug("%s does not support 1080p; using %s instead", model, res)
        if aspect_ratio not in SEEDANCE2_VALID_ASPECT_RATIOS:
            logger.debug("Aspect ratio '%s' invalid for Seedance; using 16:9", aspect_ratio)
            payload["aspect_ratio"] = "16:9"
        payload["resolution"] = res
        payload["generate_audio"] = generate_audio
        if seed is not None:
            payload["seed"] = seed

        # Blob-URL guard mirrors the non-seedance path (only error if blobs were
        # the ONLY media and no explicit URLs were provided).
        if context_files:
            valid_context_files = [f for f in context_files if not is_blob_url(f.get("url", ""))]
            blob_files = [f for f in context_files if is_blob_url(f.get("url", ""))]
            if blob_files:
                logger.warning("Filtered out %d blob: URLs from reference files", len(blob_files))
            has_explicit = bool(media_url or reference_videos or reference_images)
            if not valid_context_files and blob_files and not has_explicit:
                return (
                    "Error: The reference file could not be processed because it uses a temporary "
                    "browser URL (blob:). Please try uploading the file again or use a direct URL."
                )

        # Bridge legacy singular args (set by the chatbot's pending actions) into
        # the Seedance media model so image/reference modes still trigger even if
        # the session reference files were somehow unavailable.
        effective_media_url = media_url or reference_image
        effective_ref_videos = reference_videos
        if not effective_ref_videos and reference_video:
            effective_ref_videos = [reference_video]

        used_ref_videos = _build_seedance_media(
            payload, context_files, reference_images, effective_ref_videos,
            reference_audios, effective_media_url, end_frame,
        )
        cost = compute_seedance2_cost(model, res, duration, has_reference_videos=used_ref_videos)
        logger.debug(
            "Seedance payload: model=%s, resolution=%s, duration=%ds, ref_videos=%s, cost=%d tokens",
            model, res, duration, used_ref_videos, cost,
        )
    elif is_kling:
        # Blob-URL guard: only error if blobs were the ONLY media available.
        if context_files:
            valid_context_files = [f for f in context_files if not is_blob_url(f.get("url", ""))]
            blob_files = [f for f in context_files if is_blob_url(f.get("url", ""))]
            if blob_files:
                logger.warning("Filtered out %d blob: URLs from reference files", len(blob_files))
            has_explicit = bool(
                media_url or reference_image or reference_images or reference_video
                or reference_videos or motion_video or edit_video
            )
            if not valid_context_files and blob_files and not has_explicit:
                return (
                    "Error: The reference file could not be processed because it uses a temporary "
                    "browser URL (blob:). Please try uploading the file again or use a direct URL."
                )

        route, media_payload = _build_kling_media(
            model, mode, media_url, reference_image, reference_images,
            reference_video, reference_videos, motion_video, edit_video, context_files,
        )
        payload.update(media_payload)
        # Resolution/quality — 4K only on the base (text/image) route; the
        # backend trims further. `resolution` is the value the workflow collects.
        q = normalize_kling_quality(model, route, quality or resolution)
        payload["quality"] = q
        # Audio is opt-in (default off) and only effective on the base route.
        sound_on = (str(sound or "off").lower() == "on") and route == KLING_ROUTE_BASE
        payload["sound"] = "on" if sound_on else "off"
        if keep_sound and route in (KLING_ROUTE_MOTION, KLING_ROUTE_REFERENCE, KLING_ROUTE_EDIT):
            payload["keep_sound"] = True
        cost = compute_kling_cost(model, route, q, duration, sound_on)
        logger.debug(
            "Kling payload: model=%s route=%s quality=%s duration=%ds sound=%s cost=%d tokens",
            model, route, q, duration, sound_on, cost,
        )
    elif context_files:
        valid_context_files = [f for f in context_files if not is_blob_url(f.get("url", ""))]
        blob_files = [f for f in context_files if is_blob_url(f.get("url", ""))]
        if blob_files:
            logger.warning("Filtered out %d blob: URLs from reference files", len(blob_files))

        if not valid_context_files and blob_files:
            return (
                "Error: The reference file could not be processed because it uses a temporary "
                "browser URL (blob:). Please try uploading the file again or use a direct URL."
            )

        image_file = next((f for f in valid_context_files if f.get("type") == "image"), None)
        video_file = next((f for f in valid_context_files if f.get("type") == "video"), None)

        processed_video = False
        if video_file:
            if model == "runway-aleph":
                payload["reference_video"] = video_file["url"]
                logger.debug("Using reference video URL (runway-aleph): %s", video_file["url"])
                processed_video = True

        if not processed_video and image_file:
            payload["reference_image"] = image_file["url"]
            logger.debug("Using reference image URL: %s", image_file["url"])

    logger.debug(
        "Sending video generation request: model=%s, duration=%ds, aspect=%s",
        model, duration, aspect_ratio,
    )
    logger.debug("Payload: %s", json.dumps(payload, indent=2))

    try:
        # 900s read ceiling. The hybrid backend (runway-4.5) now holds the
        # request SYNCHRONOUSLY for up to ~590s (≈10 min) and answers 200 with
        # the finished video in the normal case; only very long generations spill
        # past that window and come back 202 (still processing). The read ceiling
        # MUST stay above that window (≥600s) so the agent never cuts off before
        # the backend can return the video_url — otherwise a slow-but-successful
        # generation surfaces as a connection error and the user never sees the
        # video. The other providers are still fully synchronous and can also take
        # minutes, so the high ceiling serves them too. AsyncClient keeps the
        # event loop free for other conversations.
        timeout = httpx.Timeout(900.0, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            logger.debug("Response status: %d", response.status_code)

            # Branch on the HTTP STATUS, never the provider name, so any provider
            # the backend migrates to the hybrid model later works here unchanged.
            #   202 -> accepted but not finished in the sync window; the user is
            #          notified (push/realtime) when it's ready — we do NOT poll.
            #   200 -> the finished video is already in the response body.
            #   else (402/422/401/5xx/...) -> raise and let the typed handlers
            #          below classify and explain it.
            if response.status_code == 202:
                logger.debug("Backend returned 202: video will be delivered asynchronously")
                return format_generation_processing("video")

            response.raise_for_status()
            result = response.json()

        logger.debug("Backend response: %s", result)

        # Never assume video_url is present — in the hybrid model it only comes
        # back on the synchronous (200) path.
        video_url = result.get("video_url") or result.get("result_url")
        if video_url:
            await session_manager.save_generated_file(conversation_uuid, video_url, "video")
            await session_manager.clear_reference_files(conversation_uuid)
            logger.debug("Reference files cleared after video generation")
            return f"Video generated successfully with {model}."
        else:
            return "Video generation initiated but URL not immediately available. Check status later."

    except httpx.HTTPStatusError as e:
        parsed = parse_backend_error(e.response.status_code, e.response.text)
        detail = parsed["detail"]
        # A 422 from the hybrid backend means the provider failed AFTER accepting
        # the job; the backend auto-refunds (refunded: true), so reassure the user
        # they were not charged.
        if e.response.status_code == 422:
            try:
                body = e.response.json()
                if isinstance(body, dict) and body.get("refunded"):
                    detail += " Your tokens were automatically refunded, so you were not charged."
            except (ValueError, TypeError):
                pass
        logger.error("Error generating video (HTTP %d): %s", parsed["status"], detail)
        return format_generation_error("video", parsed["category"], parsed["status"], detail)
    except httpx.TimeoutException:
        logger.error("Error generating video: request timeout after 900s")
        return format_generation_error(
            "video", CATEGORY_TIMEOUT, 0,
            "The video service did not respond within 900 seconds (15 min). "
            "The generation may still complete on the backend.",
        )
    except httpx.HTTPError as e:
        logger.error("Error generating video (network): %s", e)
        return format_generation_error(
            "video", CATEGORY_BACKEND_UNAVAILABLE, 0,
            f"Could not reach the video service: {e}",
        )
    except Exception as e:
        import traceback
        logger.error("Error generating video: %s\n%s", e, traceback.format_exc())
        return format_generation_error("video", CATEGORY_UNKNOWN, 0, f"{type(e).__name__}: {str(e)}")


async def generate_speech(
    text: str,
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    model_id: str = "eleven_multilingual_v2",
) -> str:
    """
    Generate speech/audio from text using the ElevenLabs API.

    COST: 1-500 characters = 1 token, 501-999 characters = 8 tokens,
          1000+ characters = 13 tokens per 1000 chars (rounded up).
    (Source of truth: pricing.py speech_cost().)
    """
    import base64
    import uuid as uuid_lib

    # Defense-in-depth: refuse disallowed text content for TTS too.
    if is_disallowed_content(text):
        logger.warning("generate_speech refused: disallowed text content")
        return get_refusal_message(text)

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return "Error: ELEVENLABS_API_KEY environment variable is not set."

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key,
    }

    data = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5,
        },
    }

    logger.debug("Generating speech for text: '%s...' with voice %s", text[:50], voice_id)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data, headers=headers)
            response.raise_for_status()
            audio_content = response.content

        audio_base64 = base64.b64encode(audio_content).decode("utf-8")
        audio_data_uri = f"data:audio/mpeg;base64,{audio_base64}"

        conversation_uuid = get_conversation_uuid() or "default"
        session_manager = get_session_manager()

        logger.debug("Speech generated: %d bytes", len(audio_content))
        await session_manager.save_generated_file(conversation_uuid, audio_data_uri, "audio")

        # Backend callback to register token usage
        backend_url = os.getenv("BACKEND_URL")
        api_token = get_api_token() or os.getenv("API_TOKEN")

        logger.debug(
            "backend_url=%s, api_token=%s",
            backend_url,
            "SET" if api_token else "NOT SET",
        )

        if backend_url and api_token:
            backend_url = backend_url.rstrip("/")
            callback_url = f"{backend_url}/api/ai/mcp-voice-generation"
            # Real tiered cost (was hardcoded to 5, under/over-billing every TTS)
            tokens_cost = speech_cost(len(text))

            # Backend requires an http/https URL — use a placeholder for the Data URI
            req_uuid = str(uuid_lib.uuid4())
            backend_audio_url = f"https://reelmotion.ai/generated/audio/{req_uuid}.mp3"

            callback_payload = {"audio_url": backend_audio_url, "tokens": tokens_cost}
            callback_headers = {
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            async with httpx.AsyncClient(timeout=30.0) as callback_client:
                try:
                    cb_response = await callback_client.post(
                        callback_url, json=callback_payload, headers=callback_headers
                    )
                    if 200 <= cb_response.status_code < 300:
                        logger.debug("Backend callback SUCCESS: %s", cb_response.text[:200])
                    else:
                        logger.warning(
                            "Backend callback FAILED: %d - %s",
                            cb_response.status_code,
                            cb_response.text,
                        )
                except Exception as cb_exc:
                    logger.error("Exception calling backend callback: %s", cb_exc)
        else:
            logger.warning(
                "Skipping backend callback - backend_url=%s, api_token=%s",
                backend_url,
                "SET" if api_token else "NOT SET",
            )

        return f"Audio generated successfully ({len(audio_content)} bytes). Link generated automatically."

    except httpx.HTTPStatusError as e:
        parsed = parse_backend_error(e.response.status_code, e.response.text)
        logger.error("Error generating speech (HTTP %d): %s", parsed["status"], parsed["detail"])
        return format_generation_error("speech", parsed["category"], parsed["status"], parsed["detail"])
    except httpx.TimeoutException:
        logger.error("Error generating speech: timeout after 30s")
        return format_generation_error(
            "speech", CATEGORY_TIMEOUT, 0,
            "The speech service did not respond within 30 seconds.",
        )
    except httpx.HTTPError as e:
        logger.error("Error generating speech (network): %s", e)
        return format_generation_error(
            "speech", CATEGORY_BACKEND_UNAVAILABLE, 0,
            f"Could not reach the speech service: {e}",
        )
    except Exception as e:
        logger.error("Error generating speech: %s", e)
        return format_generation_error("speech", CATEGORY_UNKNOWN, 0, str(e))


# ---------------------------------------------------------------------------
# Prompt crafting best-practices system instruction (shared constant)
# ---------------------------------------------------------------------------
_CRAFT_PROMPT_SYSTEM = """You are an expert AI prompt engineer specializing in image and video generation.
Your ONLY job is to help the user craft a detailed, production-ready prompt.

STRICT RULES:
- NEVER invent or assume details the user has not provided. Ask instead.
- Present ONLY the refined prompt and targeted questions/suggestions.
- DO NOT add any preamble, greetings, or unrelated commentary.
- Do NOT generate any image or video yourself.

PROCESS:
1. Analyze the user's raw idea and identify what is clear vs. what is missing.
2. List the missing aspects that would most improve the result (pick the 2-3 most important ones).
3. For each missing aspect offer 2-4 concrete options for the user to choose from.
4. If the user has already provided enough detail, output the refined prompt directly.

ASPECTS TO COVER:
For IMAGES:
  - Subject & action (what/who, doing what?)
  - Style & medium (photorealistic, illustration, watercolor, oil painting, 3D render, concept art)
  - Mood & atmosphere (dramatic, serene, mysterious, vibrant, dark, warm, nostalgic)
  - Lighting (golden hour, studio, neon glow, soft natural, harsh contrast, rim light)
  - Composition (portrait close-up, wide landscape, bird's-eye view, low angle, Dutch angle)
  - Color palette (warm tones, cool blues, monochrome, pastel, vivid & saturated)
  - Extra details (background, textures, era, props)

For VIDEOS (all of the above, plus):
  - Camera movement (static, slow pan, zoom in/out, dolly forward, handheld, orbiting)
  - Subject movement & pacing (slow motion, fast action, gentle idle, walking)
  - Scene open & close (fade in, hard cut in, motion blur out, freeze frame)

JSON PROMPT FORMAT (advanced, for VIDEO models — especially Veo):
Structured JSON prompts give video models finer control than prose. Offer this
format when the user targets a video model and wants precise control, or when
the user already pasted a JSON prompt. The template:
```json
{
  "scene": "where and when the action happens",
  "subject": "who/what the video is about",
  "action": "what happens during the clip",
  "camera": {"movement": "dolly/pan/orbit/static", "angle": "low/high/eye-level", "lens": "wide/tele/macro"},
  "lighting": "light sources, mood, time of day",
  "style": "cinematic/anime/documentary/etc.",
  "audio": {"music": "...", "sfx": "...", "dialogue": "..."},
  "duration": "8s"
}
```
Rules for JSON prompts:
- If the user supplies a JSON prompt, improve it AS JSON: suggest missing keys,
  never silently alter the values they wrote — propose changes and ask.
- Always output the complete JSON object inside a ```json fenced block.
- The JSON is sent to the generator character-for-character.

OUTPUT FORMAT when enough detail is gathered:
✨ Refined Prompt:
"[Full refined prompt, rich in detail and optimized for the target model]"
(or, for JSON prompts, the complete object in a ```json fenced block after the ✨ marker)

Then ask: "Would you like to adjust anything, or are you ready to generate?"
"""


async def craft_prompt(
    idea: str,
    media_type: str = "image",
    user_answers: str = "",
    output_format: str = "text",
) -> str:
    """
    Refine and improve a raw idea into a production-ready prompt for AI image or video generation.

    This tool asks targeted questions when details are missing and suggests concrete options
    for the user to choose from. It NEVER invents details — it only guides and refines.

    Args:
        idea: The user's raw description or idea for the image/video.
        media_type: "image" or "video". Defaults to "image".
        user_answers: Optional answers the user has already provided to follow-up questions.
        output_format: "text" (prose prompt) or "json" (Veo 3 style structured JSON prompt).
    """
    import google.generativeai as genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY environment variable is not set."

    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=_CRAFT_PROMPT_SYSTEM,
    )

    user_message = f"Media type: {media_type}\nIdea: {idea}"
    if user_answers.strip():
        user_message += f"\nAdditional details provided: {user_answers}"
    if output_format == "json":
        user_message += (
            "\nOutput format: JSON prompt (use the JSON PROMPT FORMAT template, "
            "output the complete object in a ```json fenced block)."
        )

    try:
        response = await model.generate_content_async(user_message)
        return response.text
    except Exception as e:
        logger.error("Error in craft_prompt: %s", e)
        return f"Error crafting prompt: {str(e)}"
