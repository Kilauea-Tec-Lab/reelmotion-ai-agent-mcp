import os
import httpx
import json
import logging
import re
from typing import Optional

from request_context import get_api_token, get_conversation_uuid
from session_manager import get_session_manager
from logging_config import setup_logging
from moderation import is_disallowed_content, get_refusal_message
from pricing import (
    SEEDANCE2_MODELS,
    SEEDANCE2_TOKEN_RATES,
    SEEDANCE2_VALID_ASPECT_RATIOS,
    VIDEO_DURATION_RULES,
    normalize_seedance_resolution,
    normalize_seedance_duration,
    compute_seedance2_cost,
    speech_cost,
)
from generation_errors import (
    CATEGORY_BACKEND_UNAVAILABLE,
    CATEGORY_TIMEOUT,
    CATEGORY_UNKNOWN,
    format_generation_error,
    parse_backend_error,
)

setup_logging()
logger = logging.getLogger(__name__)


def is_blob_url(url: str) -> bool:
    """Check if a URL is a browser-only blob: URL that cannot be fetched server-side."""
    return isinstance(url, str) and url.startswith("blob:")


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

    patterns = [
        # English: "with veo 3.1", "using runway aleph", etc.
        r"\s+(?:with|using|via|through|by)\s+(?:runway(?:[-\s]?(?:aleph|4\.?5))?|veo[-\s]?3\.?1(?:[-\s]?(?:flash|ultra))?|nano[-\s]?banana|gpt|freepik|luma[-\s]?labs?|seedance[-\s]?pro|kling[-\s]?(?:v?3[-\s]?omni[-\s]?(?:pro|std)|v1))\s*$",
        # Spanish: "con veo 3.1", "usando runway aleph", etc.
        r"\s+(?:con|usando|mediante|por)\s+(?:runway(?:[-\s]?(?:aleph|4\.?5))?|veo[-\s]?3\.?1(?:[-\s]?(?:flash|ultra))?|nano[-\s]?banana|gpt|freepik|luma[-\s]?labs?|seedance[-\s]?pro|kling[-\s]?(?:v?3[-\s]?omni[-\s]?(?:pro|std)|v1))\s*$",
        # Bare model name at end without preposition
        r"\s+(?:runway[-\s]?(?:aleph|4\.?5)|veo[-\s]?3\.?1[-\s]?(?:flash|ultra)|kling[-\s]?v?3[-\s]?omni[-\s]?(?:pro|std))\s*$",
    ]

    cleaned = prompt
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = cleaned.rstrip(" ,.;:")
    logger.debug("Cleaned prompt from '%s' to '%s'", prompt, cleaned)
    return cleaned if cleaned else prompt


async def generate_image(
    prompt: str,
    model: str = "GPT",
    image_type: int = 1,
    quantity: int = 1,
    reference_image: Optional[str] = None,
    reference_images: Optional[list] = None,
) -> str:
    """
    Generate or edit an image using the reelmotion backend.
    Supports text-to-image (type 1), image-to-image (type 2), and multi-image reference (type 3).

    COST: Nano Banana 2 = 7 tokens, GPT = 6 tokens, Freepik = 1 token per image.
    (Source of truth: pricing.py IMAGE_COSTS.)
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
    allowed_models = ["Nano Banana 2", "GPT", "Freepik"]
    if model not in allowed_models:
        if "nano" in model.lower():
            model = "Nano Banana 2"
        elif "gpt" in model.lower():
            model = "GPT"
        elif "freepik" in model.lower():
            model = "Freepik"
        else:
            return f"Error: Invalid model '{model}'. Allowed models are: {', '.join(allowed_models)}"

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

    headers = {"Accept": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    if context_files:
        image_files = [f for f in context_files if f.get("type") == "image"]
        valid_image_files = [f for f in image_files if not is_blob_url(f.get("url", ""))]
        blob_image_files = [f for f in image_files if is_blob_url(f.get("url", ""))]

        if blob_image_files:
            logger.warning(
                "Filtered out %d blob: URLs that cannot be processed server-side",
                len(blob_image_files),
            )

        if not valid_image_files and blob_image_files:
            return (
                "Error: The reference image could not be processed because it uses a temporary "
                "browser URL (blob:). Please try uploading the image again or use a direct image URL."
            )

        image_files = valid_image_files
        logger.debug("Sending request with %d valid image URLs", len(image_files))
        headers["Content-Type"] = "application/json"

        payload = {
            "prompt": prompt,
            "model": model,
            "type": image_type,
            "quantity": quantity,
        }

        if len(image_files) == 1:
            payload["reference_image"] = image_files[0]["url"]
        elif len(image_files) > 1:
            payload["reference_images"] = [f["url"] for f in image_files]

        try:
            timeout = httpx.Timeout(180.0, connect=10.0)
            # AsyncClient: a sync client here would block the event loop for
            # every other conversation while the backend generates.
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()

            logger.debug("Backend response: %s", result)

            images_data = result.get("images") or result.get("data")
            if images_data:
                if isinstance(images_data, list):
                    for img in images_data:
                        if isinstance(img, dict) and "url" in img:
                            await session_manager.save_generated_file(
                                conversation_uuid, img["url"], "image"
                            )
                elif isinstance(images_data, dict) and "url" in images_data:
                    await session_manager.save_generated_file(
                        conversation_uuid, images_data["url"], "image"
                    )

            await session_manager.clear_reference_files(conversation_uuid)
            logger.debug("Reference files cleared after use")

            return f"Images generated successfully with {model}."
        except httpx.HTTPStatusError as e:
            parsed = parse_backend_error(e.response.status_code, e.response.text)
            logger.error(
                "Error generating image (HTTP %d, with context images): %s",
                parsed["status"], parsed["detail"],
            )
            return format_generation_error("image", parsed["category"], parsed["status"], parsed["detail"])
        except httpx.TimeoutException:
            logger.error("Error generating image (with context images): timeout after 180s")
            return format_generation_error(
                "image", CATEGORY_TIMEOUT, 0,
                "The image service did not respond within 180 seconds.",
            )
        except httpx.HTTPError as e:
            logger.error("Error generating image (network, with context images): %s", e)
            return format_generation_error(
                "image", CATEGORY_BACKEND_UNAVAILABLE, 0,
                f"Could not reach the image service: {e}",
            )
        except Exception as e:
            logger.error("Error generating image (with context images): %s", e)
            return format_generation_error("image", CATEGORY_UNKNOWN, 0, str(e))

    else:
        # Text-only generation — no reference images
        logger.debug("Sending JSON request (text-only, no images)")
        headers["Content-Type"] = "application/json"

        payload = {
            "prompt": prompt,
            "model": model,
            "type": image_type,
            "quantity": quantity,
        }

        try:
            timeout = httpx.Timeout(180.0, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()

            logger.debug("Backend response: %s", result)

            images_data = result.get("images") or result.get("data")
            if images_data:
                if isinstance(images_data, list):
                    for img in images_data:
                        if isinstance(img, dict) and "url" in img:
                            await session_manager.save_generated_file(
                                conversation_uuid, img["url"], "image"
                            )
                elif isinstance(images_data, dict) and "url" in images_data:
                    await session_manager.save_generated_file(
                        conversation_uuid, images_data["url"], "image"
                    )

            return f"Images generated successfully with {model}."
        except httpx.HTTPStatusError as e:
            parsed = parse_backend_error(e.response.status_code, e.response.text)
            logger.error(
                "Error generating image (HTTP %d, text-only): %s",
                parsed["status"], parsed["detail"],
            )
            return format_generation_error("image", parsed["category"], parsed["status"], parsed["detail"])
        except httpx.TimeoutException:
            logger.error("Error generating image (text-only): timeout after 180s")
            return format_generation_error(
                "image", CATEGORY_TIMEOUT, 0,
                "The image service did not respond within 180 seconds.",
            )
        except httpx.HTTPError as e:
            logger.error("Error generating image (network, text-only): %s", e)
            return format_generation_error(
                "image", CATEGORY_BACKEND_UNAVAILABLE, 0,
                f"Could not reach the image service: {e}",
            )
        except Exception as e:
            logger.error("Error generating image (text-only): %s", e)
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
) -> str:
    """
    Generate or edit a video using AI based on a text prompt.
    Supports text-to-video, image-to-video, and video-to-video editing.

    Token costs per second (source of truth: pricing.py VIDEO_TOKEN_RATES):
    - runway-aleph: 17 tokens/sec (5-10s)
    - runway-4.5: 14 tokens/sec (5, 8, or 10s)
    - veo-3.1: 44 tokens/sec (8s only)
    - veo-3.1-flash: 17 tokens/sec (8s only)
    - veo-3.1-ultra: 65 tokens/sec (8s only)
    - kling-v3-omni-pro: 26 tokens/sec (3-15s)
    - kling-v3-omni-std: 19 tokens/sec (3-15s)

    Seedance 2.0 (resolution-based pricing, 4-15s, default 5s):
    - seedance-2.0: 480p=15, 720p=32, 1080p=72 tokens/sec
    - seedance-2.0-fast: 480p=12, 720p=26 tokens/sec (no 1080p -> downgraded to 720p)
    - Reference-video discount (reference_videos sent): seedance-2.0 480p=9/720p=20/1080p=43;
      seedance-2.0-fast 480p=7/720p=16.
    Seedance-only params: resolution, generate_audio, seed, media_url + end_frame
    (image mode), reference_images/reference_videos/reference_audios (reference mode).
    """
    logger.debug("Tool 'generate_video' called with prompt='%s', model='%s', duration=%d", prompt, model, duration)

    # Defense-in-depth: refuse disallowed content even if it slipped past
    # the chatbot-layer moderation (e.g., a stale pending action).
    if is_disallowed_content(prompt):
        logger.warning("generate_video refused: disallowed prompt content")
        return get_refusal_message(prompt)

    prompt = clean_prompt_from_model_mentions(prompt)
    is_seedance = model in SEEDANCE2_MODELS
    # Seedance accepts an empty/"auto" duration (normalized to 5); other models
    # expect an explicit integer.
    duration = normalize_seedance_duration(duration) if is_seedance else int(duration)

    allowed_models = [
        "runway", "runway-aleph", "runway-4.5", "veo-3.1", "veo-3.1-flash", "veo-3.1-ultra",
        "luma-labs", "seedance-pro", "kling-v1",
        "kling-v3-omni-pro", "kling-v3-omni-std",
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
        "ai_model": model,
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
            elif model in ["kling-v3-omni-std", "kling-v3-omni-pro"]:
                payload["media_url"] = video_file["url"]
                logger.debug("Using reference video URL (kling-v3): %s", video_file["url"])
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
        # 900s matches the chatbot-layer ceiling (it gives up at 15 min anyway);
        # async polling is the long-term solution but requires backend changes
        # outside this codebase. AsyncClient keeps the event loop free for
        # other conversations while the backend generates.
        timeout = httpx.Timeout(900.0, connect=15.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            logger.debug("Response status: %d", response.status_code)
            response.raise_for_status()
            result = response.json()

        logger.debug("Backend response: %s", result)

        video_url = result.get("video_url")
        if video_url:
            await session_manager.save_generated_file(conversation_uuid, video_url, "video")
            await session_manager.clear_reference_files(conversation_uuid)
            logger.debug("Reference files cleared after video generation")
            return f"Video generated successfully with {model}."
        else:
            return "Video generation initiated but URL not immediately available. Check status later."

    except httpx.HTTPStatusError as e:
        parsed = parse_backend_error(e.response.status_code, e.response.text)
        logger.error("Error generating video (HTTP %d): %s", parsed["status"], parsed["detail"])
        return format_generation_error("video", parsed["category"], parsed["status"], parsed["detail"])
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
