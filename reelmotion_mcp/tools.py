import os
import httpx
import json
import logging
import re
from typing import Optional

from request_context import get_api_token, get_conversation_uuid
from session_manager import get_session_manager
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def is_blob_url(url: str) -> bool:
    """Check if a URL is a browser-only blob: URL that cannot be fetched server-side."""
    return isinstance(url, str) and url.startswith("blob:")


def clean_prompt_from_model_mentions(prompt: str) -> str:
    """
    Remove model name mentions from the prompt before sending to the backend.

    Examples:
        "anima este video con sora 2" -> "anima este video"
        "genera una imagen con GPT" -> "genera una imagen"
        "create a video using runway aleph" -> "create a video"
    """
    if not prompt:
        return prompt

    patterns = [
        # English: "with sora 2", "using runway aleph", etc.
        r"\s+(?:with|using|via|through|by)\s+(?:sora[-\s]?2(?:\s+pro)?|runway(?:[-\s]?(?:aleph|4\.?5))?|veo[-\s]?3\.?1(?:[-\s]?(?:flash|ultra))?|nano[-\s]?banana|gpt|freepik|luma[-\s]?labs?|seedance[-\s]?pro|kling[-\s]?(?:v?3[-\s]?omni[-\s]?(?:pro|std)|v1))\s*$",
        # Spanish: "con sora 2", "usando runway aleph", etc.
        r"\s+(?:con|usando|mediante|por)\s+(?:sora[-\s]?2(?:\s+pro)?|runway(?:[-\s]?(?:aleph|4\.?5))?|veo[-\s]?3\.?1(?:[-\s]?(?:flash|ultra))?|nano[-\s]?banana|gpt|freepik|luma[-\s]?labs?|seedance[-\s]?pro|kling[-\s]?(?:v?3[-\s]?omni[-\s]?(?:pro|std)|v1))\s*$",
        # Bare model name at end without preposition
        r"\s+(?:sora[-\s]?2(?:\s+pro)?|runway[-\s]?(?:aleph|4\.?5)|veo[-\s]?3\.?1[-\s]?(?:flash|ultra)|kling[-\s]?v?3[-\s]?omni[-\s]?(?:pro|std))\s*$",
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
    """
    logger.debug("Tool 'generate_image' called with prompt='%s', model='%s'", prompt, model)

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
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload, headers=headers)
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
        except Exception as e:
            logger.error("Error generating image (with context images): %s", e)
            return f"Error generating image: {str(e)}"

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
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload, headers=headers)
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
        except Exception as e:
            logger.error("Error generating image (text-only): %s", e)
            return f"Error generating image: {str(e)}"


async def generate_video(
    prompt: str,
    model: str,
    duration: int,
    aspect_ratio: str = "16:9",
    reference_image: Optional[str] = None,
    reference_video: Optional[str] = None,
) -> str:
    """
    Generate or edit a video using AI based on a text prompt.
    Supports text-to-video, image-to-video, and video-to-video editing.

    Token costs per second:
    - runway-aleph: 17 tokens/sec (5-10s)
    - runway-4.5: 14 tokens/sec (5, 8, or 10s)
    - veo-3.1: 44 tokens/sec (8s only)
    - veo-3.1-flash: 17 tokens/sec (8s only)
    - veo-3.1-ultra: 65 tokens/sec (8s only)
    - sora-2: 11 tokens/sec (4, 8, or 12s only)
    - sora-2-pro: 33 tokens/sec (4, 8, or 12s only)
    - kling-v3-omni-pro: 26 tokens/sec (3-15s)
    - kling-v3-omni-std: 19 tokens/sec (3-15s)
    """
    logger.debug("Tool 'generate_video' called with prompt='%s', model='%s', duration=%d", prompt, model, duration)

    prompt = clean_prompt_from_model_mentions(prompt)
    duration = int(duration)

    allowed_models = [
        "runway", "runway-aleph", "runway-4.5", "veo-3.1", "veo-3.1-flash", "veo-3.1-ultra",
        "luma-labs", "seedance-pro", "kling-v1", "sora-2", "sora-2-pro",
        "kling-v3-omni-pro", "kling-v3-omni-std",
    ]

    if model not in allowed_models:
        return f"Error: Invalid model '{model}'. Allowed models are: {', '.join(allowed_models)}"

    duration_rules = {
        "sora-2": [4, 8, 12],
        "sora-2-pro": [4, 8, 12],
        "veo-3.1": [8],
        "veo-3.1-flash": [8],
        "veo-3.1-ultra": [8],
        "luma-labs": [5],
        "seedance-pro": [5],
        "runway": [5, 10],
        "runway-aleph": [5, 10],
        "runway-4.5": [5, 8, 10],
        "kling-v1": [5, 10],
        "kling-v3-omni-pro": list(range(3, 16)),
        "kling-v3-omni-std": list(range(3, 16)),
    }

    if model in duration_rules and duration not in duration_rules[model]:
        return (
            f"Error: Duration {duration}s is not valid for model '{model}'. "
            f"Allowed durations: {duration_rules[model]} seconds."
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

    if context_files:
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
        # httpx.Client with a large timeout; async polling is the long-term solution
        # but requires backend changes outside this codebase.
        with httpx.Client(timeout=12000.0) as client:
            response = client.post(url, json=payload, headers=headers)
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
        error_detail = f"HTTP Error {e.response.status_code}: {e.response.text}"
        logger.error("Error generating video (HTTP): %s", error_detail)
        return f"Error generating video: {error_detail}"
    except httpx.TimeoutException:
        logger.error("Error generating video: Request timeout")
        return "Error generating video: Request timeout after 1800s"
    except Exception as e:
        import traceback
        logger.error("Error generating video: %s\n%s", e, traceback.format_exc())
        return f"Error generating video: {type(e).__name__}: {str(e)}"


async def generate_speech(
    text: str,
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    model_id: str = "eleven_multilingual_v2",
) -> str:
    """
    Generate speech/audio from text using the ElevenLabs API.

    COST: 1-500 characters = 1 token, 500-999 characters = 8 tokens,
          1000+ characters = 13 tokens per 1000 chars.
    """
    import base64
    import uuid as uuid_lib

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
            tokens_cost = 5

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

    except Exception as e:
        logger.error("Error generating speech: %s", e)
        return f"Error generating speech: {str(e)}"


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

OUTPUT FORMAT when enough detail is gathered:
✨ Refined Prompt:
"[Full refined prompt, rich in detail and optimized for the target model]"

Then ask: "Would you like to adjust anything, or are you ready to generate?"
"""


async def craft_prompt(
    idea: str,
    media_type: str = "image",
    user_answers: str = "",
) -> str:
    """
    Refine and improve a raw idea into a production-ready prompt for AI image or video generation.

    This tool asks targeted questions when details are missing and suggests concrete options
    for the user to choose from. It NEVER invents details — it only guides and refines.

    Args:
        idea: The user's raw description or idea for the image/video.
        media_type: "image" or "video". Defaults to "image".
        user_answers: Optional answers the user has already provided to follow-up questions.
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

    try:
        response = await model.generate_content_async(user_message)
        return response.text
    except Exception as e:
        logger.error("Error in craft_prompt: %s", e)
        return f"Error crafting prompt: {str(e)}"
