from typing import Any, Optional
import os
import re
import unicodedata
import logging
from dotenv import load_dotenv
from pydantic import BaseModel

import httpx
from fastmcp import FastMCP
from fastmcp.prompts.prompt import Message
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse

from chatbot import get_chatbot
from prompts import REELMOTION_SYSTEM_PROMPT
from tools import generate_image as generate_image_impl, generate_video as generate_video_impl, generate_speech as generate_speech_impl, craft_prompt as craft_prompt_impl
from request_context import (
    set_api_token,
    set_conversation_uuid,
    set_token_balance,
    set_chat_id,
    clear_insufficient_block,
    get_insufficient_block,
)
from session_manager import get_session_manager
from logging_config import setup_logging
from pricing import min_video_cost

# Load environment variables
load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)

# Below this balance the user cannot generate ANY video — suggest buying tokens.
LOW_BALANCE_THRESHOLD = min_video_cost()
if LOW_BALANCE_THRESHOLD <= 0:
    # Pricing tables empty/misconfigured — the mechanical low-balance
    # tokens_sale hint would silently never fire. Surface it loudly.
    logger.warning(
        "LOW_BALANCE_THRESHOLD is %s (min_video_cost returned no priced models); "
        "mechanical low-balance tokens_sale hints are disabled.",
        LOW_BALANCE_THRESHOLD,
    )

# Frontend action hints returned in the /api/chat response ("actions" key).
ACTION_EDITOR = "editor"          # show a "go to editor" button
ACTION_TOKENS_SALE = "tokens_sale"  # show a "buy tokens" button

# The bot can surface an action conversationally by emitting a hidden marker
# like <<ACTION:editor>> or <<ACTION:tokens_sale>> anywhere in its reply
# (instructed in the system prompt). We strip these out before the text is
# shown to the user and fold them into the "actions" array.
# The optional leading horizontal space is absorbed so removing an in-sentence
# marker ("see <<ACTION:editor>> here") doesn't leave a double space behind.
ACTION_MARKER_PATTERN = re.compile(
    r"[ \t]?<<\s*ACTION\s*:\s*(editor|tokens_sale)\s*>>", re.IGNORECASE
)


def extract_action_markers(text: str):
    """
    Pull <<ACTION:...>> markers the bot emitted out of its reply.
    Returns (clean_text, marker_actions) where clean_text has the markers
    (and the whitespace they leave behind) removed, and marker_actions is the
    ordered, de-duplicated list of action names the bot requested.
    """
    if not text:
        return text, []

    marker_actions = []
    for match in ACTION_MARKER_PATTERN.finditer(text):
        action = match.group(1).lower()
        if action not in marker_actions:
            marker_actions.append(action)

    clean_text = ACTION_MARKER_PATTERN.sub("", text)
    # Only collapse blank lines left where an own-line marker used to be.
    # (Deliberately NOT touching horizontal whitespace runs — those may be
    # meaningful in code snippets or formatted text in the reply.)
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()
    return clean_text, marker_actions


# Deterministic intent detection on the USER's message, so the action fires
# even if Gemini forgets to emit the marker. Phrases are matched on an
# accent-folded, lower-cased copy of the message (so "dónde" == "donde").
EDITOR_REQUEST_PHRASES = (
    # Spanish — asking where/how to edit or to go to the editor
    "donde edito", "donde puedo editar", "donde se edita", "donde lo edito",
    "como edito", "como puedo editar", "como lo edito", "quiero editar",
    "ir al editor", "al editor", "abrir editor", "abrir el editor",
    "abre el editor", "llevame al editor", "ir a editar", "editar el video",
    "editar mi video", "editar este video", "editar la imagen", "editar esto",
    # English
    "where do i edit", "where can i edit", "how do i edit", "how can i edit",
    "go to the editor", "go to editor", "open the editor", "open editor",
    "take me to the editor", "i want to edit", "edit the video", "edit my video",
    "edit this video", "edit the image", "edit this",
)
# tokens_sale fires when the user clearly wants to ACQUIRE tokens. Detected as
# either a standalone recharge/top-up phrase, OR a "token/credit" word together
# with a purchase verb anywhere in the message (so "quiero comprar más ...
# tokens" matches even when the words aren't adjacent). Purchase verbs are
# deliberately acquisition-specific so "quiero generar con mis tokens" (use,
# not buy) does NOT trigger it.
TOKENS_STANDALONE_PHRASES = (
    "top up", "top-up", "topup", "recargar", "recarga", "recharge",
)
TOKEN_WORDS = ("token", "credito", "credit")
TOKEN_BUY_VERBS = (
    "comprar", "compro", "compra", "buy", "purchase", "get more",
    "conseguir", "obtener", "necesito", "dame", "give me", "quiero mas",
)


def _fold(text: str) -> str:
    """Lower-case and strip accents for tolerant phrase matching."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def _wants_to_buy_tokens(folded: str) -> bool:
    if any(phrase in folded for phrase in TOKENS_STANDALONE_PHRASES):
        return True
    has_token_word = any(word in folded for word in TOKEN_WORDS)
    has_buy_verb = any(verb in folded for verb in TOKEN_BUY_VERBS)
    return has_token_word and has_buy_verb


def detect_user_requested_actions(message: str) -> list:
    """
    Action hints derived directly from what the USER asked for, independent of
    the bot's reply or balance — e.g. "¿dónde edito el video?" → editor,
    "quiero comprar más tokens" → tokens_sale. Returns an ordered, de-duplicated
    list (editor before tokens_sale).
    """
    folded = _fold(message)
    actions = []
    if any(phrase in folded for phrase in EDITOR_REQUEST_PHRASES):
        actions.append(ACTION_EDITOR)
    if _wants_to_buy_tokens(folded):
        actions.append(ACTION_TOKENS_SALE)
    return actions


def compute_response_actions(
    files: list, insufficient_block, token_balance, marker_actions=None
) -> list:
    """
    Action hints for the frontend, derived from the request outcome AND any
    actions the bot requested conversationally via <<ACTION:...>> markers:
    - "editor" when a generation succeeded (fresh files) OR the bot suggested
      opening the editor (e.g. the user asked where to edit an existing asset).
    - "tokens_sale" when a generation was blocked for insufficient balance,
      the balance is too low to generate any video, OR the bot proactively
      suggested topping up because the user is running low.
    """
    marker_actions = marker_actions or []
    actions = []
    if files or ACTION_EDITOR in marker_actions:
        actions.append(ACTION_EDITOR)
    if (
        insufficient_block
        or (token_balance is not None and token_balance < LOW_BALANCE_THRESHOLD)
        or ACTION_TOKENS_SALE in marker_actions
    ):
        actions.append(ACTION_TOKENS_SALE)
    return actions

# Initialize FastMCP server with CORS middleware
mcp = FastMCP(
    "reelmotion",
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]
)

# Request model for the chat endpoint
class ChatRequest(BaseModel):
    message: str
    context: str = ""
    conversation_uuid: str  # UUID de la conversación (obligatorio)

async def health_endpoint(request: Request):
    """
    Health check endpoint for Docker and monitoring.
    """
    return JSONResponse({"status": "healthy", "service": "reelmotion-mcp"})

async def chat_endpoint(request: Request):
    """
    HTTP endpoint for React frontend to chat with the bot.
    """
    # CORS headers are handled by Nginx reverse proxy
    # No need to add them here to avoid duplicates
    
    # Handle preflight requests explicitly (nginx handles this too, but just in case)
    if request.method == "OPTIONS":
        return JSONResponse({}, status_code=204)

    try:
        # Check Content-Type to handle both JSON and Form Data
        content_type = request.headers.get("content-type", "")
        file_urls = []
        file_types = []
        
        if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
            data = await request.form()
            message = data.get("message")
            context = data.get("context", "")
            token = data.get("token")
            conversation_uuid = data.get("conversation_uuid")
            tokens_raw = data.get("tokens")
            chat_id = data.get("chat_id")

            # Handle file URLs from form data (no more binary uploads)
            # Expecting: files[0]=URL, files[1]=URL, etc.
            # and file_types[0]=image, file_types[1]=video, etc.
            file_index = 0
            while True:
                file_url = data.get(f"files[{file_index}]")
                if not file_url:
                    break
                file_type = data.get(f"file_types[{file_index}]", "image")
                file_urls.append(file_url)
                file_types.append(file_type)
                file_index += 1
        else:
            # Default to JSON
            data = await request.json()
            message = data.get("message")
            context = data.get("context", "")
            token = data.get("token")
            conversation_uuid = data.get("conversation_uuid")
            tokens_raw = data.get("tokens")
            chat_id = data.get("chat_id")

            # Extract file URLs from JSON
            # Expecting: {"files": ["url1", "url2"], "file_types": ["image", "video"]}
            file_urls = data.get("files", [])
            file_types = data.get("file_types", [])

        # Validar UUID
        if not conversation_uuid:
            return JSONResponse(
                {"error": "conversation_uuid is required"}, 
                status_code=400
            )

        # Set conversation UUID in context for tools to access
        set_conversation_uuid(conversation_uuid)

        if token:
            set_api_token(str(token))

        # User token balance (sent by Laravel). None = unknown -> never block.
        token_balance = None
        if tokens_raw is not None and str(tokens_raw).strip() != "":
            try:
                token_balance = max(0, int(float(str(tokens_raw))))
            except (TypeError, ValueError):
                logger.warning("Non-numeric 'tokens' value received: %r", tokens_raw)
        set_token_balance(token_balance)

        if chat_id:
            set_chat_id(str(chat_id))

        # Reset per-request insufficient-balance flag
        clear_insufficient_block()

        if not message:
            # If files are attached but no message, use a default analysis prompt
            if file_urls:
                message = "Analyze this file"
            else:
                return JSONResponse({"error": "Message is required"}, status_code=400)

        # Crear chatbot con UUID de conversación
        logger.debug(f"Creating/getting chatbot for UUID='{conversation_uuid}'")
        chatbot = get_chatbot(conversation_uuid)
        logger.debug(f"Chatbot instance uuid='{chatbot.conversation_uuid}'")
        
        # Store reference file URLs in chatbot session (persists across messages)
        if file_urls:
            await chatbot.set_reference_files(file_urls, file_types)
        
        response = await chatbot.send_message(message, context, file_urls=file_urls, file_types=file_types)

        # Pull any <<ACTION:...>> markers the bot emitted and strip them from
        # the user-facing text, then add anything the USER explicitly asked for
        # (e.g. "where do I edit?" / "I want to buy tokens") so the action fires
        # even when Gemini forgets the marker.
        response, marker_actions = extract_action_markers(response)
        for action in detect_user_requested_actions(message):
            if action not in marker_actions:
                marker_actions.append(action)

        # Get generated files and include in response
        files = await chatbot.get_generated_files()
        logger.debug(f"Retrieved {len(files)} files from chatbot")
        logger.debug(f"Files: {files}")
        
        # Limpiar archivos después de enviarlos
        session_manager = get_session_manager()
        await session_manager.clear_sent_files(conversation_uuid)
        
        final_response = {
            "response": response,
            "files": files
        }

        # Surface insufficient-balance blocks to the caller (extra keys are
        # ignored by Laravel today, so this is backward compatible).
        block = get_insufficient_block()
        if block:
            final_response["insufficient_tokens"] = True
            final_response["tokens_required"] = block["required"]
            final_response["tokens_available"] = block["available"]

        # Frontend action hints: "editor" (generation succeeded or the bot
        # suggested editing → open editor button) and/or "tokens_sale" (no/low
        # tokens, a block, or the bot suggested topping up → buy tokens button).
        # Omitted entirely when there is nothing to suggest.
        actions = compute_response_actions(files, block, token_balance, marker_actions)
        if actions:
            final_response["actions"] = actions

        logger.debug(f"Final response being sent: {final_response}")
        
        return JSONResponse(final_response)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Add the custom route to FastMCP
# We allow OPTIONS here to prevent 405 from the router before middleware/endpoint can handle it
mcp._additional_http_routes.append(
    Route("/api/chat", chat_endpoint, methods=["POST", "OPTIONS"])
)

# Constants
# Replace these with actual API endpoints/keys when available
IMAGE_GEN_API_URL = "https://api.example.com/image"
VIDEO_GEN_API_URL = "https://api.example.com/video"


@mcp.resource(
    uri="resource://reelbot/system-prompt",
    name="ReelbotSystemPrompt",
    description="Prompt base del sistema (contexto) para Reelbot.",
    mime_type="text/plain",
    tags={"reelmotion", "reelbot", "prompt"},
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def reelbot_system_prompt() -> str:
    return REELMOTION_SYSTEM_PROMPT


@mcp.prompt(
    name="reelbot_chat_context",
    description="Entrega el contexto base de Reelbot (mensaje system) para iniciar un chat.",
    tags={"reelmotion", "reelbot"},
)
def reelbot_chat_context() -> list[Message]:
    return [Message(REELMOTION_SYSTEM_PROMPT, role="system")]

@mcp.tool
def generate_image(
    prompt: str,
    model: str = "Seedream",
    image_type: int = 1,
    quantity: int = 1,
    reference_image: Optional[str] = None,
    reference_images: Optional[list[str]] = None,
    aspect_ratio: str = "16:9",
    quality: str = "2K",
) -> str:
    """
    Generate or edit an image using the reelmotion backend.
    This tool supports both text-to-image generation AND image-to-image editing/transformation.
    COST: Seedream = 4, GPT = 6, Nano Banana 2 = 7, Midjourney = 9 tokens per image.
    There is NO 'Freepik' model.

    Model selection (pick by intent):
    - Seedream: realism, photographic fidelity, cinematic scenes, reference images (recommended default).
    - Midjourney: artistic style, illustration, creative concepts.
    - Nano Banana 2: quick edits of an existing image, multi-reference composition.
    - GPT: readable text inside the image, strict instruction following.

    Use cases:
    - Text-to-image: Generate a new image from a text description (type 1).
    - Image-to-image (editing): Transform or edit an existing image using a text prompt + reference image (type 2).
      Examples: change style, add elements, modify colors, remove objects, apply effects.
    - Multi-image reference: Generate using multiple reference images (type 3).

    Delivery: Seedream and Midjourney are asynchronous (hybrid). The call returns the
    finished image (200) or, for slow jobs, a "still processing" marker (202) — the user
    is notified when it's ready; never retry a processing job. They always produce ONE
    image per call ('type'/'quantity' are ignored). A failed job (422) is auto-refunded.

    Args:
        prompt: The description of the image to generate (rich, visual, in English), or editing instructions.
        model: One of: 'Seedream', 'GPT', 'Nano Banana 2', 'Midjourney'. Defaults to 'Seedream'.
        image_type: 1 (text only), 2 (text + reference image), 3 (text + multiple references). Only GPT and Nano Banana 2 honor it.
        quantity: Number of images (GPT / Nano Banana 2 only; Seedream/Midjourney always 1). Defaults to 1.
        reference_image: URL of a reference image (image editing/transformation).
        reference_images: List of reference image URLs (multi-image reference).
        aspect_ratio: '16:9' (default), '9:16', '1:1', etc. Choose to match the destination.
        quality: '2K' or '3K' — Seedream only; ignored by other models and does not change the cost.
    """
    return generate_image_impl(
        prompt, model, image_type, quantity, reference_image, reference_images,
        aspect_ratio=aspect_ratio, quality=quality,
    )

@mcp.tool
def generate_video(
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
    This tool supports text-to-video, image-to-video, AND video-to-video editing.
    IMPORTANT: User must confirm model, duration, and token cost before calling this tool.

    Use cases:
    - Text-to-video: Generate a new video from a text description.
    - Image-to-video: Animate a reference image into a video.
    - Video-to-video (editing): Transform or edit an existing video using a text prompt + reference video.
      Examples: change style, add effects, modify movement, re-edit scenes.
      Supported models for video-to-video: runway-aleph, kling-o3 (video-edit).

    Token costs per second and valid durations:
    - runway-aleph: 17 tokens/sec (5-10s) - video-to-video editing
    - runway-4.5: 13 tokens/sec (5, 8, or 10s) - high quality (hybrid: 200 sync or 202 processing)
    - veo-3.1: 44 tokens/sec (8s only)
    - veo-3.1-flash: 17 tokens/sec (8s only)
    - veo-3.1-ultra: 65 tokens/sec (8s only) - maximum quality
    - sora-2 / sora-2-pro: cost set by the backend

    Kling v3 / o3 (RESOLUTION + route + audio based, 3-15s; reference 3-10s):
    - kling-v3: max quality, 4K, native audio, motion-control. text/image: 720p=9, 1080p=12,
      4k=42 (+audio 720p=12, 1080p=14); motion-control: 720p=13, 1080p=17.
    - kling-v3-turbo: fast/cheap drafts (text/image only, max 1080p, no audio): 720p=12, 1080p=14.
    - kling-o3: character/style consistency (reference) or edit an existing video: 720p=13, 1080p=17;
      plain text/image: 720p=9, 1080p=12, 4k=42.
      Heuristic: edit a video -> kling-o3 + edit_video; keep a character/style from images ->
      kling-o3 + reference_images; animate with a guide video -> kling-v3 + motion_video;
      fast/cheap -> kling-v3-turbo; max quality/4K/audio -> kling-v3.

    Seedance 2.0 (RESOLUTION-based pricing, 4-15s duration, default 5s):
    - seedance-2.0: 480p=15, 720p=32, 1080p=72 tokens/sec (supports 1080p)
    - seedance-2.0-fast: 480p=12, 720p=26 tokens/sec (max 720p; 1080p auto-downgraded to 720p)
    - Reference-video discount (reference_videos sent): seedance-2.0 480p=9/720p=20/1080p=43;
      seedance-2.0-fast 480p=7/720p=16.
    - Mode is auto-detected: reference_images/reference_videos/reference_audios -> reference mode;
      media_url -> image mode; prompt only -> text mode.

    Args:
        prompt: Description of the video to generate or editing instructions (exact user text, NO modifications)
        model: Provider to use (see catalog above). Sent to the backend as `provider`.
        duration: Video duration in seconds. Valid durations depend on model (see above)
        aspect_ratio: '16:9', '9:16', '1:1', etc. Seedance also accepts auto/21:9/4:3/3:4. Defaults to '16:9'
        reference_image: URL of reference image (for image-to-video generation)
        reference_video: URL of reference video (for video-to-video editing with runway-aleph / kling-o3)
        resolution: '480p'/'720p'/'1080p' (Seedance) or '720p'/'1080p'/'4k' (Kling). Defaults to '720p'
        generate_audio: Whether to generate audio (Seedance only). Defaults to True. Does not affect price.
        seed: Optional random seed for reproducibility (Seedance only)
        media_url: Reference image (or video) URL — image mode (Seedance/Kling) or video-edit (Kling)
        end_frame: Optional last-frame image URL for Seedance image mode
        reference_images: Reference image URLs — Seedance reference mode (max 9) or Kling reference/consistency
        reference_videos: Reference video URLs — Seedance reference mode (max 3, discount)
        reference_audios: List of reference audio URLs for Seedance reference mode (max 3)
        quality: Kling resolution '720p'/'1080p'/'4k' (default 720p; 4K only on kling-v3/o3 text/image)
        sound: Kling audio 'on'/'off' (default off; only effective on the text/image route of kling-v3/o3)
        mode: Force a Kling route: 'motion' | 'reference' | 'edit'
        keep_sound: Keep the original audio in Kling motion/reference/edit routes
        motion_video: Guide video URL for kling-v3 motion-control
        edit_video: Source video URL for kling-o3 video-edit
    """
    return generate_video_impl(
        prompt, model, duration, aspect_ratio, reference_image, reference_video,
        resolution=resolution, generate_audio=generate_audio, seed=seed,
        media_url=media_url, end_frame=end_frame, reference_images=reference_images,
        reference_videos=reference_videos, reference_audios=reference_audios,
        quality=quality, sound=sound, mode=mode, keep_sound=keep_sound,
        motion_video=motion_video, edit_video=edit_video,
    )

@mcp.tool
async def generate_speech(
    text: str,
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    model_id: str = "eleven_multilingual_v2"
) -> str:
    """
    Generate speech/audio from text using the ElevenLabs API.
    
    COST: 1-500 characters = 1 token, 500-999 characters = 8 tokens, 1000+ characters = 13 tokens per 1000 chars.
    
    Args:
        text: The text content to convert to speech.
        voice_id: The ID of the voice to use. Defaults to "Rachel" (21m00Tcm4TlvDq8ikWAM).
        model_id: The model ID to use. Defaults to "eleven_multilingual_v2".
    """
    return await generate_speech_impl(text, voice_id, model_id)

@mcp.tool
async def craft_prompt(
    idea: str,
    media_type: str = "image",
    user_answers: str = "",
) -> str:
    """
    Refine and improve a raw idea into a production-ready prompt for AI image or video generation.

    This tool asks targeted questions when details are missing and suggests concrete options
    for the user to choose from. It NEVER invents details — it only guides and refines.

    COST: Free (no tokens charged for prompt crafting).

    Args:
        idea: The user's raw description or idea (e.g., "a cat in space").
        media_type: Target media type — "image" or "video". Defaults to "image".
        user_answers: Optional answers to previous follow-up questions, to continue refining.
    """
    return await craft_prompt_impl(idea, media_type, user_answers)


@mcp.tool
async def chat(message: str, context: str = "") -> str:
    """
    Process a chat message and return a response using Gemini AI.
    This tool should ONLY be used for general conversation, questions, or clarification.
    
    CRITICAL: 
    - If the user asks to generate an IMAGE, use the 'generate_image' tool.
    - If the user asks to generate a VIDEO, use the 'generate_video' tool.
    DO NOT attempt to generate media or ASCII art within this chat response.
    
    Args:
        message: The user's message.
        context: Optional context or history.
    """
    chatbot = get_chatbot()
    response = await chatbot.send_message(message, context)
    return response

if __name__ == "__main__":
    import sys

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))

    # HTTP mode (SSE transport) used by our Docker deployment
    if len(sys.argv) > 1 and sys.argv[1] == "http":
        mcp.run(transport="sse", host=host, port=port)
    else:
        mcp.run()
