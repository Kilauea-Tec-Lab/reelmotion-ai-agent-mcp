from typing import Any, Optional
import os
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
from request_context import set_api_token, set_conversation_uuid
from session_manager import get_session_manager
from logging_config import setup_logging

# Load environment variables
load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)

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
    model: str = "GPT", 
    image_type: int = 1, 
    quantity: int = 1, 
    reference_image: Optional[str] = None, 
    reference_images: Optional[list[str]] = None
) -> str:
    """
    Generate or edit an image using the ReelMotion backend.
    This tool supports both text-to-image generation AND image-to-image editing/transformation.
    COST: Nano Banana 2 = 7 tokens, GPT = 6 tokens, Freepik = 1 token per image.
    
    Use cases:
    - Text-to-image: Generate a new image from a text description (type 1).
    - Image-to-image (editing): Transform or edit an existing image using a text prompt + reference image (type 2).
      Examples: change style, add elements, modify colors, remove objects, apply effects.
    - Multi-image reference: Generate using multiple reference images (type 3).
    
    Args:
        prompt: The description of the image to generate, or editing instructions for image-to-image.
        model: The model to use. MUST be one of: 'Nano Banana 2', 'GPT', 'Freepik'. Defaults to 'GPT'.
        image_type: 1 (text only), 2 (text + reference image for editing), 3 (text + multiple reference images). Defaults to 1.
        quantity: Number of images to generate. Defaults to 1.
        reference_image: URL of reference image (for type 2 - image editing/transformation).
        reference_images: List of URLs of reference images (for type 3 - multi-image reference).
    """
    return generate_image_impl(prompt, model, image_type, quantity, reference_image, reference_images)

@mcp.tool
def generate_video(
    prompt: str,
    model: str,
    duration: int,
    aspect_ratio: str = "16:9",
    reference_image: Optional[str] = None,
    reference_video: Optional[str] = None
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
      Supported models for video-to-video: runway-aleph, kling-v3-omni-std, kling-v3-omni-pro.
    
    Token costs per second and valid durations:
    - runway-aleph: 17 tokens/sec (5-10s) - video-to-video editing
    - runway-4.5: 14 tokens/sec (5, 8, or 10s) - high quality
    - veo-3.1: 44 tokens/sec (8s only)
    - veo-3.1-flash: 17 tokens/sec (8s only)
    - veo-3.1-ultra: 65 tokens/sec (8s only) - maximum quality
    - sora-2: 11 tokens/sec (4, 8, or 12s only)
    - sora-2-pro: 33 tokens/sec (4, 8, or 12s only)
    - kling-v3-omni-pro: 26 tokens/sec (3-15s) - text/image-to-video
    - kling-v3-omni-std: 19 tokens/sec (3-15s) - video-to-video editing
    
    Args:
        prompt: Description of the video to generate or editing instructions (exact user text, NO modifications)
        model: AI model to use. See token costs above.
        duration: Video duration in seconds. Valid durations depend on model (see above)
        aspect_ratio: '16:9', '9:16', or '1:1'. Defaults to '16:9'
        reference_image: URL of reference image (for image-to-video generation)
        reference_video: URL of reference video (for video-to-video editing with runway-aleph, kling-v3)
    """
    return generate_video_impl(prompt, model, duration, aspect_ratio, reference_image, reference_video)

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
