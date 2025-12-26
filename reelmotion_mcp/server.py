from typing import Any, Optional
import os
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
from tools import generate_image as generate_image_impl, generate_video as generate_video_impl
from request_context import set_api_token, set_conversation_uuid
from session_manager import get_session_manager

#COMANDS TO RUN THIS PROYECT
#.\venv\Scripts\activate
#.\start_http.bat

# Load environment variables
load_dotenv()

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
    # Manually handle CORS headers to ensure they are present
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "*",
    }

    # Handle preflight requests explicitly
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=headers)

    try:
        # Check Content-Type to handle both JSON and Form Data
        content_type = request.headers.get("content-type", "")
        images = []
        
        if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
            data = await request.form()
            message = data.get("message")
            context = data.get("context", "")
            token = data.get("token")
            conversation_uuid = data.get("conversation_uuid")
            
            # Handle files from form data
            if "files[]" in data:
                uploaded_files = data.getlist("files[]")
                for file in uploaded_files:
                    # Ensure it's an UploadFile
                    if hasattr(file, "read"):
                        content = await file.read()
                        mime_type = file.content_type or "image/jpeg"
                        images.append({"mime_type": mime_type, "data": content})
        else:
            # Default to JSON
            data = await request.json()
            message = data.get("message")
            context = data.get("context", "")
            token = data.get("token")
            conversation_uuid = data.get("conversation_uuid")

        # Validar UUID
        if not conversation_uuid:
            return JSONResponse(
                {"error": "conversation_uuid is required"}, 
                status_code=400, 
                headers=headers
            )

        # Set conversation UUID in context for tools to access
        set_conversation_uuid(conversation_uuid)

        if token:
            set_api_token(str(token))

        if not message:
             return JSONResponse({"error": "Message is required"}, status_code=400, headers=headers)

        # Crear chatbot con UUID de conversación
        chatbot = get_chatbot(conversation_uuid)
        
        # Store reference images in chatbot session (persists across messages)
        if images:
            await chatbot.set_reference_images(images)
        
        response = await chatbot.send_message(message, context, images=images)
        
        # Get generated files and include in response
        files = await chatbot.get_generated_files()
        print(f"DEBUG [server.py]: Retrieved {len(files)} files from chatbot")
        print(f"DEBUG [server.py]: Files: {files}")
        
        # Limpiar archivos después de enviarlos
        session_manager = get_session_manager()
        await session_manager.clear_sent_files(conversation_uuid)
        
        return JSONResponse({
            "response": response,
            "files": files
        }, headers=headers)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500, headers=headers)

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
    Generate an image based on a text prompt using the ReelMotion backend.
    COST: 10 tokens per image generated.
    
    Args:
        prompt: The description of the image to generate.
        model: The model to use. MUST be one of: 'Nano Banana', 'GPT'. Defaults to 'GPT'.
        image_type: 1 (text only), 2 (text + image), 3 (text + multiple images). Defaults to 1.
        quantity: Number of images to generate. Defaults to 1.
        reference_image: URL or base64 of reference image (for type 2).
        reference_images: List of URLs or base64 of reference images (for type 3).
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
    Generate a video using AI based on a text prompt.
    IMPORTANT: User must confirm model, duration, and token cost before calling this tool.
    
    Token costs per second and valid durations:
    - runway-aleph: 19 tokens/sec (5-10s) - video-to-video editing
    - veo-3.1: 48 tokens/sec (8s only)
    - veo-3.1-flash: 21 tokens/sec (8s only)
    - veo-3.1-ultra: 60 tokens/sec (8s only) - maximum quality
    - sora-2: 15 tokens/sec (4, 8, or 12s only)
    - sora-2-pro: 30 tokens/sec (4, 8, or 12s only)
    
    Args:
        prompt: Description of the video to generate (exact user text, NO modifications)
        model: AI model to use. See token costs above.
        duration: Video duration in seconds. Valid durations depend on model (see above)
        aspect_ratio: '16:9', '9:16', or '1:1'. Defaults to '16:9'
        reference_image: URL of reference image (for image-to-video models)
        reference_video: URL of reference video (only for runway-aleph)
    """
    return generate_video_impl(prompt, model, duration, aspect_ratio, reference_image, reference_video)

@mcp.tool
async def generate_video(prompt: str, duration: int = 5, fps: int = 24) -> str:
    """
    Generate a video based on a text prompt.
    
    Args:
        prompt: The description of the video to generate.
        duration: Duration of the video in seconds.
        fps: Frames per second.
    """
    # Logic to call external video generation API would go here
    return f"Video generated for prompt: '{prompt}' ({duration}s @ {fps}fps). URL: https://placeholder.com/video.mp4"

@mcp.tool
async def chat(message: str, context: str = "") -> str:
    """
    Process a chat message and return a response using Gemini AI.
    This tool can be used to handle general conversation or query processing 
    before deciding to generate media.
    
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
