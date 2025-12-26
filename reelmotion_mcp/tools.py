import os
import httpx
import json
import asyncio
from typing import Optional
from request_context import get_api_token, get_conversation_uuid

def _download_image(url: str) -> Optional[bytes]:
    try:
        print(f"DEBUG: Downloading image from {url}")
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, follow_redirects=True)
            response.raise_for_status()
            return response.content
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

async def generate_image(
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
    print(f"DEBUG: MCP Tool 'generate_image' called with prompt='{prompt}', model='{model}'")
    print(f"DEBUG: Gemini passed reference_image='{reference_image}', reference_images='{reference_images}' (IGNORING THESE)")
    
    # Validate and normalize model
    allowed_models = ["Nano Banana", "GPT"]
    if model not in allowed_models:
        # Attempt normalization for common variations (e.g. "Nano Banana 2" -> "Nano Banana")
        if "nano" in model.lower():
            model = "Nano Banana"
        elif "gpt" in model.lower():
            model = "GPT"
        else:
            return f"Error: Invalid model '{model}'. Allowed models are: {', '.join(allowed_models)}"
            
    print(f"DEBUG: Normalized model to '{model}'")
    
    backend_url = os.getenv("BACKEND_URL")
    endpoint = os.getenv("IMAGE_CREATION_ENDPOINT")
    # Try to get token from request context first, then env
    api_token = get_api_token() or os.getenv("API_TOKEN")
    
    # Get REAL reference images from chatbot session (ignoring Gemini's invalid URLs)
    from chatbot import get_chatbot
    conversation_uuid = get_conversation_uuid() or "default"
    chatbot = get_chatbot(conversation_uuid)
    
    # Get reference images asynchronously
    context_images = await chatbot.get_reference_images()
    print(f"DEBUG: Retrieved {len(context_images) if context_images else 0} images from chatbot session")

    if not backend_url or not endpoint:
        return "Error: Backend URL or Image Creation Endpoint not configured."
    
    url = f"{backend_url}{endpoint}"
    print(f"DEBUG: Calling URL: {url}")
    
    # Prepare headers
    headers = {
        "Accept": "application/json",
    }
    
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    
    # If we have images from context, use JSON with base64
    if context_images:
        print(f"DEBUG: Sending JSON request with {len(context_images)} base64 images")
        headers["Content-Type"] = "application/json"
        
        payload = {
            "prompt": prompt,
            "model": model,
            "type": image_type,
            "quantity": quantity,
        }
        
        # Send all context images as reference_images array
        if len(context_images) == 1:
            # Single image: use reference_image field
            payload["reference_image"] = context_images[0]
        else:
            # Multiple images: use reference_images array
            payload["reference_images"] = context_images

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                
                print(f"DEBUG: Backend response: {result}")
                
                # Extract image URLs from response and store in chatbot
                # Try both 'images' and 'data' keys for compatibility
                images_data = result.get("images") or result.get("data")
                if images_data:
                    if isinstance(images_data, list):
                        for img in images_data:
                            if isinstance(img, dict) and "url" in img:
                                print(f"DEBUG: Adding file to chatbot: {img['url']}")
                                await chatbot.add_generated_file(img["url"], "image")
                    elif isinstance(images_data, dict) and "url" in images_data:
                        print(f"DEBUG: Adding file to chatbot: {images_data['url']}")
                        await chatbot.add_generated_file(images_data["url"], "image")
                
                # Clear reference images after successful use
                await chatbot.clear_reference_images()
                print("DEBUG: Reference images cleared after use")
                
                # Return simple success message instead of full JSON
                return f"Imágenes generadas exitosamente con {model}."
        except Exception as e:
            print(f"Error generating image (with context images): {e}")
            return f"Error generating image: {str(e)}"
            
    else:
        # No images in context - text-only generation
        print(f"DEBUG: Sending JSON request (text-only, no images)")
        headers["Content-Type"] = "application/json"
        
        payload = {
            "prompt": prompt,
            "model": model,
            "type": image_type,
            "quantity": quantity,
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                
                print(f"DEBUG: Backend response: {result}")
                
                # Extract image URLs from response and store in chatbot
                # Try both 'images' and 'data' keys for compatibility
                images_data = result.get("images") or result.get("data")
                if images_data:
                    if isinstance(images_data, list):
                        for img in images_data:
                            if isinstance(img, dict) and "url" in img:
                                print(f"DEBUG: Adding file to chatbot: {img['url']}")
                                await chatbot.add_generated_file(img["url"], "image")
                    elif isinstance(images_data, dict) and "url" in images_data:
                        print(f"DEBUG: Adding file to chatbot: {images_data['url']}")
                        await chatbot.add_generated_file(images_data["url"], "image")
                
                # Return simple success message instead of full JSON
                return f"Imágenes generadas exitosamente con {model}."
        except Exception as e:
            print(f"Error generating image (text-only): {e}")
            return f"Error generating image: {str(e)}"

async def generate_video(
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
    
    Token costs per second:
    - Runway: 8 tokens/sec (5-10s) - image-to-video
    - Runway Aleph: 19 tokens/sec (5-10s) - video-to-video editing
    - Veo 3.1: 48 tokens/sec (8s only)
    - Veo 3.1 Flash: 21 tokens/sec (8s only)
    - Veo 3.1 Ultra: 60 tokens/sec (8s only) - maximum quality
    - Luma Labs: 13 tokens/sec (5s only)
    - Seedance Pro: 15 tokens/sec (5s only)
    - Kling V1: 35 tokens/sec (5-10s)
    - Sora 2: 15 tokens/sec (4, 8, or 12s only)
    - Sora 2 Pro: 30 tokens/sec (4, 8, or 12s only)
    
    Args:
        prompt: Description of the video to generate (exact user text, NO modifications)
        model: AI model to use. Options: 'runway', 'runway-aleph', 'veo-3.1', 
               'veo-3.1-flash', 'veo-3.1-ultra', 'luma-labs', 'seedance-pro', 'kling-v1', 
               'sora-2', 'sora-2-pro'
        duration: Video duration in seconds. Valid durations depend on model (see above)
        aspect_ratio: '16:9', '9:16', or '1:1'. Defaults to '16:9'
        reference_image: URL of reference image (for image-to-video models)
        reference_video: URL of reference video (only for Runway Aleph video-to-video)
    """
    print(f"DEBUG: MCP Tool 'generate_video' called with prompt='{prompt}', model='{model}', duration={duration}")
    print(f"DEBUG: Gemini passed reference_image='{reference_image}', reference_video='{reference_video}' (IGNORING THESE)")
    
    # Convert duration to int if it's float
    duration = int(duration)
    print(f"DEBUG: Duration converted to int: {duration}")
    
    # Validate model
    allowed_models = [
        "runway", "runway-aleph", "veo-3.1", "veo-3.1-flash", "veo-3.1-ultra",
        "luma-labs", "seedance-pro", "kling-v1", "sora-2", "sora-2-pro"
    ]
    
    if model not in allowed_models:
        return f"Error: Invalid model '{model}'. Allowed models are: {', '.join(allowed_models)}"
    
    # Validate duration per model
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
        "kling-v1": [5, 10]
    }
    
    if model in duration_rules:
        allowed_durations = duration_rules[model]
        if duration not in allowed_durations:
            return f"Error: Duration {duration}s is not valid for model '{model}'. Allowed durations: {allowed_durations} seconds."
    
    print(f"DEBUG: Duration {duration}s validated for model {model}")
    
    # Get backend configuration
    backend_url = os.getenv("BACKEND_URL")
    endpoint = os.getenv("VIDEO_CREATION_ENDPOINT")  # Cambiado de VIDEO_GENERATION_ENDPOINT a VIDEO_CREATION_ENDPOINT
    api_token = get_api_token() or os.getenv("API_TOKEN")
    
    print(f"DEBUG: backend_url={backend_url}, endpoint={endpoint}")
    
    if not backend_url or not endpoint:
        error_msg = f"Error: Backend configuration missing. BACKEND_URL={backend_url}, VIDEO_CREATION_ENDPOINT={endpoint}"
        print(error_msg)
        return error_msg
    
    # Build URL properly (remove trailing slash from backend_url if exists)
    backend_url = backend_url.rstrip('/')
    endpoint = endpoint.lstrip('/')
    url = f"{backend_url}/{endpoint}"
    print(f"DEBUG: Full URL: {url}")
    
    # Get reference images/videos from chatbot session
    from chatbot import get_chatbot
    conversation_uuid = get_conversation_uuid() or "default"
    chatbot = get_chatbot(conversation_uuid)
    context_images = await chatbot.get_reference_images()
    print(f"DEBUG: Retrieved {len(context_images) if context_images else 0} reference media from chatbot session")
    
    # Prepare headers
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    
    # Build payload
    payload = {
        "prompt": prompt,
        "ai_model": model,
        "video_duration": duration,
        "aspect_ratio": aspect_ratio
    }
    
    # Add reference media from context (only if it's a valid URL, not base64)
    if context_images and len(context_images) > 0:
        # Check if first image is a URL or base64
        first_media = context_images[0]
        if first_media.startswith('http://') or first_media.startswith('https://'):
            # It's a URL - use it directly
            payload["media_url"] = first_media
            print(f"DEBUG: Using reference URL: {first_media}")
        elif first_media.startswith('/9j/') or first_media.startswith('iVBOR'):
            # It's base64 - send in different field
            payload["reference_image"] = first_media
            print(f"DEBUG: Using reference image (base64): {first_media[:100]}...")
        else:
            print(f"DEBUG: Ignoring invalid reference media format")
    
    print(f"DEBUG: Sending video generation request with model={model}, duration={duration}s, aspect={aspect_ratio}")
    print(f"DEBUG: Payload: {json.dumps(payload, indent=2)}")
    print(f"DEBUG: Headers: Authorization={'Bearer ***' if api_token else 'None'}, Content-Type=application/json")
    
    try:
        with httpx.Client(timeout=1800.0) as client:  # 30 minutes timeout for long video generation
            print(f"DEBUG: Making POST request to {url}")
            response = client.post(url, json=payload, headers=headers)
            print(f"DEBUG: Response status code: {response.status_code}")
            print(f"DEBUG: Response headers: {dict(response.headers)}")
            
            response.raise_for_status()
            result = response.json()
            
            print(f"DEBUG: Backend response: {result}")
            
            # Extract video URL from response
            video_url = result.get("video_url")
            if video_url:
                print(f"DEBUG: Adding video file to chatbot: {video_url}")
                await chatbot.add_generated_file(video_url, "video")
                
                # Clear reference images after successful use
                await chatbot.clear_reference_images()
                print("DEBUG: Reference images cleared after video generation")
                
                return f"Video generado exitosamente con {model}."
            else:
                print(f"DEBUG: No video_url found in response")
                return f"Video generation initiated but URL not immediately available. Check status later."
                
    except httpx.HTTPStatusError as e:
        error_detail = f"HTTP Error {e.response.status_code}: {e.response.text}"
        print(f"ERROR generating video (HTTP): {error_detail}")
        return f"Error generating video: {error_detail}"
    except httpx.TimeoutException as e:
        error_detail = f"Request timeout after 1800s"
        print(f"ERROR generating video (Timeout): {error_detail}")
        return f"Error generating video: {error_detail}"
    except Exception as e:
        error_detail = f"{type(e).__name__}: {str(e)}"
        print(f"ERROR generating video (General): {error_detail}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return f"Error generating video: {error_detail}"
