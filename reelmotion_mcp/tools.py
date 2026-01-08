import os
import httpx
import json
import asyncio
from typing import Optional
from request_context import get_api_token, get_conversation_uuid

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
    
    # Get reference files from chatbot session (URLs, not base64)
    from chatbot import get_chatbot
    conversation_uuid = get_conversation_uuid() or "default"
    chatbot = get_chatbot(conversation_uuid)
    
    # Get reference files (URLs) asynchronously
    context_files = await chatbot.get_reference_files()
    print(f"DEBUG: Retrieved {len(context_files) if context_files else 0} files from chatbot session")

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
    
    # If we have files from context, send URLs directly
    if context_files:
        # Filtrar solo imágenes
        image_files = [f for f in context_files if f.get("type") == "image"]
        print(f"DEBUG: Sending request with {len(image_files)} image URLs")
        headers["Content-Type"] = "application/json"
        
        payload = {
            "prompt": prompt,
            "model": model,
            "type": image_type,
            "quantity": quantity,
        }
        
        # Send image URLs directly (no download, no base64)
        if len(image_files) == 1:
            # Single image: use reference_image field
            payload["reference_image"] = image_files[0]["url"]
        elif len(image_files) > 1:
            # Multiple images: use reference_images array
            payload["reference_images"] = [f["url"] for f in image_files]

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
                
                # Clear reference files after successful use
                await chatbot.clear_reference_files()
                print("DEBUG: Reference files cleared after use")
                
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
    
    # Get reference files from chatbot session (URLs, not base64)
    from chatbot import get_chatbot
    conversation_uuid = get_conversation_uuid() or "default"
    chatbot = get_chatbot(conversation_uuid)
    context_files = await chatbot.get_reference_files()
    print(f"DEBUG: Retrieved {len(context_files) if context_files else 0} reference files from chatbot session")
    
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
    
    # Add reference media from context (URLs directly)
    if context_files and len(context_files) > 0:
        # Buscar imagen o video en los archivos de referencia
        image_file = next((f for f in context_files if f.get("type") == "image"), None)
        video_file = next((f for f in context_files if f.get("type") == "video"), None)
        
        if video_file and model == "runway-aleph":
            # Runway Aleph acepta video-to-video
            payload["reference_video"] = video_file["url"]
            print(f"DEBUG: Using reference video URL: {video_file['url']}")
        elif image_file:
            # Usar imagen de referencia
            payload["reference_image"] = image_file["url"]
            print(f"DEBUG: Using reference image URL: {image_file['url']}")
    
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
                
                # Clear reference files after successful use
                await chatbot.clear_reference_files()
                print("DEBUG: Reference files cleared after video generation")
                
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

async def generate_speech(
    text: str,
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel
    model_id: str = "eleven_multilingual_v2"
) -> str:
    """
    Generate speech from text using ElevenLabs API.
    
    Args:
        text: The text to convert to speech.
        voice_id: The ID of the voice to use. Defaults to "Rachel" (21m00Tcm4TlvDq8ikWAM).
        model_id: The model to use. Defaults to "eleven_multilingual_v2".
    """
    import base64
    
    # Use the key provided by the user, or env var
    api_key = os.getenv("ELEVENLABS_API_KEY", "sk_2255a4e8aaeaf2c8211f2ffc968686b602250cd260314f16")
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
    data = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    print(f"DEBUG: Generating speech for text: '{text[:50]}...' with voice {voice_id}")
    
    try:
        # First: Generate speech with ElevenLabs
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data, headers=headers)
            response.raise_for_status()
            audio_content = response.content
            
        # Convert binary audio to base64 Data URI
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        audio_data_uri = f"data:audio/mpeg;base64,{audio_base64}"
        
        # Save to chatbot session
        from chatbot import get_chatbot
        from request_context import get_conversation_uuid
        
        conversation_uuid = get_conversation_uuid() or "default"
        chatbot = get_chatbot(conversation_uuid)
        
        print(f"DEBUG: Speech generated size: {len(audio_content)} bytes")
        await chatbot.add_generated_file(audio_data_uri, "audio")
        
        # --- BACKEND CALLBACK START ---
        # Consume the backend endpoint to register tool usage and deduct tokens
        backend_url = os.getenv("BACKEND_URL")
        api_token = get_api_token() or os.getenv("API_TOKEN")

        print(f"DEBUG [generate_speech]: backend_url={backend_url}, api_token={'SET' if api_token else 'NOT SET'}")

        if backend_url and api_token:
            # Ensure backend_url doesn't end with slash if we append
            if backend_url.endswith("/"):
                backend_url = backend_url[:-1]
            
            # Construct endpoint URL
            callback_url = f"{backend_url}/api/ai/mcp-voice-generation"
            
            # Calculate tokens: ~1 token per 10 characters, min 10
            calculated_tokens = len(text) // 10
            tokens_cost = max(10, calculated_tokens)
            
            print(f"DEBUG [generate_speech]: Calling backend at {callback_url} with {tokens_cost} tokens")
            
            callback_payload = {
                "audio_url": audio_data_uri,
                "tokens": tokens_cost
            }
            
            callback_headers = {
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            # Use a SEPARATE client for the backend callback
            async with httpx.AsyncClient(timeout=30.0) as callback_client:
                try:
                    callback_response = await callback_client.post(callback_url, json=callback_payload, headers=callback_headers)
                    print(f"DEBUG [generate_speech]: Backend response status: {callback_response.status_code}")
                    if callback_response.status_code >= 200 and callback_response.status_code < 300:
                        print(f"DEBUG [generate_speech]: Backend callback SUCCESS: {callback_response.text[:200]}")
                    else:
                        print(f"WARNING [generate_speech]: Backend callback FAILED: {callback_response.status_code} - {callback_response.text}")
                except Exception as cb_exc:
                    print(f"ERROR [generate_speech]: Exception calling backend: {cb_exc}")
        else:
            print(f"WARNING [generate_speech]: Skipping backend callback - backend_url={backend_url}, api_token={'SET' if api_token else 'NOT SET'}")
        # --- BACKEND CALLBACK END ---

        return f"Audio generado exitosamente ({len(audio_content)} bytes). Enlace generado automáticamente."
            
    except Exception as e:
        print(f"ERROR generating speech: {e}")
        return f"Error generating speech: {str(e)}"
