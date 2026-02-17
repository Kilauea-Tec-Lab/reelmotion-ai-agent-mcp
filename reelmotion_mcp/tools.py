import os
import httpx
import json
import asyncio
import re
from typing import Optional
from request_context import get_api_token, get_conversation_uuid


def is_blob_url(url: str) -> bool:
    """Check if a URL is a browser-only blob: URL that cannot be fetched server-side."""
    return isinstance(url, str) and url.startswith('blob:')


def clean_prompt_from_model_mentions(prompt: str) -> str:
    """
    Remove model name mentions from the prompt before sending to backend.
    Examples:
        - "anima este video con sora 2" -> "anima este video"
        - "genera una imagen con GPT" -> "genera una imagen"
        - "create a video using runway aleph" -> "create a video"
    """
    if not prompt:
        return prompt
    
    # Patterns to remove (model names with common prefixes)
    # Match patterns like "with sora 2", "con nano banana", "using veo 3.1", etc.
    patterns = [
        # English patterns
        r'\s+(?:with|using|via|through|by)\s+(?:sora[-\s]?2(?:\s+pro)?|runway(?:[-\s]?(?:aleph|4\.?5))?|veo[-\s]?3\.?1(?:[-\s]?(?:flash|ultra))?|nano[-\s]?banana|gpt|freepik|luma[-\s]?labs?|seedance[-\s]?pro|kling[-\s]?(?:v?3[-\s]?omni[-\s]?(?:pro|std)|v1))\s*$',
        # Spanish patterns
        r'\s+(?:con|usando|mediante|por)\s+(?:sora[-\s]?2(?:\s+pro)?|runway(?:[-\s]?(?:aleph|4\.?5))?|veo[-\s]?3\.?1(?:[-\s]?(?:flash|ultra))?|nano[-\s]?banana|gpt|freepik|luma[-\s]?labs?|seedance[-\s]?pro|kling[-\s]?(?:v?3[-\s]?omni[-\s]?(?:pro|std)|v1))\s*$',
        # Just model names at the end (without preposition)
        r'\s+(?:sora[-\s]?2(?:\s+pro)?|runway[-\s]?(?:aleph|4\.?5)|veo[-\s]?3\.?1[-\s]?(?:flash|ultra)|kling[-\s]?v?3[-\s]?omni[-\s]?(?:pro|std))\s*$',
    ]
    
    cleaned = prompt
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Strip trailing whitespace and punctuation that might be left over
    cleaned = cleaned.rstrip(' ,.;:')
    
    print(f"DEBUG: Cleaned prompt from '{prompt}' to '{cleaned}'")
    return cleaned if cleaned else prompt  # Return original if cleaning emptied it


async def generate_image(
    prompt: str, 
    model: str = "GPT", 
    image_type: int = 1, 
    quantity: int = 1, 
    reference_image: Optional[str] = None, 
    reference_images: Optional[list[str]] = None
) -> str:
    """
    Generate or edit an image using the ReelMotion backend.
    This tool supports text-to-image generation AND image-to-image editing/transformation.
    COST: 10 tokens per image generated.
    
    Use cases:
    - Text-to-image: Generate a new image from a text description (type 1).
    - Image-to-image (editing): Transform or edit an existing image using a text prompt + reference image (type 2).
      Examples: change style, add elements, modify colors, remove objects, apply effects.
    - Multi-image reference: Generate using multiple reference images (type 3).
    
    Args:
        prompt: The description of the image to generate, or editing instructions for image-to-image.
        model: The model to use. MUST be one of: 'Nano Banana', 'GPT', 'Freepik'. Defaults to 'GPT'.
        image_type: 1 (text only), 2 (text + reference image for editing), 3 (text + multiple reference images). Defaults to 1.
        quantity: Number of images to generate. Defaults to 1.
        reference_image: URL of reference image (for type 2 - image editing/transformation).
        reference_images: List of URLs of reference images (for type 3 - multi-image reference).
    """
    print(f"DEBUG: MCP Tool 'generate_image' called with prompt='{prompt}', model='{model}'")
    print(f"DEBUG: Gemini passed reference_image='{reference_image}', reference_images='{reference_images}' (IGNORING THESE)")
    
    # Clean prompt from model mentions before sending to backend
    prompt = clean_prompt_from_model_mentions(prompt)
    
    # Validate and normalize model
    allowed_models = ["Nano Banana", "GPT", "Freepik"]
    if model not in allowed_models:
        # Attempt normalization for common variations (e.g. "Nano Banana 2" -> "Nano Banana")
        if "nano" in model.lower():
            model = "Nano Banana"
        elif "gpt" in model.lower():
            model = "GPT"
        elif "freepik" in model.lower():
            model = "Freepik"
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
    print(f"DEBUG [tools/generate_image]: Using conversation_uuid='{conversation_uuid}'")
    chatbot = get_chatbot(conversation_uuid)
    print(f"DEBUG [tools/generate_image]: Chatbot instance uuid='{chatbot.conversation_uuid}'")
    
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
        
        # Filter out blob: URLs (browser-only, cannot be fetched server-side)
        valid_image_files = [f for f in image_files if not is_blob_url(f.get("url", ""))]
        blob_image_files = [f for f in image_files if is_blob_url(f.get("url", ""))]
        
        if blob_image_files:
            print(f"WARNING: Filtered out {len(blob_image_files)} blob: URLs that cannot be processed server-side")
            for bf in blob_image_files:
                print(f"  - Filtered blob URL: {bf.get('url', '')[:80]}...")
        
        if not valid_image_files and blob_image_files:
            # ALL reference images were blob URLs - cannot proceed with image editing
            print(f"ERROR: All {len(blob_image_files)} reference images are blob: URLs. Cannot process.")
            return ("Error: The reference image could not be processed because it uses a temporary browser URL (blob:). "
                    "Please try uploading the image again or use a direct image URL. "
                    "This happens when the image was not properly uploaded to the server.")
        
        image_files = valid_image_files
        print(f"DEBUG: Sending request with {len(image_files)} valid image URLs")
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
            timeout = httpx.Timeout(180.0, connect=10.0)
            with httpx.Client(timeout=timeout) as client:
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
                return f"Images generated successfully with {model}."
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
            timeout = httpx.Timeout(180.0, connect=10.0)
            with httpx.Client(timeout=timeout) as client:
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
                return f"Images generated successfully with {model}."
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
    Generate or edit a video using AI based on a text prompt.
    This tool supports text-to-video, image-to-video, AND video-to-video editing.
    IMPORTANT: User must confirm model, duration, and token cost before calling this tool.
    
    Use cases:
    - Text-to-video: Generate a new video from a text description.
    - Image-to-video: Animate a reference image into a video.
    - Video-to-video (editing): Transform or edit an existing video using a prompt + reference video.
      Supported models for video-to-video: runway-aleph, kling-v3-omni-std, kling-v3-omni-pro.
    
    Token costs per second:
    - Runway: 8 tokens/sec (5-10s) - image-to-video
    - Runway Aleph: 19 tokens/sec (5-10s) - video-to-video editing
    - Veo 3.1: 48 tokens/sec (8s only)
    - Veo 3.1 Flash: 21 tokens/sec (8s only)
    - Veo 3.1 Ultra: 60 tokens/sec (8s only) - maximum quality
    - Luma Labs: 13 tokens/sec (5s only)
    - Seedance Pro: 15 tokens/sec (5s only)
    - Kling V1: 35 tokens/sec (5-10s)
    - Kling V3 Omni Pro: 8 tokens/sec (3-15s) - text/image-to-video
    - Kling V3 Omni Std: 6 tokens/sec (3-15s) - video-to-video editing
    - Sora 2: 15 tokens/sec (4, 8, or 12s only)
    - Sora 2 Pro: 30 tokens/sec (4, 8, or 12s only)
    - Runway 4.5: 25 tokens/sec (5, 8, or 10s only)
    
    Args:
        prompt: Description of the video to generate or editing instructions (exact user text, NO modifications)
        model: AI model to use. Options: 'runway', 'runway-aleph', 'runway-4.5', 'veo-3.1', 
               'veo-3.1-flash', 'veo-3.1-ultra', 'luma-labs', 'seedance-pro', 'kling-v1', 
               'kling-v3-omni-pro', 'kling-v3-omni-std', 'sora-2', 'sora-2-pro'
        duration: Video duration in seconds. Valid durations depend on model (see above)
        aspect_ratio: '16:9', '9:16', or '1:1'. Defaults to '16:9'
        reference_image: URL of reference image (for image-to-video generation)
        reference_video: URL of reference video (for video-to-video editing with runway-aleph, kling-v3)
    """
    print(f"DEBUG: MCP Tool 'generate_video' called with prompt='{prompt}', model='{model}', duration={duration}")
    print(f"DEBUG: Gemini passed reference_image='{reference_image}', reference_video='{reference_video}' (IGNORING THESE)")
    
    # Clean prompt from model mentions before sending to backend
    prompt = clean_prompt_from_model_mentions(prompt)
    
    # Convert duration to int if it's float
    duration = int(duration)
    print(f"DEBUG: Duration converted to int: {duration}")
    
    # Validate model
    allowed_models = [
        "runway", "runway-aleph", "runway-4.5", "veo-3.1", "veo-3.1-flash", "veo-3.1-ultra",
        "luma-labs", "seedance-pro", "kling-v1", "sora-2", "sora-2-pro",
        "kling-v3-omni-pro", "kling-v3-omni-std"
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
        "runway-4.5": [5, 8, 10],
        "kling-v1": [5, 10],
        "kling-v3-omni-pro": list(range(3, 16)),  # 3 to 15 seconds
        "kling-v3-omni-std": list(range(3, 16))   # 3 to 15 seconds
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
    print(f"DEBUG [tools/generate_video]: Using conversation_uuid='{conversation_uuid}'")
    chatbot = get_chatbot(conversation_uuid)
    print(f"DEBUG [tools/generate_video]: Chatbot instance uuid='{chatbot.conversation_uuid}'")
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
        # Filter out blob: URLs (browser-only, cannot be fetched server-side)
        valid_context_files = [f for f in context_files if not is_blob_url(f.get("url", ""))]
        blob_files = [f for f in context_files if is_blob_url(f.get("url", ""))]
        if blob_files:
            print(f"WARNING [generate_video]: Filtered out {len(blob_files)} blob: URLs")
        
        if not valid_context_files and blob_files:
            print(f"ERROR [generate_video]: All reference files are blob: URLs. Cannot process.")
            return ("Error: The reference file could not be processed because it uses a temporary browser URL (blob:). "
                    "Please try uploading the file again or use a direct URL.")
        
        # Buscar imagen o video en los archivos de referencia válidos
        image_file = next((f for f in valid_context_files if f.get("type") == "image"), None)
        video_file = next((f for f in valid_context_files if f.get("type") == "video"), None)
        
        processed_video = False
        if video_file:
            if model == "runway-aleph":
                # Runway Aleph acepta video-to-video
                payload["reference_video"] = video_file["url"]
                print(f"DEBUG: Using reference video URL (runway-aleph): {video_file['url']}")
                processed_video = True
            elif model in ["kling-v3-omni-std", "kling-v3-omni-pro"]:
                # Kling V3 acepta video-to-video via media_url
                payload["media_url"] = video_file["url"]
                print(f"DEBUG: Using reference video URL (kling-v3): {video_file['url']}")
                processed_video = True
        
        if not processed_video and image_file:
            # Usar imagen de referencia
            payload["reference_image"] = image_file["url"]
            print(f"DEBUG: Using reference image URL: {image_file['url']}")
    
    print(f"DEBUG: Sending video generation request with model={model}, duration={duration}s, aspect={aspect_ratio}")
    print(f"DEBUG: Payload: {json.dumps(payload, indent=2)}")
    print(f"DEBUG: Headers: Authorization={'Bearer ***' if api_token else 'None'}, Content-Type=application/json")
    
    try:
        with httpx.Client(timeout=12000.0) as client:  # 200 minutes timeout for very long video generation
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
                
                return f"Video generated successfully with {model}."
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
                 Available voices:
                 - Adam (Male, American, Deep): pNInz6obpgDQGcFmaJgB
                 - Alice (Female, British, News): Xb7hH8MSUJpSbSDYk0k2
                 - Antoni (Male, American, Balanced): ErXwobaYiN019PkySvjV
                 - Bill (Male, American, Trustworthy): pqHfZKP75CvOlQylNhV4
                 - Brian (Male, American, Deep): nPczCjzI2devNBz1zQrb
                 - Callum (Male, American, Hoarse): N2lVS1w4EtoT3dr4eOWO
                 - Charlie (Male, Australian, Casual): IKne3meq5aSn9XLyUdCD
                 - Chris (Male, American, Casual): iP95p4xoKVk53GoZ742B
                 - Daniel (Male, British, Authoritative): onwK4e9ZLuTAKqWW03F9
                 - Domi (Female, American, Strong): AZnzlk1XvdvUeBnXmlld
                 - Elli (Female, American, Young): MF3mGyEYCl7XYWbV9V6O
                 - Eric (Male, American, Deep): cjVigY5qzO86Huf0OWal
                 - George (Male, British, Warm): JBFqnCBsd6RMkjVDRZzb
                 - Harry (Male, American, Anxious): SOYHLrjzK2X1ezoPC6cr
                 - Jessica (Female, American, Expressive): cgSgspJ2msm6clMCkdW9
                 - Josh (Male, American, Deep): TxGEqnHWrfWFTfGW9XjX
                 - Laura (Female, American, Upbeat): FGY2WhTYpPnrIDTdsKH5
                 - Liam (Male, American, Young): TX3LPaxmHKxFdv7VOQHJ
                 - Lily (Female, British, Warm): pFZP5JQG7iQjIQuC4Bku
                 - Matilda (Female, American, Warm): XrExE9yKIg1WjnnlVkGX
                 - Rachel (Female, American, Calm): 21m00Tcm4TlvDq8ikWAM
                 - River (Male, American, Neutral): SAz9YHcvj6GT2YYXdXww
                 - Roger (Male, American, Casual): CwhRBWXzGAHq8TQ4Fs17
                 - Sarah (Female, American, Soft): EXAVITQu4vr4xnSDxMaL
                 - Will (Male, American, Friendly): bIHbv24MWmeRgasZH58o
        model_id: The model to use. Defaults to "eleven_multilingual_v2".
    """
    import base64
    import uuid
    
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
            
            # Calculate tokens: fixed cost of 5 tokens per voice generation
            tokens_cost = 5
            
            print(f"DEBUG [generate_speech]: Calling backend at {callback_url} with {tokens_cost} tokens")
            
            # Backend validation requires a valid URL (http/https). 
            # Since we have a Data URI (base64) which Laravel rejects, we send a placeholder URL
            # for the callback. The actual audio is already delivered to the user via chatbot session.
            backend_audio_url = audio_data_uri
            if audio_data_uri.startswith("data:"):
                # Use a dummy URL that passes validation
                req_uuid = str(uuid.uuid4())
                backend_audio_url = f"https://reelmotion.ai/generated/audio/{req_uuid}.mp3"
                print(f"DEBUG [generate_speech]: Swapped Data URI for placeholder URL for backend: {backend_audio_url}")

            callback_payload = {
                "audio_url": backend_audio_url,
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

        return f"Audio generated successfully ({len(audio_content)} bytes). Link generated automatically."
            
    except Exception as e:
        print(f"ERROR generating speech: {e}")
        return f"Error generating speech: {str(e)}"
