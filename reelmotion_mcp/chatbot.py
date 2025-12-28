import os
import base64
import time
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv

from prompts import REELMOTION_SYSTEM_PROMPT
from tools import generate_image, generate_video
from session_manager import get_session_manager

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class GeminiChatbot:
    def __init__(self, model_name: str = None, conversation_uuid: str = None):
        """
        Initialize the Gemini chatbot.
        
        Args:
            model_name: The Gemini model to use (defaults to gemini-2.5-flash from .env)
            conversation_uuid: UUID de la conversación (para multi-sesión)
        """
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.conversation_uuid = conversation_uuid or "default"
        self.session_manager = get_session_manager()
        
        # Critical tool usage instructions
        tool_instructions = """
        LANGUAGE ADAPTATION:
        - Default language: English
        - AUTOMATICALLY DETECT the user's language from their messages
        - RESPOND in the SAME language the user is using
        - If user writes in Spanish, respond in Spanish
        - If user writes in English, respond in English
        - If user switches language, switch with them immediately
        - Keep technical terms and model names in their original form (e.g., "Nano Banana", "GPT")
        
        CRITICAL RULES FOR 'generate_image' TOOL:
        1. The 'prompt' parameter MUST BE EXACTLY the LITERAL TEXT the user entered.
        2. FORBIDDEN to modify, improve, summarize, translate, or reinterpret the user's text for the prompt.
        3. If user writes "Change the suit", the prompt sent to the tool must be "Change the suit".
        4. If there are attached images, always pass them in 'reference_images'.
        5. BEFORE calling generate_image, you MUST ALWAYS:
           a) Ask: "Which model do you want to use: Nano Banana or GPT?" (in user's language)
           b) Wait for user's response with chosen model
           c) Inform the cost: "This will cost X tokens (10 tokens per image × quantity)" (in user's language)
           d) Wait for explicit confirmation before proceeding
        6. Available models are ONLY: 'Nano Banana' and 'GPT'. Freepik is NO longer available.
        7. DO NOT assume the model - ALWAYS ask the user which one to use.
        8. NEVER mention URLs in your responses - images/videos are sent automatically to the user.
        9. IF THERE'S AN ERROR: Inform the user of the error, but if user says "try again" or "retry" (or "intentar de nuevo", "reintentar"),
           you MUST execute the tool again without hesitation.
        
        CRITICAL RULES FOR 'generate_video' TOOL:
        1. The 'prompt' parameter MUST BE EXACTLY the LITERAL TEXT the user entered.
        2. FORBIDDEN to modify, improve, summarize, translate, or reinterpret the user's text.
        3. BEFORE calling generate_video, you MUST ALWAYS:
           a) Ask: "Which video model do you want to use?" (in user's language) and list options with costs AND DURATIONS:
              - Runway Aleph (19 tokens/sec) - 5 or 10 seconds - video-to-video (editing)
              - Veo 3.1 (48 tokens/sec) - 8 seconds - high quality
              - Veo 3.1 Flash (21 tokens/sec) - 8 seconds - fast and economical
              - Veo 3.1 Ultra (60 tokens/sec) - 8 seconds - maximum Veo quality
              - Sora 2 (15 tokens/sec) - ONLY 4, 8 or 12 seconds
              - Sora 2 Pro (30 tokens/sec) - ONLY 4, 8 or 12 seconds - maximum quality
           b) Wait for the user to choose the model
           c) Ask: "How many seconds duration?" (in user's language) and MENTION valid options for the chosen model
           d) Wait for duration
           e) VALIDATE that duration is compatible with the model:
              - Sora 2 / Sora 2 Pro: ONLY 4, 8 or 12 seconds
              - Veo 3.1 / Veo 3.1 Flash / Veo 3.1 Ultra: ONLY 8 seconds
              - Runway Aleph: 5 or 10 seconds
           f) If duration is NOT valid, inform user of correct options and ask to choose a valid one
           g) Calculate and show: "This will cost X tokens (Y tokens/sec × Z seconds). Confirm?" (in user's language)
           h) Wait for explicit confirmation before proceeding
        4. IMPORTANT: When calling the tool, use EXACT names:
           - 'veo-3.1' (NOT 'veo 3.1' or 'Veo 3.1')
           - 'veo-3.1-flash' (NOT 'veo 3.1 flash')
           - 'veo-3.1-ultra' (NOT 'veo 3.1 ultra')
           - 'runway-aleph', 'sora-2', 'sora-2-pro'
        5. If there are attached images, use them as reference automatically.
        6. For Runway Aleph, if there's an attached VIDEO, use it as reference (video-to-video).
        7. NEVER mention video URLs in your responses - they are sent automatically.
        8. IF THERE'S AN ERROR (404, timeout, missing config, etc.):
           - Inform user of the error clearly and simply (in user's language)
           - Explain possible causes (e.g., "The endpoint doesn't exist in backend", "Missing configuration")
           - IF user says "try again", "retry", "intentar de nuevo", "reintentar", or similar,
             you MUST execute the tool again WITHOUT QUESTIONING, using the same parameters.
        9. Reference images are NOT lost after errors - they persist in the session.
        10. ALWAYS try when the user asks, even if there were previous errors.
        """
        
        full_system_prompt = f"{REELMOTION_SYSTEM_PROMPT}\n\n{tool_instructions}"
        
        self.model = genai.GenerativeModel(
            self.model_name, 
            system_instruction=full_system_prompt,
            tools=[generate_image, generate_video]
        )
        self.chat_session = None
        
    async def start_chat(self):
        """Start a new chat session with history from Redis."""
        # Cargar historial desde Redis
        session = await self.session_manager.get_session(self.conversation_uuid)
        
        history = []
        if session and session.get("messages"):
            # Convertir mensajes a formato Gemini
            for msg in session["messages"]:
                # Gemini espera role: 'user' o 'model' (no 'assistant')
                role = "model" if msg["role"] == "assistant" else msg["role"]
                history.append({
                    "role": role,
                    "parts": [msg["content"]]
                })
        
        self.chat_session = self.model.start_chat(
            history=history,
            enable_automatic_function_calling=False
        )
    
    async def set_reference_files(self, file_urls: list, file_types: list = None):
        """Store reference file URLs in Redis for this chat session."""
        if not file_types:
            file_types = ["image"] * len(file_urls)
        
        # Store as list of dicts with URL and type
        files_data = [
            {"url": url, "type": file_type}
            for url, file_type in zip(file_urls, file_types)
        ]
        await self.session_manager.save_reference_files(self.conversation_uuid, files_data)
    
    async def get_reference_files(self) -> list:
        """Get stored reference file URLs from Redis."""
        return await self.session_manager.get_reference_files(self.conversation_uuid)
    
    async def clear_reference_files(self):
        """Clear stored reference files."""
        refs_key = self.session_manager._get_refs_key(self.conversation_uuid)
        self.session_manager.redis_client.delete(refs_key)
    
    # Mantener compatibilidad con métodos antiguos
    async def set_reference_images(self, images: list):
        """Legacy method - converts to URLs if needed."""
        # Si recibe URLs directamente, usarlas
        if images and all(isinstance(img, str) for img in images):
            await self.set_reference_files(images)
        else:
            # Compatibilidad con formato antiguo (no debería usarse)
            pass
    
    async def get_reference_images(self) -> list:
        """Legacy method - returns file URLs."""
        files = await self.get_reference_files()
        return [f["url"] for f in files] if files else []
    
    async def clear_reference_images(self):
        """Legacy method."""
        await self.clear_reference_files()
    
    async def add_generated_file(self, url: str, file_type: str = "image", metadata: dict = None):
        """Add a generated file URL to pending files in Redis."""
        if url:
            await self.session_manager.save_generated_file(
                self.conversation_uuid,
                url,
                file_type,
                metadata
            )
    
    async def get_generated_files(self) -> list:
        """Get pending generated files (URLs) from Redis."""
        files = await self.session_manager.get_pending_files(self.conversation_uuid)
        return [{"url": f["url"], "type": f["type"]} for f in files]
        
    async def send_message(self, message: str, context: str = "", images: list = None, file_urls: list = None, file_types: list = None) -> str:
        """
        Send a message to the Gemini chatbot and get a response.
        
        Args:
            message: The user's message
            context: Optional context or conversation history
            images: DEPRECATED - Optional list of image data
            file_urls: Optional list of file URLs
            file_types: Optional list of file types (image, video, etc.)
            
        Returns:
            The chatbot's response
        """
        try:
            if not self.chat_session:
                await self.start_chat()
            
            # Guardar mensaje del usuario en Redis
            await self.session_manager.add_message(
                self.conversation_uuid,
                "user",
                message
            )
            
            # Prepare message parts
            parts = []
            
            # Add context if provided
            if context:
                parts.append(f"Context: {context}\n\n")
            
            # Add reference files for Gemini to analyze
            ref_files = await self.get_reference_files()
            if ref_files:
                # Download and send files to Gemini for analysis
                for file_data in ref_files:
                    file_url = file_data['url']
                    file_type = file_data.get('type', 'image')
                    
                    try:
                        print(f"DEBUG: Downloading {file_type} from {file_url} for Gemini analysis")
                        import httpx
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            response = await client.get(file_url)
                            response.raise_for_status()
                            file_bytes = response.content
                            
                            # Determine MIME type
                            mime_type = file_type
                            if file_type == 'image':
                                # Detectar formato de imagen
                                if file_url.endswith('.png'):
                                    mime_type = 'image/png'
                                elif file_url.endswith('.webp'):
                                    mime_type = 'image/webp'
                                elif file_url.endswith('.gif'):
                                    mime_type = 'image/gif'
                                else:
                                    mime_type = 'image/jpeg'
                            elif file_type == 'video':
                                if file_url.endswith('.webm'):
                                    mime_type = 'video/webm'
                                elif file_url.endswith('.mov'):
                                    mime_type = 'video/mov'
                                else:
                                    mime_type = 'video/mp4'
                            
                            # Add file to parts for Gemini to analyze
                            parts.append({
                                'mime_type': mime_type,
                                'data': file_bytes
                            })
                            print(f"DEBUG: Added {file_type} ({mime_type}) to Gemini message for analysis")
                    except Exception as e:
                        print(f"ERROR: Failed to download {file_type} for analysis: {e}")
                        parts.append(f"Note: Unable to load {file_type} from URL. Error: {str(e)}\n\n")
            
            # Add the user's message
            parts.append(message)
            
            # Send to Gemini
            response = await self.chat_session.send_message_async(parts)
            
            # Handle function calls manually
            while response.parts and response.parts[0].function_call:
                fc = response.parts[0].function_call
                func_name = fc.name
                func_args = dict(fc.args)
                
                print(f"DEBUG: Handling function call: {func_name}")
                
                tool_result = "Error: Unknown function"
                try:
                    if func_name == "generate_image":
                        tool_result = await generate_image(**func_args)
                    elif func_name == "generate_video":
                        tool_result = await generate_video(**func_args)
                except Exception as e:
                    tool_result = f"Error executing {func_name}: {str(e)}"
                
                # Send result back
                response = await self.chat_session.send_message_async(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=func_name,
                            response={"result": tool_result}
                        )
                    )
                )
            
            # Guardar respuesta del asistente en Redis
            await self.session_manager.add_message(
                self.conversation_uuid,
                "assistant",
                response.text
            )
            
            return response.text
            
        except Exception as e:
            error_msg = f"Error communicating with Gemini: {str(e)}"
            # Guardar error en historial
            await self.session_manager.add_message(
                self.conversation_uuid,
                "assistant",
                error_msg
            )
            return error_msg
    
    async def reset_chat(self):
        """Reset the chat session and clear Redis data."""
        self.chat_session = None
        await self.session_manager.delete_session(self.conversation_uuid)


# Cache de chatbots por UUID con timestamp de último acceso
_chatbot_instances: dict[str, GeminiChatbot] = {}
_last_access: dict[str, float] = {}

def get_chatbot(conversation_uuid: str = "default") -> GeminiChatbot:
    """Get or create a chatbot instance for a specific conversation."""
    
    # Limpieza simple: si hay más de 1000 instancias en memoria, borrar las viejas
    if len(_chatbot_instances) > 1000:
        current_time = time.time()
        # Borrar instancias que no se usan hace más de 1 hora (3600 segundos)
        keys_to_delete = [k for k, t in _last_access.items() if current_time - t > 3600]
        for k in keys_to_delete:
            if k in _chatbot_instances:
                del _chatbot_instances[k]
            if k in _last_access:
                del _last_access[k]
    
    if conversation_uuid not in _chatbot_instances:
        _chatbot_instances[conversation_uuid] = GeminiChatbot(conversation_uuid=conversation_uuid)
    
    # Actualizar tiempo de último acceso
    _last_access[conversation_uuid] = time.time()
    
    return _chatbot_instances[conversation_uuid]
