import os
import base64
import time
import re
import asyncio
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv

from prompts import REELMOTION_SYSTEM_PROMPT
from tools import generate_image, generate_video, generate_speech
from session_manager import get_session_manager
from request_context import set_conversation_uuid

# Load environment variables
load_dotenv()

# Pattern matching for confirmations
CONFIRMATION_PATTERNS = re.compile(
    r'^(?:ok|okey|okay|si|sí|yes|dale|confirmo|confirmar|procede|proceed|hazlo|do it|'
    r'adelante|claro|sure|yep|yeah|afirmativo|correcto|eso|exacto|perfecto|listo|va|'
    r'venga|vamos|go|go ahead|lets go|let\'s go|bueno|bien|hecho|agreed|confirm|acepto|'
    r'accept|y|s|1|👍|✅|done|ready|ok dale|si dale|ya|anda|órale|sale)[\s.,!?]*$',
    re.IGNORECASE
)

# Pattern to detect when Gemini is asking for confirmation (cost question)
COST_CONFIRMATION_PATTERN = re.compile(
    r'(?:costará|cost|costar[aá]n|tokens?|créditos?|credits?|¿confirmas?|confirm|proceder|proceed)',
    re.IGNORECASE
)

# Patterns to detect video model in conversation
VIDEO_MODEL_PATTERNS = {
    'sora-2': re.compile(r'\b(?:sora[-\s]?2(?!\s*pro))\b', re.IGNORECASE),
    'sora-2-pro': re.compile(r'\bsora[-\s]?2[-\s]?pro\b', re.IGNORECASE),
    'veo-3.1': re.compile(r'\bveo[-\s]?3\.?1(?!\s*(?:flash|ultra))\b', re.IGNORECASE),
    'veo-3.1-flash': re.compile(r'\bveo[-\s]?3\.?1[-\s]?flash\b', re.IGNORECASE),
    'veo-3.1-ultra': re.compile(r'\bveo[-\s]?3\.?1[-\s]?ultra\b', re.IGNORECASE),
    'runway-aleph': re.compile(r'\brunway[-\s]?aleph\b', re.IGNORECASE),
    'runway-4.5': re.compile(r'\brunway[-\s]?4\.?5\b', re.IGNORECASE),
    'kling-v3-omni-pro': re.compile(r'\bkling[-\s]?v?3[-\s]?omni[-\s]?pro\b', re.IGNORECASE),
    'kling-v3-omni-std': re.compile(r'\bkling[-\s]?v?3[-\s]?omni[-\s]?std\b', re.IGNORECASE),
}

# Patterns to extract duration
DURATION_PATTERN = re.compile(r'(\d+)\s*(?:segundos?|seconds?|sec|s\b)', re.IGNORECASE)

# Patterns to detect image model in conversation
IMAGE_MODEL_PATTERNS = {
    'GPT': re.compile(r'\bgpt\b', re.IGNORECASE),
    'Nano Banana': re.compile(r'\bnano[-\s]?banana\b', re.IGNORECASE),
    'Freepik': re.compile(r'\bfreepik\b', re.IGNORECASE),
}

def is_confirmation(message: str) -> bool:
    """Check if the message is a simple confirmation."""
    cleaned = message.strip().lower()
    return bool(CONFIRMATION_PATTERNS.match(cleaned))

def needs_clarification(message: str, has_ref_files: bool) -> tuple[bool, str]:
    """
    Check if a message is ambiguous and needs clarification.
    Returns (needs_clarification, suggested_question)
    """
    msg_lower = message.lower().strip()
    
    # === CLEAR INTENT: VIDEO MODEL MENTIONED ===
    video_model_keywords = ['sora', 'veo', 'kling', 'runway', 'haiper', 'minimax', 'aleph']
    if any(model in msg_lower for model in video_model_keywords):
        return False, ""  # Clear: wants to generate VIDEO
    
    # === CLEAR INTENT: IMAGE MODEL MENTIONED ===
    image_model_keywords = ['gpt', 'nano', 'banana', 'freepik']
    if any(model in msg_lower for model in image_model_keywords):
        return False, ""  # Clear: wants to generate IMAGE
    
    # === CLEAR INTENT: ACTION KEYWORDS ===
    video_action_keywords = ['video', 'anima', 'animate', 'movimiento', 'movement', 'mover']
    if any(keyword in msg_lower for keyword in video_action_keywords):
        return False, ""  # Clear: wants video
    
    image_action_keywords = ['imagen', 'image', 'foto', 'photo', 'picture', 'draw']
    if any(keyword in msg_lower for keyword in image_action_keywords):
        return False, ""  # Clear: wants image
    
    # === CLEAR INTENT: NUMBERS (likely answering duration/settings) ===
    if re.match(r'^\d+$', msg_lower):
        return False, ""  # Answering a question about duration/settings
    
    # === CLEAR INTENT: CONFIRMATION ===
    if is_confirmation(msg_lower):
        return False, ""  # Confirming an action
    
    # === AMBIGUOUS: Very short/generic messages ===
    if len(msg_lower) < 10 and not has_ref_files:
        # Generic creation requests without context
        if any(word in msg_lower for word in ['crea', 'genera', 'haz', 'make', 'create', 'generate']):
            return True, "¿Qué quieres crear exactamente? ¿Una imagen o un video?"
    
    # === AMBIGUOUS: Has reference file but VERY unclear what to do ===
    if has_ref_files and len(msg_lower) < 5:
        # Only very vague single words without context
        vague_words = ['eso', 'esto', 'that', 'this', 'aquí', 'here']
        if msg_lower in vague_words:
            return True, "¿Qué quieres hacer con esta imagen? ¿Generar un video a partir de ella o crear una nueva imagen similar?"
    
    return False, ""

def detect_video_params_from_history(history: list) -> dict:
    """
    Try to extract video generation parameters from conversation history.
    Returns dict with 'model', 'duration', 'prompt' if found.
    """
    params = {}
    
    # Search recent messages (last 10) for model and duration
    recent = history[-10:] if len(history) > 10 else history
    
    # Detect model from MOST RECENT messages first (priority to latest mention)
    # This prevents older model mentions from overriding the current one
    for msg in reversed(recent):
        content = msg.get('content', '')
        for model_name, pattern in VIDEO_MODEL_PATTERNS.items():
            if pattern.search(content):
                params['model'] = model_name
                break
        if 'model' in params:
            break
    
    # Detect duration ONLY from ASSISTANT messages (where we mention the cost/duration)
    # This prevents picking up random numbers from user messages
    for msg in reversed(recent):
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            # Priority 1: Look for cost calculation pattern like "X tokens/seg × Y segundos" or "X tokens/second * Y seconds"
            cost_match = re.search(r'[×x\*]\s*(\d+)\s*(?:segundos?|seconds?)', content, re.IGNORECASE)
            if cost_match:
                params['duration'] = int(cost_match.group(1))
                break
            # Priority 2: Look for "video de X segundos" pattern
            video_dur_match = re.search(r'(?:video\s+de|duración\s+de?)\s*(\d+)\s*(?:segundos?|seconds?)', content, re.IGNORECASE)
            if video_dur_match:
                params['duration'] = int(video_dur_match.group(1))
                break
    
    # If no duration found in assistant messages, try user messages (but be more careful)
    if 'duration' not in params:
        for msg in reversed(recent):
            if msg.get('role') == 'user':
                content = msg.get('content', '').strip()
                # Only match standalone duration (e.g., "4", "8 segundos", "5s")
                if re.match(r'^\d+\s*(?:segundos?|seconds?|seg|sec|s)?[\s.,!?]*$', content, re.IGNORECASE):
                    duration_match = re.match(r'^(\d+)', content)
                    if duration_match:
                        params['duration'] = int(duration_match.group(1))
                        break
    
    # Get the prompt (most recent user message that's not a confirmation or model/duration selection)
    for msg in reversed(recent):
        if msg.get('role') == 'user':
            content = msg.get('content', '')
            # Skip confirmation messages
            if is_confirmation(content):
                continue
            # Skip STANDALONE model selection (just "sora 2" alone, "kling v3 omni pro", etc.)
            if re.match(r'^\s*(sora[-\s]?2[-\s]?(?:pro)?|veo[-\s]?3\.?1[-\s]?(?:flash|ultra)?|kling[-\s]?(?:v?3[-\s]?omni[-\s]?(?:pro|std))?|runway[-\s]?(?:aleph|4\.?5)?|haiper|minimax)\s*[.,!?]*$', content, re.IGNORECASE):
                continue
            # Skip duration-only messages like "5 seconds", "5s", or just "4"
            if re.match(r'^\s*\d+\s*(?:segundos?|seconds?|seg|sec|s)?\s*[\.!?]*$', content, re.IGNORECASE):
                continue
            # This is likely the actual prompt - USE IT AS IS, don't modify it
            if len(content) > 3:
                params['prompt'] = content
                break
    
    return params

def detect_image_params_from_history(history: list) -> dict:
    """
    Try to extract image generation parameters from conversation history.
    Returns dict with 'model', 'prompt' if found.
    """
    params = {}
    
    # Search recent messages for model
    recent = history[-10:] if len(history) > 10 else history
    full_text = ' '.join([msg.get('content', '') for msg in recent])
    
    # Detect model
    for model_name, pattern in IMAGE_MODEL_PATTERNS.items():
        if pattern.search(full_text):
            params['model'] = model_name
            break
    
    # Get the prompt (most recent user message that's not a confirmation or model selection)
    for msg in reversed(recent):
        if msg.get('role') == 'user':
            content = msg.get('content', '')
            # Skip confirmation messages
            if is_confirmation(content):
                continue
            # Skip model selection messages
            if re.match(r'^(gpt|nano[-\s]?banana)[\s.,!?]*$', content, re.IGNORECASE):
                continue
            # Skip generic "create image" type messages (too vague)
            if re.match(r'^(i want to create|quiero crear|genera?r?)\s+(an?\s+)?image?s?[\s.,!?]*$', content, re.IGNORECASE):
                continue
            # This is likely the actual prompt (skip very short messages)
            if len(content) > 3:
                params['prompt'] = content
                break
    
    return params

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
        - Keep technical terms and model names in their original form (e.g., "Nano Banana", "GPT", "Freepik")
        
        MCP ACTION DETECTION (CRITICAL - APPLY FIRST):
        Before responding, ALWAYS analyze if the user wants to execute an MCP tool action.
        Look for these patterns:
        
        1. VIDEO GENERATION INTENT:
           - "Anima/Animate" + reference to image/video = generate_video tool
           - "Crea/Create video" = generate_video tool
           - Mentions video models: Sora 2, Sora 2 Pro, Veo 3.1, Runway Aleph, Runway 4.5, etc.
           - Example: "Anima esta imagen con sora 2" = EXECUTE generate_video with model sora-2
           - Example: "Animate this with runway" = EXECUTE generate_video with model runway-aleph
        
        2. IMAGE GENERATION INTENT:
           - "Genera/Generate imagen/image" = generate_image tool
           - "Crea/Create una imagen" = generate_image tool
           - Mentions image models: GPT, Nano Banana, Freepik
           - Example: "Genera una imagen con Freepik" = EXECUTE generate_image with model Freepik
        
        3. SPEECH/AUDIO INTENT:
           - "Di/Say", "Voz/Voice", "Audio", "Narración/Narration" = generate_speech tool
        
        IMPORTANT: When you detect an MCP action, DO NOT just describe what you could do.
        Instead, START the workflow for that tool (ask for missing parameters, confirm cost, etc.)
        
        PROJECT CREATION WORKFLOW (NEW):
        If the user wants to create a "project" (a complete video production, story, or multiple assets), YOU MUST FOLLOW THIS STRICT PROCESS:
        
        1. PHASE 1: DISCOVERY
           - Ask the user to describe the project concept/idea.
           - Ask: "Do you have any reference images or videos?" (If provided, use them as context).
           - Ask specific details: "How many characters?", "What is the setting (day/night)?", "What is the mood?".
        
        2. PHASE 2: PLANNING
           - Based on the answers, present a detailed PLAN to the user.
           - List the assets needed (e.g., "Scene 1: Image of hero", "Scene 2: Video of hero running").
           - Ask for approval to proceed with the plan.
        
        3. PHASE 3: EXECUTION (CRITICAL)
           - Once approved, start generating the assets ONE BY ONE.
           - DO NOT try to generate everything at once.
           - For each asset in the plan, YOU MUST USE THE EXISTING TOOLS ('generate_image', 'generate_video') EXACTLY AS DEFINED BELOW.
           - You must still ask for Model, Cost, and Confirmation for EACH individual asset as per the tool rules.
           - Example: "Okay, let's start with Scene 1. We need an image of the hero. Which model do you want to use: Nano Banana or GPT?"
        
        ⛔ ABSOLUTE PROHIBITION - FALSE COMPLETION MESSAGES:
        - NEVER say "Done!", "Ready!", "Your video is ready", "Tu video está listo", "Your image is ready", 
          "Tu imagen está lista", "generado exitosamente", or ANY completion/success message UNLESS
          you have ACTUALLY called a tool (generate_image, generate_video, generate_speech) in THIS 
          SPECIFIC interaction AND the tool returned a success result.
        - If a tool was NOT called in the current interaction, you MUST NOT claim something was generated.
        - If a tool returned an error, you MUST inform the user about the error, not claim success.
        - If you are unsure whether a tool succeeded, say so honestly.
        - When in doubt, ASK the user what they need rather than claiming something is done.
        
        CRITICAL RULES FOR 'generate_image' TOOL:
        1. The 'prompt' parameter MUST BE EXACTLY the LITERAL TEXT the user entered.
        2. FORBIDDEN to modify, improve, summarize, translate, or reinterpret the user's text for the prompt.
        3. If user writes "Change the suit", the prompt sent to the tool must be "Change the suit".
        4. If there are attached images, always pass them in 'reference_images'.
        5. BEFORE calling generate_image, you MUST ALWAYS:
           a) Ask: "Which model do you want to use: Nano Banana, GPT, or Freepik?" (in user's language)
           b) Wait for user's response with chosen model
           c) Inform the cost: "This will cost X tokens (10 tokens per image × quantity)" (in user's language)
           d) Wait for explicit confirmation before proceeding
        6. Available models are: 'Nano Banana', 'GPT', and 'Freepik' (all cost 10 tokens per image).
        7. DO NOT assume the model - ALWAYS ask the user which one to use.
        8. NEVER mention URLs in your responses - images/videos are sent automatically to the user.
        9. IF THERE'S AN ERROR: Inform the user of the error, but if user says "try again" or "retry" (or "intentar de nuevo", "reintentar"),
           you MUST execute the tool again without hesitation.
        
        CRITICAL RULES FOR 'generate_video' TOOL:
        ⛔ REMINDER: NEVER say "video ready/listo" unless generate_video was ACTUALLY called AND returned success.
        1. The 'prompt' parameter MUST BE EXACTLY the LITERAL TEXT the user entered.
        2. FORBIDDEN to modify, improve, summarize, translate, or reinterpret the user's text.
        3. BEFORE calling generate_video, you MUST ALWAYS:
           a) Ask: "Which video model do you want to use?" (in user's language) and list options with costs AND DURATIONS:
              - Runway Aleph (19 tokens/sec) - 5 or 10 seconds - video-to-video (editing)
              - Runway 4.5 (25 tokens/sec) - 5, 8 or 10 seconds - high quality
              - Veo 3.1 (48 tokens/sec) - 8 seconds - high quality
              - Veo 3.1 Flash (21 tokens/sec) - 8 seconds - fast and economical
              - Veo 3.1 Ultra (60 tokens/sec) - 8 seconds - maximum Veo quality
              - Sora 2 (15 tokens/sec) - ONLY 4, 8 or 12 seconds
              - Sora 2 Pro (30 tokens/sec) - ONLY 4, 8 or 12 seconds - maximum quality
              - Kling V3 Omni Pro (8 tokens/sec) - 3 to 15 seconds - text/image-to-video
              - Kling V3 Omni Std (6 tokens/sec) - 3 to 15 seconds - video-to-video
           b) Wait for the user to choose the model
           c) Ask: "How many seconds duration?" (in user's language) and MENTION valid options for the chosen model
           d) Wait for duration
           e) VALIDATE that duration is compatible with the model:
              - Sora 2 / Sora 2 Pro: ONLY 4, 8 or 12 seconds
              - Veo 3.1 / Veo 3.1 Flash / Veo 3.1 Ultra: ONLY 8 seconds
              - Runway Aleph: 5 or 10 seconds
              - Runway 4.5: 5, 8 or 10 seconds
              - Kling V3 Omni Pro / Std: 3 to 15 seconds (integer)
           f) If duration is NOT valid, inform user of correct options and ask to choose a valid one
           g) Calculate and show: "This will cost X tokens (Y tokens/sec × Z seconds). Confirm?" (in user's language)
           h) Wait for explicit confirmation before proceeding
        4. IMPORTANT: When calling the tool, use EXACT names:
           - 'veo-3.1' (NOT 'veo 3.1' or 'Veo 3.1')
           - 'veo-3.1-flash' (NOT 'veo 3.1 flash')
           - 'veo-3.1-ultra' (NOT 'veo 3.1 ultra')
           - 'runway-aleph', 'runway-4.5', 'sora-2', 'sora-2-pro'
           - 'kling-v3-omni-pro', 'kling-v3-omni-std'
        5. If there are attached images, use them as reference automatically.
        6. For Runway Aleph OR Kling V3 Omni Std, if there's an attached VIDEO, use it as reference (video-to-video).
        7. NEVER mention video URLs in your responses - they are sent automatically.
        8. IF THERE'S AN ERROR (404, timeout, missing config, etc.):
           - Inform user of the error clearly and simply (in user's language)
           - Explain possible causes (e.g., "The endpoint doesn't exist in backend", "Missing configuration")
           - IF user says "try again", "retry", "intentar de nuevo", "reintentar", or similar,
             you MUST execute the tool again WITHOUT QUESTIONING, using the same parameters.
        9. Reference images are NOT lost after errors - they persist in the session.
        10. ALWAYS try when the user asks, even if there were previous errors.
        
        CRITICAL RULES FOR 'generate_speech' TOOL:
        1. Use this tool when user asks to generate speech, audio, voiceover, or "say something".
        2. 'voice_id' default is "Rachel" (21m00Tcm4TlvDq8ikWAM).
        3. 'model_id' default is "eleven_multilingual_v2".
        4. AVAILABLE VOICES (If user asks for voices, list them by gender/style):
           - MALE: 
             Adam (Deep), Antoni (Balanced), Bill (Trustworthy), Brian (Deep), Callum (Hoarse), 
             Charlie (Australian Casual), Chris (Casual), Daniel (British Authoritative), 
             Eric (Deep), George (British Warm), Harry (Anxious), Josh (Deep), 
             Liam (Young), River (Neutral), Roger (Laid-back), Will (Friendly).
           - FEMALE: 
             Alice (British News), Domi (Strong), Elli (Young), Jessica (Expressive), 
             Laura (Upbeat), Lily (British Warm), Matilda (Warm), Rachel (Professional), Sarah (Soft).
        5. NEVER mention the output URL/Data URI in the conversation text. The audio player will appear automatically.
        """
        
        full_system_prompt = f"{REELMOTION_SYSTEM_PROMPT}\n\n{tool_instructions}"
        
        self.model = genai.GenerativeModel(
            self.model_name, 
            system_instruction=full_system_prompt,
            tools=[generate_image, generate_video, generate_speech]
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
        print(f"DEBUG [chatbot]: add_generated_file called for UUID='{self.conversation_uuid}', url='{url}', type='{file_type}'")
        if url:
            await self.session_manager.save_generated_file(
                self.conversation_uuid,
                url,
                file_type,
                metadata
            )
    
    async def get_generated_files(self) -> list:
        """Get pending generated files (URLs) from Redis."""
        print(f"DEBUG [chatbot]: get_generated_files called for UUID='{self.conversation_uuid}'")
        files = await self.session_manager.get_pending_files(self.conversation_uuid)
        print(f"DEBUG [chatbot]: Got {len(files)} files from session_manager")
        return [{"url": f["url"], "type": f["type"]} for f in files]
    
    async def save_pending_action(self, function_name: str, args: dict, cost_message: str = ""):
        """Save a pending action waiting for user confirmation."""
        action = {
            "function": function_name,
            "args": args,
            "cost_message": cost_message
        }
        await self.session_manager.save_pending_action(self.conversation_uuid, action)
    
    async def get_pending_action(self) -> dict:
        """Get pending action waiting for confirmation."""
        return await self.session_manager.get_pending_action(self.conversation_uuid)
    
    async def clear_pending_action(self):
        """Clear the pending action after execution."""
        await self.session_manager.clear_pending_action(self.conversation_uuid)
    
    async def execute_pending_action(self) -> tuple[str, str]:
        """
        Execute a pending action directly without going through Gemini.
        Returns (tool_result, response_text).
        """
        action = await self.get_pending_action()
        if not action:
            return None, None
        
        function_name = action.get("function")
        args = action.get("args", {}).copy()  # Copy to avoid modifying original
        
        print(f"DEBUG [chatbot]: Executing pending action '{function_name}' directly with args: {args}")
        
        # Get reference images if needed and not already in args
        ref_files = await self.get_reference_files()
        if ref_files and function_name in ["generate_image", "generate_video"]:
            ref_urls = [f["url"] for f in ref_files if f.get("type") == "image"]
            if ref_urls:
                if function_name == "generate_image" and "reference_images" not in args:
                    args["reference_images"] = ref_urls
                    print(f"DEBUG [chatbot]: Added {len(ref_urls)} reference images to pending action")
                elif function_name == "generate_video" and "reference_image" not in args:
                    args["reference_image"] = ref_urls[0]
                    print(f"DEBUG [chatbot]: Added reference image to pending video action")
        
        tool_result = None
        try:
            if function_name == "generate_image":
                tool_result = await generate_image(**args)
            elif function_name == "generate_video":
                tool_result = await generate_video(**args)
            elif function_name == "generate_speech":
                tool_result = await generate_speech(**args)
            else:
                return None, f"Error: Unknown pending function '{function_name}'"
            
            # Clear the pending action after successful execution
            await self.clear_pending_action()
            
            # Generate success message - tool WAS actually called here
            response_text = self._generate_contextual_success_message(tool_result, tool_was_called=True)
            if not response_text:
                # Fallback if message generation returns None (shouldn't happen for pending actions)
                response_text = tool_result if tool_result else "⚠️ No se pudo determinar el resultado de la operación."
            return tool_result, response_text
            
        except Exception as e:
            print(f"ERROR [chatbot]: Failed to execute pending action: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            await self.clear_pending_action()
            return None, f"Error executing {function_name}: {str(e)}"
    
    def _extract_function_from_history(self) -> tuple[str, dict]:
        """
        Try to extract function call info from recent chat history.
        Used for MALFORMED_FUNCTION_CALL recovery.
        Returns (function_name, args) or (None, None).
        """
        # This is a best-effort extraction based on conversation context
        # Will be implemented to parse the chat session history
        return None, None
        
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
        # Ensure the conversation context is set for tools
        set_conversation_uuid(self.conversation_uuid)

        try:
            if not self.chat_session:
                await self.start_chat()
            
            # === FAST PATH: Check for confirmation with pending action ===
            if is_confirmation(message):
                pending_action = await self.get_pending_action()
                if pending_action:
                    print(f"DEBUG [chatbot]: User confirmed! Executing pending action directly.")
                    # Save user message
                    await self.session_manager.add_message(
                        self.conversation_uuid,
                        "user",
                        message
                    )
                    
                    # Execute directly without going through Gemini
                    tool_result, response_text = await self.execute_pending_action()
                    
                    if response_text:
                        # Save response
                        await self.session_manager.add_message(
                            self.conversation_uuid,
                            "assistant",
                            response_text
                        )
                        return response_text
                    # If execution failed, fall through to normal Gemini flow
            
            # Guardar mensaje del usuario en Redis
            await self.session_manager.add_message(
                self.conversation_uuid,
                "user",
                message
            )
            
            # === CHECK FOR AMBIGUITY BEFORE PROCESSING ===
            ref_files = await self.get_reference_files()
            needs_clarif, question = needs_clarification(message, bool(ref_files))
            if needs_clarif:
                print(f"DEBUG [chatbot]: Ambiguous request detected, asking for clarification")
                await self.session_manager.add_message(
                    self.conversation_uuid,
                    "assistant",
                    question
                )
                return question
            
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
            
            # Send to Gemini with timeout (20 seconds max for initial response)
            GEMINI_TIMEOUT = 20  # seconds - if it takes longer, something is wrong
            try:
                print(f"DEBUG [chatbot]: Sending message to Gemini (timeout={GEMINI_TIMEOUT}s)...")
                import time
                start_time = time.time()
                response = await asyncio.wait_for(
                    self.chat_session.send_message_async(parts),
                    timeout=GEMINI_TIMEOUT
                )
                elapsed = time.time() - start_time
                print(f"DEBUG [chatbot]: Gemini responded in {elapsed:.1f}s")
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                print(f"WARNING: Gemini timed out after {elapsed:.1f}s without calling any tools")
                # Check if we have a pending action to execute directly
                pending_action = await self.get_pending_action()
                if pending_action:
                    print(f"DEBUG: Timeout recovery - executing pending action: {pending_action.get('function')}")
                    tool_result, response_text = await self.execute_pending_action()
                    if response_text:
                        await self.session_manager.add_message(
                            self.conversation_uuid,
                            "assistant",
                            response_text
                        )
                        return response_text
                # No pending action - ask user to clarify
                clarification = "Lo siento, no entendí bien qué quieres hacer. ¿Podrías ser más específico? Por ejemplo:\n" \
                               "- '¿Quieres crear un **video** o una **imagen**?'\n" \
                               "- Si tienes una imagen de referencia: '¿Quieres **animarla** o **generar una imagen similar**?'"
                await self.session_manager.add_message(
                    self.conversation_uuid,
                    "assistant",
                    clarification
                )
                return clarification
            
            # Handle function calls manually
            try:
                last_tool_result = None
                tool_was_actually_called = False  # Track if ANY tool was actually executed
                while True:
                    fc = None
                    candidate_parts = None
                    if response.candidates and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if candidate.content and candidate.content.parts:
                            candidate_parts = candidate.content.parts
                    parts_to_check = candidate_parts if candidate_parts is not None else response.parts
                    if parts_to_check:
                        for part in parts_to_check:
                            if hasattr(part, "function_call") and part.function_call:
                                fc = part.function_call
                                break

                    if not fc:
                        break

                    func_name = fc.name
                    func_args = dict(fc.args)
                    
                    print(f"DEBUG: Handling function call: {func_name}")
                    
                    tool_result = "Error: Unknown function"
                    try:
                        if func_name == "generate_image":
                            print(f"DEBUG [chatbot]: Calling generate_image...")
                            tool_result = await generate_image(**func_args)
                            print(f"DEBUG [chatbot]: generate_image returned: {tool_result[:200] if tool_result else 'None'}...")
                            # Files are saved inside generate_image tool - no need to extract URLs here
                                    
                        elif func_name == "generate_video":
                            print(f"DEBUG [chatbot]: Calling generate_video...")
                            tool_result = await generate_video(**func_args)
                            print(f"DEBUG [chatbot]: generate_video returned: {tool_result[:200] if tool_result else 'None'}...")
                            # Files are saved inside generate_video tool - no need to extract URLs here
                        
                        elif func_name == "generate_speech":
                            print(f"DEBUG [chatbot]: Calling generate_speech...")
                            tool_result = await generate_speech(**func_args)
                            print(f"DEBUG [chatbot]: generate_speech returned: {tool_result[:200] if tool_result else 'None'}...")
                            # Files are saved inside generate_speech tool - no need to extract URLs here
                                    
                        else:
                            print(f"ERROR [chatbot]: Unknown function: {func_name}")
                            tool_result = f"Error: Unknown function '{func_name}'"
                    except Exception as e:
                        print(f"ERROR [chatbot]: Exception executing {func_name}: {e}")
                        import traceback
                        print(f"Traceback: {traceback.format_exc()}")
                        tool_result = f"Error executing {func_name}: {str(e)}"
                    
                    last_tool_result = tool_result
                    tool_was_actually_called = True
                    
                    # Send result back with timeout
                    try:
                        response = await asyncio.wait_for(
                            self.chat_session.send_message_async(
                                genai.protos.Part(
                                    function_response=genai.protos.FunctionResponse(
                                        name=func_name,
                                        response={"result": tool_result}
                                    )
                                )
                            ),
                            timeout=30  # 30s for function result response
                        )
                    except asyncio.TimeoutError:
                        print(f"WARNING: Gemini timed out processing function result")
                        # If we have a tool result AND tool was actually called, return success
                        if tool_was_actually_called and tool_result:
                            response_text = self._generate_contextual_success_message(tool_result, tool_was_called=True)
                            if response_text:
                                await self.session_manager.add_message(
                                    self.conversation_uuid,
                                    "assistant",
                                    response_text
                                )
                                return response_text
                        # No tool was called or result is empty - inform user
                        error_msg = "⚠️ Hubo un problema procesando tu solicitud. Por favor, intenta de nuevo."
                        await self.session_manager.add_message(
                            self.conversation_uuid,
                            "assistant",
                            error_msg
                        )
                        return error_msg
                
                # Get text response safely - handle empty responses
                response_text = None
                try:
                    # Check if response has valid parts before accessing .text
                    if response.candidates and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if candidate.content and candidate.content.parts:
                            # Try to get text from parts
                            text_parts = []
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    text_parts.append(part.text)
                            if text_parts:
                                response_text = ''.join(text_parts)
                    
                    # Fallback to .text accessor if we didn't get text above
                    if not response_text:
                        response_text = response.text
                except ValueError:
                    # response.text accessor failed - response has no valid parts
                    pass
                
                # If still no response, handle based on whether a tool was ACTUALLY called
                if not response_text:
                    print(f"DEBUG: No text in Gemini response. tool_was_actually_called={tool_was_actually_called}")
                    
                    if tool_was_actually_called and last_tool_result:
                        # Tool WAS called - check if it succeeded or failed
                        tool_result_lower = last_tool_result.lower() if last_tool_result else ""
                        is_error = tool_result_lower.startswith("error")
                        
                        if is_error:
                            # Tool was called but FAILED - tell the user what went wrong
                            response_text = f"⚠️ Hubo un problema al generar tu contenido: {last_tool_result}\nPor favor, intenta de nuevo o ajusta los parámetros."
                        else:
                            # Tool was called and SUCCEEDED - generate success message
                            try:
                                followup_prompt = f"""The tool was executed successfully. Here is the result:
{last_tool_result}

Please generate a SHORT, friendly response to the user confirming the operation was successful. 
- If an image was generated, tell them their image is ready.
- If a video was generated, tell them their video is ready.
- If audio/speech was generated, tell them their audio is ready.
- NEVER mention URLs or technical details.
- Respond in the same language the user was using.
- Keep it brief and friendly (1-2 sentences max)."""
                                
                                followup_response = await self.chat_session.send_message_async(followup_prompt)
                                if followup_response.text:
                                    response_text = followup_response.text
                                else:
                                    response_text = self._generate_contextual_success_message(last_tool_result, tool_was_called=True)
                            except Exception as e:
                                print(f"DEBUG: Failed to get followup response: {e}")
                                response_text = self._generate_contextual_success_message(last_tool_result, tool_was_called=True)
                    else:
                        # NO tool was called - NEVER say "Done" or "Your video is ready"
                        # Ask Gemini to produce a real response
                        print("DEBUG: No tool was called and no text response - recovering")
                        try:
                            recovery_prompt = f"""IMPORTANT: No tool was executed. The previous response had no text.
User message: {message}

You MUST respond to the user directly, in the user's language. Do NOT call any tools.
Do NOT say "Done", "Ready", "Your video/image is ready" or any completion message.
If the user requested an image or video, ask for the missing parameters (model, duration, cost confirmation).
If you cannot determine what the user wants, ask them to clarify.
Keep it brief and helpful."""
                            recovery_response = await self.chat_session.send_message_async(recovery_prompt)
                            if recovery_response.text:
                                response_text = recovery_response.text
                            else:
                                response_text = "⚠️ No pude procesar tu solicitud. ¿Podrías intentar de nuevo con más detalles?"
                        except Exception as e:
                            print(f"DEBUG: Failed to get recovery response: {e}")
                            response_text = "⚠️ No pude procesar tu solicitud. ¿Podrías intentar de nuevo con más detalles?"

            except ValueError as e:
                # Handle Gemini safety or malformed content errors
                error_str = str(e)
                if "MALFORMED_FUNCTION_CALL" in error_str:
                    print(f"WARNING: Gemini MALFORMED_FUNCTION_CALL: {error_str}")
                    
                    # Try to execute pending action if we have one (user already confirmed)
                    pending_action = await self.get_pending_action()
                    if pending_action:
                        print(f"DEBUG: MALFORMED recovery - executing pending action: {pending_action.get('function')}")
                        tool_result, response_text = await self.execute_pending_action()
                        if response_text:
                            await self.session_manager.add_message(
                                self.conversation_uuid,
                                "assistant",
                                response_text
                            )
                            return response_text
                    
                    # No pending action - ask Gemini to retry without tool calls
                    try:
                        recovery_prompt = f"""The response had a malformed function call.
User message: {message}

Please respond to the user directly, in the user's language. Do NOT call any tools.
If the user requested an image or video, ask for the required model/cost/confirmation steps per the rules.
Keep it brief and helpful."""
                        recovery_response = await self.chat_session.send_message_async(recovery_prompt)
                        if recovery_response.text:
                            response_text = recovery_response.text
                        else:
                            response_text = "⚠️ No pude procesar tu solicitud. ¿Podrías intentar de nuevo con más detalles?"
                    except Exception as e:
                        print(f"DEBUG: Failed to get recovery response after MALFORMED_FUNCTION_CALL: {e}")
                        response_text = "⚠️ No pude procesar tu solicitud. ¿Podrías intentar de nuevo con más detalles?"
                elif "finish_reason" in error_str or "response.text" in error_str:
                    print(f"WARNING: Gemini response error: {error_str}")
                    if tool_was_actually_called and last_tool_result:
                        # Tool was called - show its result
                        result_msg = self._generate_contextual_success_message(last_tool_result, tool_was_called=True)
                        response_text = result_msg if result_msg else str(last_tool_result)
                    else:
                        # No tool was called - ask user to retry
                        response_text = "⚠️ Hubo un problema procesando tu solicitud. Por favor, intenta de nuevo."
                else:
                    raise e
            
            # === Detect cost confirmation and save pending action ===
            if response_text and COST_CONFIRMATION_PATTERN.search(response_text):
                # Gemini is asking for confirmation - extract params and save pending action
                session = await self.session_manager.get_session(self.conversation_uuid)
                history = session.get("messages", []) if session else []
                
                # Get reference images for the action
                ref_files = await self.get_reference_files()
                ref_urls = [f["url"] for f in ref_files if f.get("type") == "image"] if ref_files else []
                
                # Detect if this is video or image
                response_lower = response_text.lower()
                if any(word in response_lower for word in ['video', 'vídeo', 'animar', 'animate', 'sora', 'veo', 'runway', 'kling']):
                    # Include the current response_text in history for better param detection
                    history_with_current = history + [{'role': 'assistant', 'content': response_text}]
                    params = detect_video_params_from_history(history_with_current)
                    if params.get('model') and params.get('duration'):
                        action_args = {
                            "prompt": params.get('prompt', 'animate the image'),
                            "model": params['model'],
                            "duration": params['duration']
                        }
                        if ref_urls:
                            action_args["reference_image"] = ref_urls[0]
                        print(f"DEBUG: Saving pending VIDEO action: {action_args}")
                        await self.save_pending_action(
                            "generate_video",
                            action_args,
                            response_text
                        )
                elif any(word in response_lower for word in ['imagen', 'image', 'gpt', 'nano banana']):
                    # Detect image generation
                    params = detect_image_params_from_history(history)
                    if params.get('model'):
                        action_args = {
                            "prompt": params.get('prompt', 'generate image'),
                            "model": params['model']
                        }
                        if ref_urls:
                            action_args["reference_images"] = ref_urls
                        print(f"DEBUG: Saving pending IMAGE action: {action_args}")
                        await self.save_pending_action(
                            "generate_image",
                            action_args,
                            response_text
                        )
            
            # Guardar respuesta del asistente en Redis
            await self.session_manager.add_message(
                self.conversation_uuid,
                "assistant",
                response_text
            )
            
            return response_text
            
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
    
    def _generate_contextual_success_message(self, tool_result: str, tool_was_called: bool = False) -> str:
        """Generate a contextual success message based on tool result.
        
        CRITICAL: Only returns success messages if tool_was_called is True AND
        the tool_result indicates actual success (contains 'exitosamente' / 'successfully').
        Otherwise returns an informative error/status message.
        """
        if not tool_result or not tool_was_called:
            # No tool was executed - NEVER return a success message
            return None
        
        tool_result_lower = tool_result.lower()
        
        # Check if the tool result indicates an actual error
        if tool_result_lower.startswith("error"):
            return f"⚠️ Hubo un problema al procesar tu solicitud: {tool_result}"
        
        # Only return success if the tool result confirms success
        is_success = any(word in tool_result_lower for word in [
            'exitosamente', 'successfully', 'generado', 'generated', 'generada'
        ])
        
        if not is_success:
            # Tool returned but result is unclear - return the raw result
            return tool_result
        
        if "image" in tool_result_lower or "imagen" in tool_result_lower:
            return "Your image is ready!"
        elif "video" in tool_result_lower or "vídeo" in tool_result_lower:
            return "Your video is ready!"
        elif "audio" in tool_result_lower or "speech" in tool_result_lower or "voz" in tool_result_lower:
            return "Your audio is ready!"
        else:
            return tool_result


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
