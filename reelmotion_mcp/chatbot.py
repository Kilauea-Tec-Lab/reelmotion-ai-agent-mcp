import os
import base64
import time
import re
import asyncio
import logging
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv
from cachetools import TTLCache

from prompts import REELMOTION_SYSTEM_PROMPT
from tools import generate_image, generate_video, generate_speech
from session_manager import get_session_manager
from request_context import (
    set_conversation_uuid,
    get_token_balance,
    set_insufficient_block,
)
from pricing import (
    estimate_generation_cost,
    affordable_options,
    build_insufficient_balance_message,
    is_spanish,
    detect_language,
    is_insufficient_balance_message,
    min_video_cost,
)
from generation_errors import (
    GENERATION_ERROR_PREFIX,
    SUPPORT_EMAIL,
    parse_generation_error,
    fallback_error_message,
    is_generation_processing,
    is_generation_success,
    generation_processing_type,
    processing_message,
    success_message,
    success_gen_type,
)
from logging_config import setup_logging
from moderation import (
    is_disallowed_content,
    is_disallowed_content_full,
    get_refusal_message,
)
from workflow_state import (
    WORKFLOW_UNKNOWN,
    apply_user_message,
    build_action_args,
    capture_refined_prompt,
    detect_json_prompt,
    detect_workflow_intent,
    is_confirmation,
    is_reaction,
    is_ready_for_confirmation,
    is_refine_decline,
    merge_extracted,
    new_state,
    state_context_note,
)
from param_extractor import extract_generation_params

# Load environment variables
load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)

# Pattern to detect when Gemini is asking for confirmation (cost question)
COST_CONFIRMATION_PATTERN = re.compile(
    r'(?:costará|cost|costar[aá]n|tokens?|créditos?|credits?|¿confirmas?|confirm|proceder|proceed)',
    re.IGNORECASE
)

# Replies that reject or postpone a quoted cost. When the last assistant
# message asked for cost confirmation, any reply WITHOUT one of these words is
# treated as acceptance if Gemini responds with a tool call — re-asking on
# every phrasing the CONFIRMATION_PATTERNS regex doesn't know caused endless
# confirmation loops.
COST_DECLINE_PATTERN = re.compile(
    r"\b(?:no|nope|nah|not\s+yet|cancel\w*|cancela\w*|stop|wait|espera|"
    r"a[uú]n\s+no|todav[ií]a\s+no|mejor\s+no|cambi\w+|change|otro|otra|"
    r"different|instead)\b",
    re.IGNORECASE,
)

# Tools that spend the user's tokens — they may ONLY run after the user
# explicitly confirmed a cost message (enforced in code, not just the prompt).
GENERATION_TOOL_NAMES = ("generate_image", "generate_video", "generate_speech")

# How many unconfirmed tool calls we bounce back to Gemini before answering
# with a code-generated cost confirmation ourselves.
MAX_CONFIRMATION_INTERCEPTIONS = 2

# Clarification prompts for ambiguous requests, keyed by intent then language.
# These are code-generated and returned to the user WITHOUT passing through
# Gemini, so they must be localized to the conversation language explicitly —
# the model never sees them and cannot adapt their language. Only 'en'/'es' are
# needed because the resolved conversation language is always one of those
# (other languages fall back to English, matching the system prompt default).
CLARIFICATION_PROMPTS = {
    "generic": {
        "en": "What exactly would you like to create? An image or a video?",
        "es": "¿Qué quieres crear exactamente? ¿Una imagen o un video?",
    },
    "ref_file": {
        "en": (
            "What would you like to do with this image? Generate a video from it "
            "or create a new, similar image?"
        ),
        "es": (
            "¿Qué quieres hacer con esta imagen? ¿Generar un video a partir de ella "
            "o crear una nueva imagen similar?"
        ),
    },
}

# Friendly acknowledgment for a post-generation reaction ("perfect!", "genial"),
# used ONLY as a fallback when Gemini is unavailable (timeout/error). Localized
# so a non-English conversation never falls back to an English-only reply.
REACTION_ACK = {
    "en": "Glad you liked it! Let me know if you need anything else. 😊",
    "es": "¡Me alegra que te haya gustado! Dime si necesitas algo más. 😊",
}

def needs_clarification(message: str, has_ref_files: bool) -> tuple[bool, str]:
    """
    Check if a message is ambiguous and needs clarification.

    Returns (needs_clarification, clarification_key) where clarification_key
    identifies WHICH question to ask (a key into CLARIFICATION_PROMPTS). The
    caller localizes it to the conversation language, because these prompts
    bypass Gemini and would otherwise leak a hardcoded language to the user.
    """
    msg_lower = message.lower().strip()
    
    # === CLEAR INTENT: VIDEO MODEL MENTIONED ===
    video_model_keywords = ['veo', 'kling', 'runway', 'haiper', 'minimax', 'aleph', 'seedance']
    if any(model in msg_lower for model in video_model_keywords):
        return False, ""  # Clear: wants to generate VIDEO
    
    # === CLEAR INTENT: IMAGE MODEL MENTIONED ===
    image_model_keywords = ['gpt', 'nano', 'banana', 'seedream', 'midjourney']
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
            return True, "generic"
    
    # === AMBIGUOUS: Has reference file but VERY unclear what to do ===
    if has_ref_files and len(msg_lower) < 5:
        # Only very vague single words without context
        vague_words = ['eso', 'esto', 'that', 'this', 'aquí', 'here']
        if msg_lower in vague_words:
            return True, "ref_file"
    
    return False, ""

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

        # Resolved language of the CURRENT conversation ('es' | 'en'), set once
        # per send_message from the user's running history. None = not yet
        # resolved → helpers fall back to a per-text heuristic (see _lang_for).
        self._conv_lang: Optional[str] = None
        # Recent user text used to render the balance block in ANY language when
        # the conversation is neither Spanish nor English (see
        # _localize_balance_block). Empty outside an active send_message turn.
        self._lang_sample: str = ""
        
        # Critical tool usage instructions
        tool_instructions = """
        LANGUAGE ADAPTATION (CRITICAL - HIGHEST PRIORITY):
        - DEFAULT LANGUAGE: English. If you have NO prior conversation history or cannot determine the user's language, ALWAYS use English.
        - You MUST respond in the SAME language the user is writing in.
        - Detect the language from EACH user message. Track the LAST CLEARLY IDENTIFIABLE language.
        - If the user writes in English, respond in English. If the user writes in Spanish, respond in Spanish. Same for any other language.
        - AMBIGUOUS MESSAGES: Short replies like "no", "ok", "yes", "gpt", "veo 3.1", "5s", model names, or single words that exist in multiple languages are NOT a language switch. KEEP the last clearly detected language (default: English).
        - LANGUAGE SWITCH: Only change language if the user writes a CLEAR sentence in a different language or explicitly requests it.
        - This applies to ALL messages: questions, confirmations, cost info, errors, EVERYTHING.
        - Keep technical terms and model names in their original form (e.g., "Seedream", "GPT", "Nano Banana 2", "Midjourney").
        - IMPORTANT: All instructions below are written in English for clarity, but you MUST always respond to the user in THEIR language (default: English).

        🧭 WORKFLOW STATE CONTEXT (AUTHORITATIVE):
        - User messages may include a "[SYSTEM CONTEXT — workflow state ...]" note injected by the system (NOT written by the user).
        - That note is the AUTHORITATIVE record of the current workflow: its type, the current step, every parameter already collected (prompt, model, duration, resolution, voice), and which fields are still missing.
        - TRUST THE NOTE over your own reading of the conversation history. Ask ONLY for the fields it lists as missing. NEVER re-ask for a value the note already shows.
        - If the note says "ready for cost confirmation", present the cost summary and ask for confirmation — do not repeat earlier steps.
        - If the note shows prompt=<user's JSON prompt ...>, the prompt is a JSON object that is sent to the generator VERBATIM. Never rewrite it as prose, never quote it in full — refer to it as "your JSON prompt".

        ⛔ MANDATORY WORKFLOW - NEVER SKIP STEPS:
        You MUST NEVER call generate_image or generate_video unless ALL workflow steps have been completed.
        Each step MUST happen in a SEPARATE message exchange (user sends message → you respond → user sends next message → you respond).
        You CANNOT complete multiple steps in a single response.
        If ANY step is missing, you MUST ask for it before proceeding.

        🔒 HARD RULE — COST CONFIRMATION BEFORE EVERY GENERATION:
        NEVER call generate_image / generate_video / generate_speech unless the user's
        MOST RECENT message explicitly confirms the cost you quoted in your previous message.
        Even when you already have ALL the parameters, your next reply must be the cost
        summary + confirmation question — NOT a tool call. The system BLOCKS unconfirmed
        tool calls, so calling early only wastes a turn.
        
        MCP ACTION DETECTION (CRITICAL - APPLY FIRST):
        Before responding, ALWAYS analyze if the user wants to execute an MCP tool action.
        Look for these patterns:
        
        1. VIDEO GENERATION/EDITING INTENT:
           - "Animate" + reference to image/video = VIDEO workflow
           - "Create video" = VIDEO workflow
           - "Edit video" + reference video = VIDEO-TO-VIDEO workflow
           - Mentions video models: Seedance 2.0, Veo 3.1, Runway Aleph, Runway 4.5, etc.
           - IMPORTANT: Start the VIDEO WORKFLOW, do NOT call the tool directly.
           - For video EDITING (video-to-video), the user MUST provide a reference video.
             Supported models for video editing: Runway Aleph, Kling O3 (video-edit).
        
        2. IMAGE GENERATION/EDITING INTENT:
           - "Generate image" = IMAGE workflow
           - "Create an image" = IMAGE workflow
           - "Edit image" + reference image = IMAGE-TO-IMAGE workflow
           - Mentions image models: Seedream, Midjourney, GPT, Nano Banana 2
           - IMPORTANT: Start the IMAGE WORKFLOW, do NOT call the tool directly.
           - For image EDITING (image-to-image), the user MUST provide a reference image.
             The tool uses type 2 (single ref) or type 3 (multiple refs).
        
        3. SPEECH/AUDIO INTENT:
           - "Say", "Voice", "Audio", "Narration", "Create a voice" = SPEECH workflow
           - IMPORTANT: Start the SPEECH WORKFLOW, do NOT call the tool directly.
        
        NOTE: These patterns apply in ANY language. If the user says the equivalent in Spanish, French, etc., detect the intent the same way.
        
        IMPORTANT: When you detect image/video/speech intent, START the step-by-step workflow.
        DO NOT call the tool directly. Follow ALL steps in order.
        
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
           - You must still complete ALL workflow steps for EACH individual asset.
           - Example: "Okay, let's start with Scene 1. We need an image of the hero. Which model do you want to use: Seedream, GPT, Nano Banana 2, or Midjourney?"
        
        ⛔ ABSOLUTE PROHIBITION - FALSE COMPLETION MESSAGES:
        - NEVER say "Done!", "Ready!", "Your video is ready", "Your image is ready",
          or ANY completion/success message UNLESS
          you have ACTUALLY called a tool (generate_image, generate_video, generate_speech) in THIS 
          SPECIFIC interaction AND the tool returned a success result.
        - If a tool was NOT called in the current interaction, you MUST NOT claim something was generated.
        - If a tool returned an error, you MUST inform the user about the error, not claim success.
        - If you are unsure whether a tool succeeded, say so honestly.
        - When in doubt, ASK the user what they need rather than claiming something is done.
        
        ═══════════════════════════════════════════════════
        CRITICAL RULES FOR 'generate_image' TOOL - MANDATORY WORKFLOW
        ═══════════════════════════════════════════════════
        
        You MUST follow these steps IN ORDER. Each step requires a SEPARATE user response.
        NEVER skip a step. NEVER combine steps. NEVER call the tool until Step 4 is confirmed.
        
        STEP 1 - IDENTIFY INTENT AND ASK FOR THE PROMPT:
        - When you detect the user wants to create an IMAGE, ask: "What do you want the image to show? Describe it in detail." (in user's language)
        - If the user already provided a clear descriptive prompt in their message, take that as the prompt and IMMEDIATELY move to Step 2 in the SAME response.
        - Wait for the user's response. SAVE this descriptive text mentally as THE_PROMPT.
        
        STEP 2 - OFFER TO HELP WITH THE PROMPT:
        - Tell the user you can help improve their prompt for better results.
        - Ask: "Would you like me to help you refine or improve your prompt for better results?" (in user's language)
        - If user says YES/OK/SI/DALE: 
          → You MUST write an improved, more detailed version of the prompt.
          → Show it to the user between quotes.
          → Ask: "Do you like this version?" (in user's language)
          → If user approves: UPDATE THE_PROMPT to the refined version. Move to Step 3.
          → If user wants changes: iterate until satisfied, then move to Step 3.
        - If user says NO or wants to skip: Keep THE_PROMPT as-is and move to Step 3.
        - ⚠️ CRITICAL: When user says "ok/sí/dale" to this step, it means they WANT HELP WITH THE PROMPT.
          You MUST write the refined prompt. Do NOT interpret this as final confirmation to generate.
        - 🧾 JSON PROMPTS: If THE_PROMPT is a JSON object (the state note shows prompt=<user's JSON prompt>),
          SKIP the text-refinement offer. Instead offer JSON-aware suggestions: point out useful missing
          keys (camera, lighting, style, audio) and, only if the user wants changes, show the improved
          version as a COMPLETE ```json block after the ✨ marker. Never alter their values silently.

        STEP 3 - CHOOSE THE MODEL (PICK BY INTENT, WITH SUGGESTION):
        - Based on THE_PROMPT, choose the BEST model for the user's intent and explain why briefly.
          Do NOT ask unnecessary questions — pick by intent:
          → realism / photographic fidelity / cinematic scenes / has reference images → Seedream
          → artistic style / illustration / creative concept ("Midjourney look") → Midjourney
          → editing an existing image / composing several references → Nano Banana 2
          → readable text inside the image / strict instruction following → GPT
        - ⚠️ ALWAYS present the models as a FORMATTED LIST (one model per line) so the user can override your pick.
        - Available models (exact names, case-sensitive):
          → Seedream (4 tokens): realism, photographic fidelity, cinematic scenes, reference images. ⭐ Best quality/price — recommended default.
          → GPT (6 tokens): readable text inside the image, strict instruction following.
          → Nano Banana 2 (7 tokens): quick edits of an existing image, multi-reference composition.
          → Midjourney (9 tokens): artistic style, illustration, creative concepts.
        - ⛔ There is NO "Freepik" model — never offer or select it.
        - Ask: "I suggest [model] because [reason]. Which model would you like to use?" (in user's language)
        - Wait for the user to choose (or accept your suggestion).
        - Token costs per image: Seedream = 4, GPT = 6, Nano Banana 2 = 7, Midjourney = 9 tokens.

        STEP 4 - CONFIRM COST AND EXECUTE:
        - Summarize what will be generated:
          → "I'm going to generate: [brief description of THE_PROMPT]"
          → "Model: [chosen model]"
          → "Cost: [X] tokens" (Seedream=4, GPT=6, Nano Banana 2=7, Midjourney=9)
          → "Do you confirm?" (in user's language)
        - ⛔ DO NOT call the tool until the user explicitly confirms in this step.
        - Once confirmed, CALL generate_image immediately using THE_PROMPT (the descriptive text, NOT the confirmation message).
        
        ADDITIONAL IMAGE RULES:
        1. ⚠️ PROMPT PARAMETER RULE (EXTREMELY IMPORTANT):
           - The 'prompt' parameter MUST be the DESCRIPTIVE TEXT, never a user's conversational reply.
           - 🚨 REFINED PROMPT RULE: If you refined the prompt in Step 2 and user accepted, use THE REFINED TEXT as the prompt.
           - NEVER use "ok", "si", "dale", "gpt", "confirmo" as the prompt parameter.
        2. FORBIDDEN to modify the user's agreed-upon prompt without their consent.
        3. IMAGE-TO-IMAGE EDITING: If user wants to EDIT an image, they MUST attach the reference image.
           → The prompt should describe the EDITING instructions (e.g., "change background to sunset", "make it look like a painting").
           → Pass reference images in 'reference_images' and set image_type to 2 (single ref) or 3 (multiple refs).
           → Editing/composing references works best with Nano Banana 2 (or Seedream); for Midjourney img2img the reference MUST be a public URL.
        4. If there are attached images, always pass them in 'reference_images'.
        5. Available models are: 'Seedream' (4 tokens), 'GPT' (6 tokens), 'Nano Banana 2' (7 tokens), 'Midjourney' (9 tokens). There is NO 'Freepik' model.
        6. ONE IMAGE PER CALL for Seedream and Midjourney — 'type'/'quantity' are ignored for them (always 1 image). Only GPT and Nano Banana 2 honor 'quantity' and multi-image 'type'. If the user wants several images with Seedream/Midjourney, generate them with separate calls (each is billed again).
        7. ASPECT RATIO: pass 'aspect_ratio' to match the destination — '16:9' (default, horizontal scenes), '9:16' (vertical/mobile/portraits), '1:1' (square). 'quality' ('2K'/'3K') only affects Seedream and does NOT change the cost.
        8. ASYNC DELIVERY: Seedream and Midjourney may take longer than the sync window. If the tool reports the image is "still processing", tell the user it's being generated and they'll be notified when ready — do NOT retry (the tokens were already charged). On a "failed" result the backend auto-refunds; only retry if the user asks.
        9. NEVER invent reference image URLs — use only the ones the user provides.
        10. NEVER mention URLs in your responses - images are sent automatically to the user.
        11. IF THERE'S AN ERROR: Inform the user. If user says "try again"/"retry", execute the tool again without hesitation.
        
        ═══════════════════════════════════════════════════
        CRITICAL RULES FOR 'generate_video' TOOL - MANDATORY WORKFLOW
        ═══════════════════════════════════════════════════
        
        IMPORTANT: There are TWO different workflows for video:
        A) VIDEO GENERATION (text-to-video or image-to-video) = creating NEW videos
        B) VIDEO EDITING (video-to-video) = editing EXISTING videos
        
        Detect which one the user wants:
        - "Edit video", "Change the video", "Modify the video" + reference video = WORKFLOW B (editing)
        - "Create video", "Generate video", "Animate" = WORKFLOW A (generation)
        
        ═══════════════════════════════════════════════════
        WORKFLOW A: VIDEO GENERATION (text-to-video / image-to-video)
        ═══════════════════════════════════════════════════
        
        You MUST follow these steps IN ORDER. Each step requires a SEPARATE user response.
        NEVER skip a step. NEVER combine steps. NEVER call the tool until Step 5 is confirmed.
        
        STEP 1 - IDENTIFY INTENT AND ASK FOR THE PROMPT:
        - When you detect the user wants to create a VIDEO, ask: "What do you want the video to show? Describe the action, scene, or animation." (in user's language)
        - If the user already provided a clear DESCRIPTIVE prompt, take it and IMMEDIATELY move to Step 2 in the SAME response.
        - ⚠️ COMMAND vs PROMPT: "Create a video with veo 3.1" is a COMMAND (not a prompt). "A frog jumping in the jungle" IS a prompt.
        - If message has model names or durations, it's a COMMAND → still ask for descriptive prompt.
        - Wait for response. SAVE as THE_PROMPT.
        
        STEP 2 - OFFER TO HELP WITH THE PROMPT:
        - Tell the user you can help improve their prompt for a more cinematic result.
        - Ask: "Would you like me to help you refine or improve your prompt for better results?" (in user's language)
        - If user says YES/OK/SI/DALE:
          → Write an improved, more cinematic version of the prompt.
          → Show it between quotes.
          → Ask: "Do you like this version?" (in user's language)
          → If approved: UPDATE THE_PROMPT. Move to Step 3.
          → If wants changes: iterate until satisfied, then move to Step 3.
        - If user says NO or skips: Keep THE_PROMPT as-is, move to Step 3.
        - ⚠️ CRITICAL: "ok/sí/dale" here means HELP ME WITH THE PROMPT. Write the refined version. Do NOT treat it as a generation confirmation.
        
        STEP 3 - ASK FOR THE MODEL (WITH SUGGESTION):
        - Based on THE_PROMPT, suggest a model and explain why briefly.
        - ⚠️ ALWAYS present the models as a FORMATTED LIST (one model per line with its cost and durations), never as inline text.
        - Show available models with costs AND valid durations:
          → Kling V3 (resolution-based: 720p=9, 1080p=12, 4K=42 tokens/sec) - 3 to 15 sec - max quality, 4K, native audio, motion-control
          → Kling V3 Turbo (resolution-based: 720p=12, 1080p=14 tokens/sec) - 3 to 15 sec - fast & cheap drafts (max 1080p, no audio)
          → Kling O3 (resolution-based: 720p=9, 1080p=12, 4K=42 tokens/sec) - 3 to 15 sec - character/style consistency or edit an existing video
          → Seedance 2.0 Fast (resolution-based: 480p=12, 720p=26 tokens/sec) - 4 to 15 sec - fast & economical (max 720p)
          → Seedance 2.0 (resolution-based: 480p=15, 720p=32, 1080p=72 tokens/sec) - 4 to 15 sec - supports 1080p
          → Runway 4.5 (13 tokens/sec) - 5, 8 or 10 sec - high quality
          → Runway Aleph (17 tokens/sec) - 5 or 10 sec - versatile
          → Veo 3.1 Flash (17 tokens/sec) - 8 sec only - fast and good quality
          → Veo 3.1 (44 tokens/sec) - 8 sec only - high quality
          → Veo 3.1 Ultra (65 tokens/sec) - 8 sec only - maximum Veo quality
        - Kling quick-pick heuristic: fast/cheap → Kling V3 Turbo; max quality / 4K / audio → Kling V3;
          keep a character or style from reference images, or edit an existing video → Kling O3.
        - Say: "I suggest [model] because [reason]. Which model would you like to use?" (in user's language)
        - Wait for user to choose model.

        STEP 3.5 - ASK FOR RESOLUTION (ONLY for Seedance and Kling V3/Turbo/O3):
        - ⚠️ This step applies ONLY when the chosen model is Seedance 2.0 / Seedance 2.0 Fast OR Kling V3 / Kling V3 Turbo / Kling O3. For ALL OTHER models, SKIP this step entirely.
        - Their pricing depends on the resolution, so you MUST ask for it before quoting the cost.
          → Seedance 2.0: offer 480p, 720p, or 1080p.
          → Seedance 2.0 Fast: offer ONLY 480p or 720p. If the user asks for 1080p, tell them the Fast tier does not support it and it will use 720p (or suggest switching to Seedance 2.0).
          → Kling V3 / Kling O3 (text-to-video or image-to-video): offer 720p, 1080p, or 4K.
          → Kling V3 Turbo: offer ONLY 720p or 1080p (no 4K).
          → Kling O3 reference/video-edit and Kling V3 motion-control: offer ONLY 720p or 1080p (no 4K).
        - Ask: "Which resolution? Options: [valid resolutions for the chosen model]" (in user's language)
        - Wait for the user to choose. SAVE as THE_RESOLUTION.

        STEP 4 - ASK FOR DURATION:
        - Based on the chosen model, tell the user the valid durations:
          → Seedance 2.0 / Seedance 2.0 Fast: any whole number from 4 to 15 seconds (default 5)
          → Veo 3.1 / Veo 3.1 Flash / Veo 3.1 Ultra: ONLY 8 seconds (auto-set, just inform)
          → Runway Aleph: 5 or 10 seconds
          → Runway 4.5: 5, 8 or 10 seconds
          → Kling V3 / Kling V3 Turbo / Kling O3: 3 to 15 seconds (Kling O3 reference mode: 3 to 10 seconds)
        - If the model only allows ONE duration (e.g., Veo 3.1 = 8s), inform the user and auto-set it. Move to Step 5 in the SAME response.
        - Otherwise ask: "How many seconds? Options: [valid durations]" (in user's language)
        - Wait for the user to choose. VALIDATE the duration is valid for the model.

        STEP 5 - CONFIRM COST AND EXECUTE:
        - Calculate cost: tokens_per_second × duration.
        - 💎 SEEDANCE 2.0 PRICING (resolution-based — pick the per-second rate from the chosen RESOLUTION):
          → Normal rate:
            • Seedance 2.0: 480p = 15, 720p = 32, 1080p = 72 tokens/sec
            • Seedance 2.0 Fast: 480p = 12, 720p = 26 tokens/sec
          → Discounted rate — applies ONLY when the user attached a REFERENCE VIDEO (video-to-video / reference mode):
            • Seedance 2.0: 480p = 9, 720p = 20, 1080p = 43 tokens/sec
            • Seedance 2.0 Fast: 480p = 7, 720p = 16 tokens/sec
          → Use the DISCOUNTED rate ONLY if a reference video is attached; otherwise use the NORMAL rate.
        - 💎 KLING PRICING (resolution + route + audio based — tokens/sec):
          → Kling V3 / Kling O3, text-to-video or image-to-video: 720p = 9, 1080p = 12, 4K = 42
            (with audio add the surcharge: 720p = 12, 1080p = 14; audio only on this route)
          → Kling V3 Turbo (text/image-to-video only): 720p = 12, 1080p = 14 (no 4K, no audio)
          → Kling O3 reference mode / video-edit: 720p = 13, 1080p = 17 (no 4K, no audio)
          → Kling V3 motion-control (guide video): 720p = 13, 1080p = 17 (no 4K, no audio)
          → Audio defaults to OFF (cheaper); only quote the +audio rate if the user explicitly asked for audio.
        - Summarize what will be generated:
          → "I'm going to generate a video:"
          → "Prompt: [brief description of THE_PROMPT]"
          → "Model: [model]"
          → "Resolution: [THE_RESOLUTION]"   ← include this line ONLY for Seedance models
          → "Duration: [X] seconds"
          → "Cost: [Y] tokens ([Z] tokens/sec × [X] sec)"
          → "Do you confirm?" (in user's language)
        - ⛔ DO NOT call the tool until the user explicitly confirms this step.
        - Once confirmed, CALL generate_video immediately using THE_PROMPT. For Seedance models, also pass resolution=THE_RESOLUTION.
        
        ═══════════════════════════════════════════════════
        WORKFLOW B: VIDEO EDITING (video-to-video)
        ═══════════════════════════════════════════════════
        
        This workflow is for EDITING an existing video. The user must provide a reference video.
        You MUST follow these steps IN ORDER. NEVER skip steps. NEVER call the tool until Step 4.
        
        STEP 1 - ASK FOR THE VIDEO, EDITING INSTRUCTIONS, AND OFFER PROMPT HELP:
        - When you detect the user wants to EDIT a video, ask them to:
          a) Upload/attach the reference video (if not already attached)
          b) Describe what they want to change (e.g., "change skin color to purple", "add rain", "change style to anime")
        - If the user already provided both the video AND a description, take them directly.
        - SAVE the editing description as THE_EDIT_PROMPT.
        - ⛔ If no reference video is attached, ask the user to attach it before proceeding.
        - Once you have THE_EDIT_PROMPT, ask: "Would you like me to help you refine or improve your prompt for better results?" (in user's language)
        - If user says YES/OK/SI/DALE:
          → Write an improved, more detailed version of the editing prompt.
          → Show it between quotes.
          → Ask: "Do you like this version?" (in user's language)
          → If approved: UPDATE THE_EDIT_PROMPT. Move to Step 2.
          → If wants changes: iterate until satisfied, then move to Step 2.
        - If user says NO or skips: Keep THE_EDIT_PROMPT as-is, move to Step 2.
        
        STEP 2 - SHOW VIDEO EDITING MODELS ONLY:
        - ⚠️ ALWAYS present the models as a FORMATTED LIST (one model per line with its cost and durations), never as inline text.
        - Show ONLY the models that support video-to-video editing:
          → **Kling O3** (resolution-based: 720p = 13, 1080p = 17 tokens/sec) - 3 to 15 sec - video-edit ⭐ Recommended
          → **Runway Aleph** (17 tokens/sec) - 5 or 10 sec - High quality editing
        - ⛔ DO NOT show any other models (Veo, Runway 4.5, Seedance, Kling V3/Turbo, etc.) - they do NOT support video-to-video editing here.
        - Suggest Kling O3 as the recommended option for editing an existing video.
        - For Kling O3 you MUST also ask for the resolution (720p or 1080p) before quoting the cost.
        - Ask: "Which model would you like to use?" (in user's language)
        - Wait for user to choose. SAVE as THE_MODEL.

        STEP 3 - ASK FOR DURATION:
        - Based on THE_MODEL:
          → Kling O3: 3 to 15 seconds
          → Runway Aleph: 5 or 10 seconds
        - Ask: "How many seconds? Options: [valid durations]" (in user's language)
        - Wait for user to choose. SAVE as THE_DURATION. VALIDATE it's valid for the model.
        
        STEP 4 - CONFIRM AND EXECUTE:
        - Calculate cost: tokens_per_second × duration
        - Summarize the edit:
          → "I'm going to edit your video:"
          → "Edit: [THE_EDIT_PROMPT]"
          → "Model: [THE_MODEL]"
          → "Duration: [THE_DURATION] seconds"
          → "Cost: [Y] tokens ([Z] tokens/sec × [X] sec)"
          → "Do you confirm?" (in user's language)
        - ⛔ DO NOT call the tool until the user explicitly confirms.
        - Once confirmed, CALL generate_video immediately using:
          → prompt = THE_EDIT_PROMPT
          → model = the chosen model name (exact: 'kling-o3' or 'runway-aleph'). It is sent to the backend as `provider`.
          → duration = THE_DURATION
          → reference_video = the attached video URL (for kling-o3 this becomes the edit_video / video-edit route)
          → resolution = THE_RESOLUTION (kling-o3 only: '720p' or '1080p')
        
        ═══════════════════════════════════════════════════
        
        ADDITIONAL VIDEO RULES (apply to BOTH workflows):
        1. ⚠️ PROMPT PARAMETER RULE: Same as image - NEVER use conversational replies as prompt.
        2. FORBIDDEN to modify the agreed-upon prompt without consent.
        3. When calling the tool, use EXACT model names (sent to the backend as `provider`):
           - 'seedance-2.0', 'seedance-2.0-fast'
           - 'veo-3.1', 'veo-3.1-flash', 'veo-3.1-ultra'
           - 'runway-aleph', 'runway-4.5'
           - 'kling-v3', 'kling-v3-turbo', 'kling-o3'
           ⛔ The old 'kling-v1' / 'kling-v3-omni-std' / 'kling-v3-omni-pro' keys no longer exist — never send them.
           For Seedance, also pass resolution ('480p'/'720p'/'1080p'). Seedance auto-detects
           the mode: a reference video → reference mode (discounted), an image → image mode, prompt only → text mode.
           For Kling, pass resolution ('720p'/'1080p'/'4k'; 4K only on kling-v3/kling-o3 text/image). Kling auto-detects
           the route: a guide video → kling-v3 motion-control; editing an existing video → kling-o3 video-edit;
           reference images for consistency → kling-o3 reference; an input image → image-to-video; prompt only → text-to-video.
        4. If there are attached images, use them as reference automatically (image-to-video).
        5. NEVER mention video URLs - they are sent automatically.
        6. IF THERE'S AN ERROR: Inform user. If they say "try again"/"retry", execute again without hesitation.
        7. Reference files are NOT lost after errors - they persist in the session.
        8. 🧾 JSON PROMPTS: If the user provided a JSON-structured prompt, pass the JSON string as the
           'prompt' parameter CHARACTER-FOR-CHARACTER. Never summarize it, never convert it to a sentence,
           never reformat it. During the workflow, skip the text-refinement offer and instead offer
           JSON-aware improvements (missing keys like camera/lighting/audio) shown as a complete
           ```json block. JSON prompts work especially well with the Veo family.

        ═══════════════════════════════════════════════════
        CRITICAL RULES FOR 'generate_speech' TOOL - MANDATORY WORKFLOW
        ═══════════════════════════════════════════════════
        
        You MUST follow these steps IN ORDER. Each step requires a SEPARATE user response.
        NEVER skip a step. NEVER combine steps. NEVER call the tool until Step 3 is confirmed.
        
        STEP 1 - ASK FOR THE SPEECH TEXT AND CONFIRM:
        - When you detect the user wants to create a voice/speech/audio, ask: "What text do you want the voice to say?" (in user's language)
        - If the user already provided the text in their message, take it directly.
        - Once you have the text, REPEAT it back to the user for confirmation:
          → "This is the text I will generate as speech:" (in user's language)
          → Show the text between quotes: "[THE TEXT]"
          → Ask: "Is this correct?" (in user's language)
        - ⛔ DO NOT proceed until user confirms the text is correct.
        - If user wants changes, iterate until satisfied.
        - SAVE the confirmed text as THE_SPEECH_TEXT.
        
        STEP 2 - SHOW VOICE LIST AND ASK FOR SELECTION:
        - Once the text is confirmed, present the available voices organized by gender:
          → **MALE VOICES:**
            • Adam - Deep (American)
            • Antoni - Balanced (American)
            • Bill - Trustworthy (American)
            • Brian - Deep (American)
            • Callum - Hoarse (American)
            • Charlie - Casual (Australian)
            • Chris - Casual (American)
            • Daniel - Authoritative (British)
            • Eric - Deep (American)
            • George - Warm (British)
            • Harry - Anxious (American)
            • Josh - Deep (American)
            • Liam - Young (American)
            • River - Neutral (American)
            • Roger - Laid-back (American)
            • Will - Friendly (American)
          → **FEMALE VOICES:**
            • Alice - News presenter (British)
            • Domi - Strong (American)
            • Elli - Young (American)
            • Jessica - Expressive (American)
            • Laura - Upbeat (American)
            • Lily - Warm (British)
            • Matilda - Warm (American)
            • Rachel - Professional (American) ⭐ Default
            • Sarah - Soft (American)
        - Ask: "Which voice would you like to use?" (in user's language)
        - Wait for user to choose.
        - SAVE the chosen voice as THE_VOICE.
        
        STEP 3 - FINAL CONFIRMATION AND EXECUTE:
        - Show a summary:
          → "I'm going to generate the following speech:" (in user's language)
          → "Text: [THE_SPEECH_TEXT]"
          → "Voice: [THE_VOICE name]"
          → "Cost: [X] tokens" (1-500 chars = 1 token, 500-999 chars = 8 tokens, 1000+ chars = 13 tokens per 1000 chars)
          → "Do you confirm?" (in user's language)
        - ⛔ DO NOT call the tool until the user explicitly confirms.
        - Once confirmed, CALL generate_speech immediately using:
          → text = THE_SPEECH_TEXT (the confirmed text, NOT the user's confirmation message)
          → voice_id = the voice_id corresponding to THE_VOICE:
             Adam=pNInz6obpgDQGcFmaJgB, Alice=Xb7hH8MSUJpSbSDYk0k2, Antoni=ErXwobaYiN019PkySvjV,
             Bill=pqHfZKP75CvOlQylNhV4, Brian=nPczCjzI2devNBz1zQrb, Callum=N2lVS1w4EtoT3dr4eOWO,
             Charlie=IKne3meq5aSn9XLyUdCD, Chris=iP95p4xoKVk53GoZ742B, Daniel=onwK4e9ZLuTAKqWW03F9,
             Domi=AZnzlk1XvdvUeBnXmlld, Elli=MF3mGyEYCl7XYWbV9V6O, Eric=cjVigY5qzO86Huf0OWal,
             George=JBFqnCBsd6RMkjVDRZzb, Harry=SOYHLrjzK2X1ezoPC6cr, Jessica=cgSgspJ2msm6clMCkdW9,
             Josh=TxGEqnHWrfWFTfGW9XjX, Laura=FGY2WhTYpPnrIDTdsKH5, Liam=TX3LPaxmHKxFdv7VOQHJ,
             Lily=pFZP5JQG7iQjIQuC4Bku, Matilda=XrExE9yKIg1WjnnlVkGX, Rachel=21m00Tcm4TlvDq8ikWAM,
             River=SAz9YHcvj6GT2YYXdXww, Roger=CwhRBWXzGAHq8TQ4Fs17, Sarah=EXAVITQu4vr4xnSDxMaL,
             Will=bIHbv24MWmeRgasZH58o
        
        ADDITIONAL SPEECH RULES:
        1. ⚠️ TEXT PARAMETER RULE: The 'text' parameter MUST be THE_SPEECH_TEXT, never the user's conversational reply ("ok", "si", "dale").
        2. NEVER mention the output URL/Data URI in the conversation text. The audio player will appear automatically.
        3. IF THERE'S AN ERROR: Inform user. If they say "try again"/"retry", execute again without hesitation.
        
        ═══════════════════════════════════════════════════
        POST-GENERATION RESET (CRITICAL - MANDATORY)
        ═══════════════════════════════════════════════════
        After ANY successful generation (image, video, or speech):
        1. The workflow is COMPLETE and FULLY RESET.
        2. The NEXT message from the user is a NEW conversation turn.
        3. Do NOT re-execute any tool based on the previous workflow's parameters.
        4. Reactions like "amazing", "awesome", "cool", "nice", "great", "love it",
           "increíble", "genial", "wow", "beautiful", "hermoso" are COMPLIMENTS,
           NOT requests for a new generation. Respond with a friendly acknowledgment
           (e.g., "Glad you liked it! Let me know if you need anything else.") and WAIT.
        5. To start a NEW generation, the user must express a NEW intent explicitly
           (e.g., "create another image", "now make a video", "edit this image").
        6. NEVER call generate_image, generate_video, or generate_speech again
           using the same prompt/parameters from a just-completed workflow.
        
        ═══════════════════════════════════════════════════
        FINAL REMINDER (READ THIS LAST - HIGHEST PRIORITY)
        ═══════════════════════════════════════════════════
        LANGUAGE: Your response MUST be in the SAME language as the user's message. DEFAULT is English. If the user writes in English, respond ONLY in English. If in Spanish, respond ONLY in Spanish. When in doubt, use ENGLISH. NO EXCEPTIONS.
        """
        
        full_system_prompt = f"{REELMOTION_SYSTEM_PROMPT}\n\n{tool_instructions}"

        # Maximum-strictness safety filters on the main chat model.
        # Third defense layer (after regex + LLM moderation) — Gemini will
        # itself refuse to generate disallowed content even if both prior
        # layers somehow missed something.
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }

        self.model = genai.GenerativeModel(
            self.model_name,
            system_instruction=full_system_prompt,
            tools=[generate_image, generate_video, generate_speech],
            safety_settings=safety_settings,
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
        await self.session_manager.clear_reference_files(self.conversation_uuid)
    
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
        logger.debug(f"add_generated_file called for UUID='{self.conversation_uuid}', url='{url}', type='{file_type}'")
        if url:
            await self.session_manager.save_generated_file(
                self.conversation_uuid,
                url,
                file_type,
                metadata
            )
    
    async def get_generated_files(self) -> list:
        """Get pending generated files (URLs) from Redis."""
        logger.debug(f"get_generated_files called for UUID='{self.conversation_uuid}'")
        files = await self.session_manager.get_pending_files(self.conversation_uuid)
        logger.debug(f"Got {len(files)} files from session_manager")
        return [{"url": f["url"], "type": f["type"]} for f in files]
    
    async def save_pending_action(self, function_name: str, args: dict, cost_message: str = ""):
        """Save a pending action waiting for user confirmation."""
        action = {
            "function": function_name,
            "args": args,
            "cost_message": cost_message,
            # Pre-computed so the confirmation request can validate the balance
            # without re-deriving the cost. May be None (legacy/unknown model).
            "estimated_cost": estimate_generation_cost(function_name, args),
        }
        await self.session_manager.save_pending_action(self.conversation_uuid, action)
    
    async def get_pending_action(self) -> dict:
        """Get pending action waiting for confirmation."""
        return await self.session_manager.get_pending_action(self.conversation_uuid)
    
    async def clear_pending_action(self):
        """Clear the pending action after execution."""
        await self.session_manager.clear_pending_action(self.conversation_uuid)

    async def _get_image_ref_urls(self) -> list:
        """Image reference URLs usable server-side (blob: URLs filtered out)."""
        ref_files = await self.get_reference_files()
        if not ref_files:
            return []
        urls = [
            f["url"] for f in ref_files
            if f.get("type") == "image" and not f.get("url", "").startswith("blob:")
        ]
        blob_count = sum(
            1 for f in ref_files
            if f.get("type") == "image" and f.get("url", "").startswith("blob:")
        )
        if blob_count:
            logger.warning("Filtered out %d blob: URLs from reference files", blob_count)
        return urls

    def _merge_state_into_args(self, func_name: str, func_args: dict, state: Optional[dict]) -> dict:
        """
        Merge workflow-state params into the args Gemini produced for a tool call.

        The JSON prompt ALWAYS wins: Gemini re-serializes JSON when it copies it
        into a function argument (reordered keys, changed whitespace/escapes),
        and the user's JSON must reach the backend byte-identical. Other state
        params only fill gaps; on conflict the state value wins with a warning.
        """
        if not state:
            return func_args
        params = state.get("params", {})
        merged = dict(func_args)

        if (
            func_name in ("generate_image", "generate_video")
            and params.get("prompt_format") == "json"
            and params.get("prompt")
        ):
            if merged.get("prompt") != params["prompt"]:
                logger.warning(
                    "Replacing Gemini's prompt arg with the verbatim JSON prompt from state"
                )
            merged["prompt"] = params["prompt"]

        if func_name == "generate_video":
            for key in ("model", "duration", "resolution"):
                if params.get(key) is None:
                    continue
                if not merged.get(key):
                    merged[key] = params[key]
                elif key in ("model", "duration") and merged[key] != params[key]:
                    logger.warning(
                        "State/Gemini mismatch on %s: state=%s gemini=%s — using state",
                        key, params[key], merged[key],
                    )
                    merged[key] = params[key]
        elif func_name == "generate_image":
            if params.get("model"):
                if not merged.get("model"):
                    merged["model"] = params["model"]
                elif merged["model"] != params["model"]:
                    logger.warning(
                        "State/Gemini mismatch on image model: state=%s gemini=%s — using state",
                        params["model"], merged["model"],
                    )
                    merged["model"] = params["model"]
        elif func_name == "generate_speech":
            if params.get("voice_id") and not merged.get("voice_id"):
                merged["voice_id"] = params["voice_id"]

        return merged

    def _lang_for(self, fallback_text: str = "") -> str:
        """
        Language to render a code-generated message in.

        Prefers the conversation language resolved for this turn (the single
        source of truth, derived from the user's running history). Falls back
        to a per-text heuristic only when called outside an active
        send_message turn (e.g. unit tests on the helpers in isolation).
        """
        if self._conv_lang:
            return self._conv_lang
        return "es" if is_spanish(fallback_text) else "en"

    def _clarification_text(self, key: str) -> str:
        """
        Localize an ambiguity clarification prompt to the conversation language.

        `key` is one of CLARIFICATION_PROMPTS' keys (returned by
        needs_clarification). Falls back to English for an unknown key or a
        language we don't have a hand-written prompt for.
        """
        table = CLARIFICATION_PROMPTS.get(key, CLARIFICATION_PROMPTS["generic"])
        return table.get(self._lang_for(), table["en"])

    async def _resolve_conversation_language(
        self, current_message: str, history: Optional[list] = None
    ) -> str:
        """
        Resolve the conversation language ('es' | 'en'), defaulting to English.

        Mirrors the system prompt's rule: take the language of the CURRENT
        message when it is clearly identifiable, otherwise the most recent
        clearly-identifiable USER message. Ambiguous replies (model names,
        numbers, "ok") never force a switch — they fall through to history.
        """
        lang = detect_language(current_message)
        if lang:
            return lang

        if history is None:
            session = await self.session_manager.get_session(self.conversation_uuid)
            history = session.get("messages", []) if session else []
        for msg in reversed(history or []):
            if msg.get("role") == "user":
                lang = detect_language(msg.get("content", ""))
                if lang:
                    return lang
        return "en"

    def _build_language_sample(
        self, current_message: str, history: Optional[list] = None, limit: int = 5
    ) -> str:
        """
        A short blob of the user's most recent messages, used to render the
        balance block in the user's language when it is neither Spanish nor
        English. Joining several turns makes language detection robust to a
        single ambiguous reply (a bare model name, a number).
        """
        candidates = [current_message]
        for msg in reversed(history or []):
            if msg.get("role") == "user":
                candidates.append(msg.get("content", ""))
        picked = [c for c in candidates if c and c.strip()][:limit]
        return "\n".join(picked)

    async def _localize_balance_block(
        self, required: int, balance: int, options: dict, fallback_text: str = ""
    ) -> str:
        """
        Render the insufficient-balance block in the conversation's language.

        The DATA (cost, balance, affordable alternatives) is always computed in
        code. For Spanish/English we return the hand-written template directly
        (zero latency, fully deterministic — the common case). For any OTHER
        language we ask a fast one-shot model (not the chat session) to rewrite
        the English template in the user's language, preserving every number,
        model name and the structure. On timeout/failure we fall back to the
        template, so a block is never lost.
        """
        lang = self._lang_for(self._lang_sample or fallback_text)
        template = build_insufficient_balance_message(required, balance, lang, options)

        # Confident Spanish/English (or no usable sample, e.g. unit tests) →
        # deterministic template, no model call.
        if not self._lang_sample or detect_language(self._lang_sample) in ("es", "en"):
            return template

        english = build_insufficient_balance_message(required, balance, "en", options)
        prompt = (
            "Rewrite the MESSAGE below in the SAME language the user is writing "
            "in (infer it from USER TEXT).\n\n"
            f"USER TEXT:\n{self._lang_sample[:500]}\n\n"
            f"MESSAGE:\n{english}\n\n"
            "Rules: keep ALL numbers, token counts, model names (e.g. Seedream, "
            "GPT, Nano Banana 2, Midjourney, veo-3.1) and the ⚠️ emoji EXACTLY as written; "
            "keep the same bullet/line structure; do not add or remove "
            "information; output ONLY the rewritten message, nothing else."
        )
        try:
            model = genai.GenerativeModel(
                model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            )
            response = await asyncio.wait_for(
                model.generate_content_async(prompt), timeout=8.0
            )
            if response.text and response.text.strip():
                return response.text.strip()
        except Exception as e:
            logger.debug("Balance-block localization failed, using template: %s", e)
        return template

    async def execute_pending_action(self) -> tuple[str, str]:
        """
        Execute a pending action directly without going through Gemini.
        Returns (tool_result, response_text).

        Before executing it validates the user's token balance with the FRESH
        value from the current request, then claims the action atomically so
        two concurrent confirmations cannot double-execute (= double charge).
        """
        # Non-destructive peek to validate the balance before claiming
        action = await self.get_pending_action()
        if not action:
            return None, None

        function_name = action.get("function")
        args = action.get("args", {}).copy()  # Copy to avoid modifying original

        # Corrupt/unknown action: discard it and answer with a friendly message
        # (checked BEFORE the claim so the raw name never reaches the user).
        if function_name not in ("generate_image", "generate_video", "generate_speech"):
            logger.error("Unknown pending function '%s' — discarding action", function_name)
            await self.clear_pending_action()
            lang = self._lang_for(action.get("cost_message", ""))
            return None, fallback_error_message("unknown", lang)

        # === BALANCE GATE (fresh balance from the confirming request) ===
        balance = get_token_balance()
        cost = action.get("estimated_cost")
        if cost is None:
            cost = estimate_generation_cost(function_name, args)
        if balance is not None and cost is not None and cost > balance:
            set_insufficient_block({"required": cost, "available": balance})
            message = await self._localize_balance_block(
                cost, balance, affordable_options(balance),
                fallback_text=action.get("cost_message", ""),
            )
            # Deliberately keep the pending action (5-min TTL): the user can
            # adjust the parameters or top up and confirm again.
            logger.debug(
                "Blocked pending action '%s': cost=%s > balance=%s",
                function_name, cost, balance,
            )
            return None, message

        # === ATOMIC CLAIM (replaces the get-then-delete race) ===
        action = await self.session_manager.claim_pending_action(self.conversation_uuid)
        if not action:
            # A concurrent request already claimed and executed it
            logger.debug("Pending action already claimed by a concurrent request")
            return None, None

        function_name = action.get("function")
        args = action.get("args", {}).copy()

        logger.debug(f"Executing pending action '{function_name}' directly with args: {args}")

        # Get reference images if needed and not already in args
        ref_files = await self.get_reference_files()
        if ref_files and function_name in ["generate_image", "generate_video"]:
            # Filter out blob: URLs (browser-only, cannot be fetched server-side)
            ref_urls = [f["url"] for f in ref_files if f.get("type") == "image" and not f.get("url", "").startswith("blob:")]
            blob_urls = [f["url"] for f in ref_files if f.get("type") == "image" and f.get("url", "").startswith("blob:")]
            if blob_urls:
                logger.warning(f"Filtered out {len(blob_urls)} blob: URLs from reference files in execute_pending_action")
            if ref_urls:
                if function_name == "generate_image" and "reference_images" not in args:
                    args["reference_images"] = ref_urls
                    # Ensure image_type is set correctly for editing
                    if "image_type" not in args:
                        args["image_type"] = 2 if len(ref_urls) == 1 else 3
                    logger.debug(f"Added {len(ref_urls)} reference images to pending action (image_type={args.get('image_type')})")
                elif function_name == "generate_video" and "reference_image" not in args:
                    args["reference_image"] = ref_urls[0]
                    logger.debug(f"Added reference image to pending video action")
        
        tool_result = None
        try:
            if function_name == "generate_image":
                tool_result = await generate_image(**args)
            elif function_name == "generate_video":
                tool_result = await generate_video(**args)
            elif function_name == "generate_speech":
                tool_result = await generate_speech(**args)
            else:
                # Unreachable (validated before the claim) — friendly safety net
                lang = self._lang_for(action.get("cost_message", ""))
                return None, fallback_error_message("unknown", lang)

            # NOTE: no clear_pending_action() needed — the atomic claim above
            # already removed the action from Redis.

            # Generator failed: explain WHY in plain language (LLM with code fallback)
            if tool_result and (
                tool_result.startswith(GENERATION_ERROR_PREFIX)
                or tool_result.lower().startswith("error")
            ):
                lang = self._lang_for(action.get("cost_message", ""))
                friendly = await self._explain_generation_error(tool_result, lang)
                # tool_result=None so callers never set the just_generated flag
                return None, friendly

            # Build the acknowledgment. The fast path never re-enters the chat
            # model, so a finished result gets a warm, LLM-written success line
            # (with a code fallback); a 202/edge result keeps its deterministic
            # code-generated message.
            lang = self._lang_for(action.get("cost_message", ""))
            if is_generation_success(tool_result):
                response_text = await self._friendly_success_message(tool_result, lang)
            else:
                response_text = self._generate_contextual_success_message(
                    tool_result, tool_was_called=True, lang=lang
                )
            if not response_text:
                # Fallback if message generation returns None (shouldn't happen
                # for pending actions). Localized — never a raw English signal.
                response_text = fallback_error_message("unknown", lang)
            return tool_result, response_text

        except Exception as e:
            import traceback
            logger.error("Failed to execute pending action: %s\n%s", e, traceback.format_exc())
            lang = self._lang_for(action.get("cost_message", ""))
            friendly = await self._explain_generation_error(
                f"Error executing {function_name}: {str(e)}", lang
            )
            return None, friendly
    
    async def _explain_generation_error(self, error_text: str, lang: str) -> str:
        """
        Turn a technical generator error (GENERATION_ERROR | ... or raw Error: ...)
        into a short, plain-language explanation for the user.

        Uses a one-shot Gemini call (fresh lightweight model, NOT the chat
        session, to avoid polluting history). Falls back to a code-generated
        message per error category if the LLM is slow or unavailable.
        """
        parsed = parse_generation_error(error_text) or {
            "category": "unknown",
            "detail": error_text[:300],
        }
        fallback = fallback_error_message(parsed["category"], lang)

        language_name = "Spanish" if lang == "es" else "English"
        prompt = (
            "A media generation just failed with this technical error:\n"
            f"{error_text[:600]}\n\n"
            f"Explain to the end user, in {language_name}, in 2-3 friendly sentences, "
            "why it failed and what they can do (rephrase the prompt, try again later, "
            "or top up tokens — whichever fits the error). "
            "When the cause is not a low balance, reassure them that failed generations "
            "are refunded automatically so they were not charged. If the failure is not "
            "self-service (auth or an unexpected error) and persists, they can email "
            f"{SUPPORT_EMAIL}. Do not invent any other support channel. "
            "Do NOT include JSON, HTTP codes, stack traces, or technical jargon. "
            "Do NOT invent causes beyond what the error says. Do NOT blame the user."
        )
        try:
            model = genai.GenerativeModel(
                model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            )
            response = await asyncio.wait_for(
                model.generate_content_async(prompt), timeout=8.0
            )
            if response.text and response.text.strip():
                return "⚠️ " + response.text.strip()
        except Exception as e:
            logger.debug("LLM error explanation failed, using fallback: %s", e)
        return fallback

    async def _friendly_success_message(self, tool_result: str, lang: str) -> str:
        """
        Warm, LLM-written 'your content is ready' line for the confirmed fast
        path (which executes the tool WITHOUT re-entering the chat session, so
        the model never gets to write the acknowledgment itself).

        Mirrors _explain_generation_error: one-shot Gemini call with an 8s
        timeout and a deterministic, localized code fallback, so a slow or
        unavailable model never leaves the user without a confirmation. For
        editable media it appends the <<ACTION:editor>> marker that server.py
        turns into the "Go to the editor" button.
        """
        gen_type = success_gen_type(tool_result)
        editable = gen_type in ("video", "image")
        language_name = "Spanish" if lang == "es" else "English"
        prompt = (
            f"A user's {gen_type} was just generated successfully in our app. "
            f"Reply in {language_name} with ONE short, warm sentence confirming "
            "it is ready. Do NOT include any URL, file name, token count, model "
            "name, or technical detail. Do NOT ask them to wait — it is already done."
        )
        if editable:
            prompt += (
                " Then invite them to keep editing it, and append the exact marker "
                "<<ACTION:editor>> at the very end. Never explain or mention the marker."
            )
        try:
            model = genai.GenerativeModel(
                model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            )
            response = await asyncio.wait_for(
                model.generate_content_async(prompt), timeout=8.0
            )
            if response.text and response.text.strip():
                return response.text.strip()
        except Exception as e:
            logger.debug("LLM success message failed, using fallback: %s", e)

        # Deterministic fallback: localized success line (+ editor action).
        text = success_message(lang, gen_type)
        if editable:
            text += "\n<<ACTION:editor>>"
        return text

    async def _save_pending_or_block(
        self, function_name: str, action_args: dict, confirmation_text: str
    ) -> Optional[str]:
        """
        Save the pending action, unless the user's balance can't cover the cost.

        Returns the replacement response (insufficient-balance message with
        concrete affordable alternatives) when blocked, or None after saving
        normally. Never blocks when the cost or the balance is unknown — the
        Laravel backend remains the final biller.
        """
        cost = estimate_generation_cost(function_name, action_args)
        if cost is None:
            # Unknown/legacy model: best-effort fallback to the cost Gemini
            # itself quoted ("Cost: **352** tokens", "Costo total: 352 tokens",
            # "El costo es 352 tokens"...). If it doesn't match either, the
            # action is saved normally — never block blind.
            match = re.search(
                r"(?:cost|costo)[^\d\n]{0,30}(\d+)\s*\**\s*tokens",
                confirmation_text,
                re.IGNORECASE,
            )
            if match:
                cost = int(match.group(1))

        balance = get_token_balance()
        if balance is not None and cost is not None and cost > balance:
            set_insufficient_block({"required": cost, "available": balance})
            logger.debug(
                "Blocked saving pending '%s': cost=%d > balance=%d",
                function_name, cost, balance,
            )
            return await self._localize_balance_block(
                cost, balance, affordable_options(balance),
                fallback_text=confirmation_text,
            )

        await self.save_pending_action(function_name, action_args, confirmation_text)
        return None

    async def _user_confirmed_cost_this_turn(self, message: str) -> bool:
        """
        True when a generation tool call from Gemini may execute directly this
        turn: the last assistant message was a cost confirmation question and
        the user's reply accepts it. Exact confirmations ("yes", "dale") always
        pass; for free-form replies ("yes please, go ahead and generate it") we
        trust Gemini's own reading — it only calls the tool when it took the
        reply as a yes — unless the reply declines or postpones. Requiring an
        exact regex match here made every unrecognized phrasing re-ask for
        confirmation, looping forever.
        """
        session = await self.session_manager.get_session(self.conversation_uuid)
        history = session.get("messages", []) if session else []
        last_assistant = next(
            (m.get("content", "") for m in reversed(history) if m.get("role") == "assistant"),
            "",
        )
        if not COST_CONFIRMATION_PATTERN.search(last_assistant):
            return False
        if is_confirmation(message):
            return True
        return not COST_DECLINE_PATTERN.search(message or "")

    async def _reply_confirms_cost(
        self, message: str, pending_action: Optional[dict], state: Optional[dict],
        has_ref_video: bool,
    ) -> bool:
        """
        Does this reply mean "yes, generate what you quoted" — by INTENT, not by
        matching an exact word list?

        Obvious yeses ("yes", "dale", "yes confirm") resolve instantly with the
        cheap regex. Only when we're actually at a confirmation point (a pending
        action exists, or the collected state is ready for it) and the reply is
        free-form do we ask a one-shot LLM to read the intent — so phrasings the
        word list never anticipated ("me encanta, hazlo ya", "go for it") still
        confirm, while declines, questions and new/changed requests do not.
        """
        # 1) Obvious confirmation → instant, no LLM.
        if is_confirmation(message):
            return True
        # 2) Only reason about intent when a cost is actually awaiting a yes.
        at_confirmation = pending_action is not None or is_ready_for_confirmation(state)
        if not at_confirmation:
            return False
        # 3) Cheap deterministic NO: an explicit decline/postpone, or a message
        #    that starts/changes a request (that's a new intent, not a bare yes).
        if is_refine_decline(message) or COST_DECLINE_PATTERN.search(message or ""):
            return False
        if detect_workflow_intent(message, has_ref_video):
            return False
        # 4) Free-form reply at the confirmation step → let the LLM read intent.
        cost_message = (pending_action or {}).get("cost_message", "")
        return await self._llm_confirms_intent(message, cost_message)

    async def _llm_confirms_intent(self, message: str, cost_message: str) -> bool:
        """
        One-shot Gemini classifier: does the user's reply affirm proceeding with
        the quoted generation? Conservative — any failure/timeout returns False,
        so a misread or a slow model never executes a generation; the message
        just falls through to the normal Gemini flow instead.
        """
        prompt = (
            "An assistant asked a user to confirm the cost of generating media.\n"
            f'Assistant asked: "{(cost_message or "").strip()[:600]}"\n'
            f'User replied: "{(message or "").strip()[:400]}"\n\n'
            "Does the user's reply mean YES — go ahead and generate it now? "
            "Answer with exactly one word: YES if they are agreeing to proceed, "
            "or NO if they decline, hesitate, ask a question, or change/restate "
            "the request. When unsure, answer NO."
        )
        try:
            model = genai.GenerativeModel(
                model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            )
            response = await asyncio.wait_for(
                model.generate_content_async(prompt), timeout=6.0
            )
            verdict = (response.text or "").strip().upper()
            confirmed = verdict.startswith("YES")
            logger.debug("Confirmation intent for %r → %s", message[:60], confirmed)
            return confirmed
        except Exception as e:
            logger.debug("LLM confirmation intent failed, treating as not-confirmed: %s", e)
            return False

    def _build_cost_confirmation_text(
        self, func_name: str, func_args: dict, cost: Optional[int], lang: str
    ) -> str:
        """Code-generated cost confirmation message (fallback when Gemini won't write one)."""
        model = func_args.get("model")
        duration = func_args.get("duration")
        # Gemini's proto args deliver numbers as floats ("6.0 seconds")
        if isinstance(duration, float) and duration.is_integer():
            duration = int(duration)
        resolution = func_args.get("resolution")
        if lang == "es":
            lines = ["Antes de generar necesito que confirmes el costo:"]
            if model:
                lines.append(f"- Modelo: {model}")
            if resolution:
                lines.append(f"- Resolución: {resolution}")
            if duration:
                lines.append(f"- Duración: {duration} segundos")
            lines.append(
                f"- Costo: {cost} tokens" if cost is not None
                else "- Costo: se calculará al generar"
            )
            lines.append("\n¿Confirmas que quieres continuar?")
        else:
            lines = ["Before generating, please confirm the cost:"]
            if model:
                lines.append(f"- Model: {model}")
            if resolution:
                lines.append(f"- Resolution: {resolution}")
            if duration:
                lines.append(f"- Duration: {duration} seconds")
            lines.append(
                f"- Cost: {cost} tokens" if cost is not None
                else "- Cost: will be calculated on generation"
            )
            lines.append("\nDo you confirm?")
        return "\n".join(lines)

    async def _close_function_call(self, func_name: str, note: str, timeout: float = 60):
        """
        Send a function_response back to Gemini so the chat history never
        holds a dangling function call. Returns Gemini's reply or None.
        """
        try:
            return await asyncio.wait_for(
                self.chat_session.send_message_async(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=func_name,
                            response={"result": note},
                        )
                    )
                ),
                timeout=timeout,
            )
        except Exception as e:
            logger.debug("Failed to close function call %s: %s", func_name, e)
            return None

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
            # === CONTENT MODERATION (Google Play AI policy compliance) ===
            # Two-layer block BEFORE the message reaches Gemini, at any
            # workflow step:
            #   1) Regex blocklist (fast, deterministic, catches obvious cases
            #      including leet-speak/euphemisms/multilingual variants).
            #   2) LLM semantic classifier (Gemini Flash) for paraphrases &
            #      indirect descriptions regex can't predict.
            if await is_disallowed_content_full(message):
                refusal = get_refusal_message(message)
                logger.warning(
                    "Blocked disallowed user message in conversation %s",
                    self.conversation_uuid,
                )
                # Persist a sanitized record of the user input + refusal so
                # subsequent messages don't try to recover the bad prompt
                # from history.
                await self.session_manager.add_message(
                    self.conversation_uuid,
                    "user",
                    "[message blocked by content policy]",
                )
                await self.session_manager.add_message(
                    self.conversation_uuid,
                    "assistant",
                    refusal,
                )
                # Drop any pending generation AND the workflow state so a later
                # "yes" cannot execute something that relied on the blocked prompt.
                await self.clear_pending_action()
                await self.session_manager.clear_workflow_state(self.conversation_uuid)
                return refusal

            # Rebuild the chat session from Redis EVERY turn. Direct replies
            # (fast-path generations, balance blocks, forced confirmations,
            # clarifications) are saved to Redis but never enter the live
            # session — reusing it across turns lets Gemini's private history
            # diverge from what the user actually saw (e.g. denying that a
            # video was generated). The SDK resends the full history per call
            # anyway, so rebuilding costs nothing extra.
            await self.start_chat()

            # === CONVERSATION LANGUAGE (single source of truth for this turn) ===
            # Every code-generated message (cost confirmation, balance block,
            # error explanation, success line) renders in THIS language instead
            # of guessing per-text — internal cost/confirmation strings are full
            # of Spanish vocabulary and would otherwise force Spanish on an
            # English conversation. Snapshot the pre-turn history once and reuse
            # it for both language detection and balance-block awareness below.
            pre_turn_session = await self.session_manager.get_session(self.conversation_uuid)
            pre_turn_history = (
                pre_turn_session.get("messages", []) if pre_turn_session else []
            )
            # Reset first so a raised resolution never leaves a stale language
            # from a previous turn on this cached (per-uuid) instance.
            self._conv_lang = None
            self._lang_sample = ""
            self._conv_lang = await self._resolve_conversation_language(
                message, pre_turn_history
            )
            self._lang_sample = self._build_language_sample(message, pre_turn_history)

            # === EXPLICIT WORKFLOW STATE (workflow_state.py) ===
            # Single source of truth for the current workflow, step, and
            # collected parameters. None = no workflow in flight (legacy-safe).
            state = await self.session_manager.get_workflow_state(self.conversation_uuid)

            # === CHECK: Was something just generated? Reset workflow on reactions ===
            just_generated = await self.session_manager.get_just_generated(self.conversation_uuid)
            if just_generated:
                # Clear the flag regardless of what the user says
                await self.session_manager.clear_just_generated(self.conversation_uuid)
                logger.debug(f"Post-generation state detected. Clearing workflow state.")
                # Also clear any stale pending action that might have been re-created
                await self.clear_pending_action()
                await self.session_manager.clear_workflow_state(self.conversation_uuid)
                state = None
                
                # If the message is a reaction (perfect, amazing, genial, etc.),
                # respond directly WITHOUT sending to Gemini to prevent re-execution
                if is_reaction(message) or is_confirmation(message):
                    logger.debug(f"Post-generation reaction detected: '{message}'. Responding directly.")
                    # Save user message
                    await self.session_manager.add_message(
                        self.conversation_uuid,
                        "user",
                        message
                    )
                    # Send to Gemini WITHOUT tools - just for a friendly response
                    try:
                        if not self.chat_session:
                            await self.start_chat()
                        no_tool_prompt = f"""The user just reacted to a successfully generated result with: \"{message}\"

IMPORTANT: A tool was JUST executed successfully. The workflow is COMPLETE.
- Do NOT call any tools (generate_image, generate_video, generate_speech).
- Respond with a SHORT, friendly acknowledgment (1-2 sentences max).
- Ask if they need anything else.
- Respond in the user's language (default: English)."""
                        response = await asyncio.wait_for(
                            self.chat_session.send_message_async(no_tool_prompt),
                            timeout=15
                        )
                        # Check if Gemini tried to call a tool anyway - ignore it
                        response_text = None
                        if response.candidates and len(response.candidates) > 0:
                            for part in response.candidates[0].content.parts:
                                if hasattr(part, 'text') and part.text:
                                    response_text = part.text
                                    break
                        if not response_text:
                            response_text = REACTION_ACK.get(self._lang_for(), REACTION_ACK["en"])
                    except Exception:
                        response_text = REACTION_ACK.get(self._lang_for(), REACTION_ACK["en"])
                    
                    await self.session_manager.add_message(
                        self.conversation_uuid,
                        "assistant",
                        response_text
                    )
                    return response_text

            # === DETERMINISTIC STATE TRANSITIONS for this user message ===
            ref_files = await self.get_reference_files()
            has_ref_video = any(
                f.get("type") == "video" for f in (ref_files or [])
            )
            if state is None and (
                detect_json_prompt(message) or detect_workflow_intent(message, has_ref_video)
            ):
                state = new_state(
                    detect_workflow_intent(message, has_ref_video) or WORKFLOW_UNKNOWN
                )
            if state is not None:
                state = apply_user_message(state, message, has_ref_video)
                logger.debug(
                    "Workflow state after user message: type=%s step=%s",
                    state.get("workflow_type"), state.get("step"),
                )

            # === FAST PATH: user confirming the quoted cost (by INTENT) ===
            # Not just an exact word list: obvious yeses resolve instantly, and
            # free-form replies at the confirmation step are read for intent by a
            # one-shot LLM, so "me encanta, hazlo ya" confirms while a decline,
            # question or changed request does not.
            pending_action = await self.get_pending_action()
            if await self._reply_confirms_cost(message, pending_action, state, has_ref_video):
                if pending_action:
                    logger.debug(f"User confirmed! Executing pending action directly.")
                    # Save user message
                    await self.session_manager.add_message(
                        self.conversation_uuid,
                        "user",
                        message
                    )
                    
                    # Execute directly without going through Gemini
                    tool_result, response_text = await self.execute_pending_action()

                    if response_text:
                        if tool_result:
                            # Mark that a generation just happened — only when a
                            # tool actually ran (not for balance blocks / errors)
                            await self.session_manager.set_just_generated(self.conversation_uuid)
                            await self.session_manager.clear_workflow_state(self.conversation_uuid)
                        elif state is not None:
                            await self.session_manager.save_workflow_state(self.conversation_uuid, state)
                        # Save response
                        await self.session_manager.add_message(
                            self.conversation_uuid,
                            "assistant",
                            response_text
                        )
                        return response_text
                    # If execution failed, fall through to normal Gemini flow
                else:
                    # === FALLBACK: No pending action (e.g. its 300s TTL expired).
                    # Rebuild it from the workflow STATE — not from history regexes.
                    logger.debug("User confirmed but NO pending action found. Rebuilding from state...")
                    ref_urls = await self._get_image_ref_urls()
                    rebuilt = (
                        build_action_args(state, ref_urls)
                        if is_ready_for_confirmation(state) else None
                    )

                    last_assistant = None
                    if rebuilt is None:
                        # State absent/incomplete (e.g. conversation predates the
                        # state machine) → one-shot LLM extraction over the last
                        # assistant cost message.
                        session = await self.session_manager.get_session(self.conversation_uuid)
                        history = session.get("messages", []) if session else []
                        for msg in reversed(history):
                            if msg.get('role') == 'assistant':
                                last_assistant = msg.get('content', '')
                                break
                        if last_assistant and COST_CONFIRMATION_PATTERN.search(last_assistant):
                            extracted = await extract_generation_params(last_assistant, "confirmation")
                            if extracted:
                                state = merge_extracted(state, extracted)
                                if is_ready_for_confirmation(state):
                                    rebuilt = build_action_args(state, ref_urls)

                    if rebuilt:
                        function_name, action_args = rebuilt
                        logger.debug("Rebuilt %s action from state", function_name)
                        await self.session_manager.add_message(self.conversation_uuid, "user", message)
                        # Balance gate (a) before saving; gate (b) + atomic claim
                        # happen inside execute_pending_action.
                        blocked = await self._save_pending_or_block(
                            function_name, action_args, last_assistant or ""
                        )
                        if blocked:
                            if state is not None:
                                await self.session_manager.save_workflow_state(self.conversation_uuid, state)
                            await self.session_manager.add_message(self.conversation_uuid, "assistant", blocked)
                            return blocked
                        tool_result, response_text = await self.execute_pending_action()
                        if response_text:
                            if tool_result:
                                # Only flag real generations (not blocks/errors)
                                await self.session_manager.set_just_generated(self.conversation_uuid)
                                await self.session_manager.clear_workflow_state(self.conversation_uuid)
                            await self.session_manager.add_message(self.conversation_uuid, "assistant", response_text)
                            return response_text
                    # else: fall through to Gemini — it will re-ask for whatever
                    # is missing; nothing executes blind.
            
            # Guardar mensaje del usuario en Redis
            await self.session_manager.add_message(
                self.conversation_uuid,
                "user",
                message
            )
            
            # === CHECK FOR AMBIGUITY BEFORE PROCESSING ===
            ref_files = await self.get_reference_files()
            needs_clarif, clarif_key = needs_clarification(message, bool(ref_files))
            if needs_clarif:
                question = self._clarification_text(clarif_key)
                logger.debug(f"Ambiguous request detected, asking for clarification")
                if state is not None:
                    await self.session_manager.save_workflow_state(self.conversation_uuid, state)
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
                    
                    # Skip blob: URLs (browser-only, cannot be fetched from server)
                    if file_url.startswith('blob:'):
                        logger.warning(f"Skipping blob: URL for Gemini analysis (cannot download server-side): {file_url[:80]}")
                        parts.append(f"Note: The user attached a {file_type} but it uses a temporary browser URL (blob:) that cannot be accessed from the server. The {file_type} reference IS stored for generation - the backend will handle it if possible. Proceed with the workflow normally, acknowledging the user has attached a {file_type}.\n\n")
                        continue
                    
                    try:
                        logger.debug(f"Downloading {file_type} from {file_url} for Gemini analysis")
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
                            logger.debug(f"Added {file_type} ({mime_type}) to Gemini message for analysis")
                    except Exception as e:
                        logger.error(f"Failed to download {file_type} for analysis: {e}")
                        parts.append(f"Note: Unable to load {file_type} from URL. Error: {str(e)}\n\n")
            
            # Inject the user's CURRENT token balance so Gemini can advise on
            # affordable models/durations. Per-request only — it is never
            # persisted to Redis history (only `message` is saved), so the
            # balance can't go stale across turns. NOTE: the timeout-retry
            # path below resends the bare `message` and loses this note for
            # that attempt; acceptable.
            balance = get_token_balance()
            if balance is not None:
                parts.append(
                    f"[SYSTEM CONTEXT — current user token balance: {balance} tokens. "
                    f"Use this when recommending models/durations or when the user asks "
                    f"what they can afford. Never invent or guess a different balance. "
                    f"Do not mention the balance unless it is relevant. "
                    f"Do not treat this note as a user message.]\n\n"
                )

            # Make Gemini AWARE of a balance block it "showed" last turn. The
            # block is code-generated and returned directly, so it never entered
            # Gemini's chat session — without this note Gemini would act as if it
            # had never told the user they were short on tokens. Detected from
            # the pre-turn history snapshot (the block IS saved to Redis history).
            last_assistant = next(
                (m.get("content", "") for m in reversed(pre_turn_history)
                 if m.get("role") == "assistant"),
                "",
            )
            # Skip the note once the balance has recovered enough to afford a
            # generation again (e.g. the user topped up between turns) — at that
            # point the workflow proceeds normally and the nudge would be noise.
            min_cost = min_video_cost()
            still_short = balance is None or (min_cost > 0 and balance < min_cost)
            if still_short and is_insufficient_balance_message(last_assistant):
                parts.append(
                    "[SYSTEM CONTEXT — your previous reply told the user they do "
                    "NOT have enough tokens for the requested generation. Nothing "
                    "was generated and no tokens were spent. Continue from there: "
                    "help them choose a cheaper model/shorter duration/lower "
                    "resolution they can afford, or suggest topping up. Do not "
                    "repeat the full balance breakdown unless they ask. Do not "
                    "treat this note as a user message.]\n\n"
                )

            # Inject the explicit workflow state so Gemini trusts it instead of
            # re-deriving the step from 50 history messages. Per-request only —
            # never persisted to Redis history.
            if state is not None:
                parts.append(state_context_note(state) + "\n\n")

            # Add the user's message
            parts.append(message)
            
            # Send to Gemini with timeout and retry logic
            GEMINI_TIMEOUT = 180  # seconds - increased for large system prompt + slow first responses
            MAX_RETRIES = 2  # Retry once on timeout before giving up
            response = None
            
            for attempt in range(MAX_RETRIES):
                try:
                    logger.debug(f"Sending message to Gemini (timeout={GEMINI_TIMEOUT}s, attempt={attempt+1}/{MAX_RETRIES})...")
                    import time
                    start_time = time.time()
                    response = await asyncio.wait_for(
                        self.chat_session.send_message_async(parts) if attempt == 0 else self.chat_session.send_message_async(message),
                        timeout=GEMINI_TIMEOUT
                    )
                    elapsed = time.time() - start_time
                    logger.debug(f"Gemini responded in {elapsed:.1f}s (attempt {attempt+1})")
                    break  # Success - exit retry loop
                except asyncio.TimeoutError:
                    elapsed = time.time() - start_time
                    logger.warning(f"Gemini timed out after {elapsed:.1f}s (attempt {attempt+1}/{MAX_RETRIES})")
                    
                    if attempt < MAX_RETRIES - 1:
                        # Retry - reinitialize chat to clear any stuck state
                        logger.debug("Retrying after timeout...")
                        await self.start_chat()
                        continue
                    
                    # Final attempt failed - try recovery
                    # Check if we have a pending action to execute directly
                    pending_action = await self.get_pending_action()
                    if pending_action:
                        logger.debug(f"Timeout recovery - executing pending action: {pending_action.get('function')}")
                        tool_result, response_text = await self.execute_pending_action()
                        if response_text:
                            await self.session_manager.add_message(
                                self.conversation_uuid,
                                "assistant",
                                response_text
                            )
                            return response_text
                    
                    # No pending action - try sending a lightweight context-aware recovery prompt
                    try:
                        logger.debug("Timeout - sending lightweight recovery prompt to Gemini...")
                        recovery_msg = f"""The user just said: "{message}"
                        
Based on the conversation history, continue the workflow naturally. 
If the user confirmed something (yes/ok/confirm/dale/si), proceed to the NEXT step of the workflow.
If unsure, ask for clarification. Respond in the user's language (default: English). Do NOT call any tools."""
                        
                        recovery_response = await asyncio.wait_for(
                            self.chat_session.send_message_async(recovery_msg),
                            timeout=30
                        )
                        if recovery_response.text:
                            response_text = recovery_response.text
                            await self.session_manager.add_message(
                                self.conversation_uuid,
                                "assistant",
                                response_text
                            )
                            return response_text
                    except Exception as recovery_error:
                        logger.debug(f"Recovery prompt also failed: {recovery_error}")
                    
                    # Last resort fallback
                    clarification = "I'm sorry, there was a temporary issue processing your request. Could you please repeat your last message?"
                    await self.session_manager.add_message(
                        self.conversation_uuid,
                        "assistant",
                        clarification
                    )
                    return clarification
            
            if response is None:
                clarification = "I'm sorry, there was a temporary issue processing your request. Could you please repeat your last message?"
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
                # CONFIRMATION GATE: a generation tool may only run directly
                # when the user's current message confirms a cost we quoted.
                # Anything else gets intercepted into a pending action.
                user_confirmed_cost = await self._user_confirmed_cost_this_turn(message)
                forced_confirmation_text = None
                confirmation_interceptions = 0
                while True:
                    fc = None
                    candidate_parts = None
                    if response.candidates and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if candidate.content and candidate.content.parts:
                            candidate_parts = candidate.content.parts
                    parts_to_check = candidate_parts if candidate_parts is not None else response.parts
                    
                    # Debug: log all parts to understand Gemini's response
                    if parts_to_check:
                        for idx, part in enumerate(parts_to_check):
                            has_fc = hasattr(part, 'function_call')
                            has_text = hasattr(part, 'text') and part.text
                            fc_name = None
                            if has_fc:
                                try:
                                    fc_name = part.function_call.name if part.function_call else None
                                except Exception:
                                    pass
                            logger.debug(f"Part {idx}: has_fc={has_fc}, fc_name={fc_name}, has_text={has_text}")
                    
                    if parts_to_check:
                        for part in parts_to_check:
                            if hasattr(part, "function_call"):
                                try:
                                    fc_candidate = part.function_call
                                    if fc_candidate and hasattr(fc_candidate, 'name') and fc_candidate.name:
                                        fc = fc_candidate
                                        break
                                except Exception as e:
                                    logger.debug(f"Error accessing function_call: {e}")

                    if not fc:
                        break

                    func_name = fc.name
                    func_args = dict(fc.args)
                    # State params override/fill Gemini's args — critically, a
                    # JSON prompt always replaces Gemini's re-serialized copy.
                    func_args = self._merge_state_into_args(func_name, func_args, state)

                    logger.debug(f"Handling function call: {func_name}")

                    # === CONFIRMATION GATE (deterministic) ===
                    # Gemini tried to generate without the user confirming the
                    # cost first. Save the action as pending and force a cost
                    # confirmation question instead — no tokens are spent.
                    if func_name in GENERATION_TOOL_NAMES and not user_confirmed_cost:
                        confirmation_interceptions += 1
                        gate_cost = estimate_generation_cost(func_name, func_args)
                        gate_lang = self._lang_for(message)
                        logger.warning(
                            "Intercepted unconfirmed %s call (cost=%s) — forcing cost confirmation",
                            func_name, gate_cost,
                        )
                        blocked = await self._save_pending_or_block(func_name, func_args, message)
                        if blocked:
                            # Insufficient balance: answer with the block
                            # message, after closing the dangling call.
                            await self._close_function_call(
                                func_name,
                                "BLOCKED: insufficient token balance. Do not call any tool again.",
                            )
                            if state is not None:
                                await self.session_manager.save_workflow_state(
                                    self.conversation_uuid, state
                                )
                            await self.session_manager.add_message(
                                self.conversation_uuid, "assistant", blocked
                            )
                            return blocked

                        gate_note = (
                            "CONFIRMATION_REQUIRED: the user has NOT confirmed this "
                            "generation yet. Nothing was generated and no tokens were "
                            "spent. "
                            + (f"The exact cost is {gate_cost} tokens. " if gate_cost is not None else "")
                            + "Reply to the user in their language with a short summary "
                            "(model, duration/resolution if applicable) and the cost in "
                            "tokens, then ask them to confirm. Do NOT call any tool "
                            "again until the user confirms."
                        )
                        response = await self._close_function_call(func_name, gate_note)
                        if response is None or confirmation_interceptions >= MAX_CONFIRMATION_INTERCEPTIONS:
                            forced_confirmation_text = self._build_cost_confirmation_text(
                                func_name, func_args, gate_cost, gate_lang
                            )
                            break
                        continue

                    tool_result = "Error: Unknown function"
                    try:
                        if func_name == "generate_image":
                            logger.debug(f"Calling generate_image...")
                            tool_result = await generate_image(**func_args)
                            logger.debug(f"generate_image returned: {tool_result[:200] if tool_result else 'None'}...")
                            # Files are saved inside generate_image tool - no need to extract URLs here
                                    
                        elif func_name == "generate_video":
                            logger.debug(f"Calling generate_video...")
                            tool_result = await generate_video(**func_args)
                            logger.debug(f"generate_video returned: {tool_result[:200] if tool_result else 'None'}...")
                            # Files are saved inside generate_video tool - no need to extract URLs here
                        
                        elif func_name == "generate_speech":
                            logger.debug(f"Calling generate_speech...")
                            tool_result = await generate_speech(**func_args)
                            logger.debug(f"generate_speech returned: {tool_result[:200] if tool_result else 'None'}...")
                            # Files are saved inside generate_speech tool - no need to extract URLs here
                                    
                        else:
                            logger.error(f"Unknown function: {func_name}")
                            tool_result = f"Error: Unknown function '{func_name}'"
                    except Exception as e:
                        import traceback
                        logger.error("Exception executing %s: %s\n%s", func_name, e, traceback.format_exc())
                        tool_result = f"Error executing {func_name}: {str(e)}"
                    
                    last_tool_result = tool_result
                    tool_was_actually_called = True

                    # Mark that a generation just happened (for post-generation reset)
                    await self.session_manager.set_just_generated(self.conversation_uuid)
                    await self.session_manager.clear_workflow_state(self.conversation_uuid)
                    state = None
                    
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
                            timeout=900  # 15min - video generation (Kling) can take 10+ minutes
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"Gemini timed out processing function result")
                        # If we have a tool result AND tool was actually called, return success
                        if tool_was_actually_called and tool_result:
                            response_text = self._generate_contextual_success_message(tool_result, tool_was_called=True, lang=self._lang_for(message))
                            if response_text:
                                await self.session_manager.add_message(
                                    self.conversation_uuid,
                                    "assistant",
                                    response_text
                                )
                                return response_text
                        # No tool was called or result is empty - inform user
                        error_msg = "⚠️ There was a problem processing your request. Please try again."
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

                # Gemini kept insisting on unconfirmed tool calls — answer with
                # our own cost confirmation (the pending action is already saved).
                if forced_confirmation_text:
                    response_text = forced_confirmation_text

                # If still no response, handle based on whether a tool was ACTUALLY called
                if not response_text:
                    logger.debug(f"No text in Gemini response. tool_was_actually_called={tool_was_actually_called}")
                    
                    if tool_was_actually_called and last_tool_result:
                        # Tool WAS called - check if it succeeded or failed
                        tool_result_lower = last_tool_result.lower() if last_tool_result else ""
                        is_error = (
                            tool_result_lower.startswith("error")
                            or last_tool_result.startswith(GENERATION_ERROR_PREFIX)
                        )

                        if is_error:
                            # Tool was called but FAILED - explain why in plain language
                            response_text = await self._explain_generation_error(
                                last_tool_result, self._lang_for(message)
                            )
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
                                    response_text = self._generate_contextual_success_message(last_tool_result, tool_was_called=True, lang=self._lang_for(message))
                            except Exception as e:
                                logger.debug(f"Failed to get followup response: {e}")
                                response_text = self._generate_contextual_success_message(last_tool_result, tool_was_called=True, lang=self._lang_for(message))
                    else:
                        # NO tool was called - NEVER say "Done" or "Your video is ready"
                        # Ask Gemini to produce a real response
                        logger.debug("No tool was called and no text response - recovering")
                        try:
                            recovery_prompt = f"""IMPORTANT: No tool was executed. The previous response had no text.
User message: {message}

You MUST respond to the user directly, in the user's language. Do NOT call any tools.
Do NOT say "Done", "Ready", "Your video/image is ready" or any completion message.
If the user requested an image or video, ask for the missing parameters (model, duration, cost confirmation).
If you cannot determine what the user wants, ask them to clarify.
Keep it brief and helpful."""
                            recovery_response = await self.chat_session.send_message_async(recovery_prompt)
                            
                            # Check if recovery response has a function_call (Gemini insists on calling tool)
                            recovery_fc = None
                            if recovery_response.candidates and len(recovery_response.candidates) > 0:
                                rc = recovery_response.candidates[0]
                                if rc.content and rc.content.parts:
                                    for rpart in rc.content.parts:
                                        if hasattr(rpart, 'function_call'):
                                            try:
                                                rfc = rpart.function_call
                                                if rfc and hasattr(rfc, 'name') and rfc.name:
                                                    recovery_fc = rfc
                                                    break
                                            except Exception:
                                                pass
                            
                            if recovery_fc:
                                # Recovery also wants to call a tool - execute it
                                rfunc_name = recovery_fc.name
                                rfunc_args = self._merge_state_into_args(
                                    rfunc_name, dict(recovery_fc.args), state
                                )
                                logger.debug(f"Recovery found function_call: {rfunc_name}({rfunc_args})")
                                # Balance gate: this path doesn't go through a
                                # pending action, so it must check the fresh
                                # balance itself before spending tokens.
                                rcost = estimate_generation_cost(rfunc_name, rfunc_args)
                                rbalance = get_token_balance()
                                if rfunc_name in GENERATION_TOOL_NAMES and not user_confirmed_cost:
                                    # CONFIRMATION GATE — same rule as the main
                                    # loop: never generate without the user
                                    # confirming the cost first.
                                    logger.warning(
                                        "Intercepted unconfirmed recovery %s call (cost=%s)",
                                        rfunc_name, rcost,
                                    )
                                    rlang = self._lang_for(message)
                                    blocked = await self._save_pending_or_block(
                                        rfunc_name, rfunc_args, message
                                    )
                                    response_text = blocked or self._build_cost_confirmation_text(
                                        rfunc_name, rfunc_args, rcost, rlang
                                    )
                                elif rbalance is not None and rcost is not None and rcost > rbalance:
                                    set_insufficient_block({"required": rcost, "available": rbalance})
                                    response_text = await self._localize_balance_block(
                                        rcost, rbalance, affordable_options(rbalance),
                                        fallback_text=message,
                                    )
                                    logger.warning(
                                        "Blocked recovery tool call '%s': cost=%s > balance=%s",
                                        rfunc_name, rcost, rbalance,
                                    )
                                else:
                                    try:
                                        if rfunc_name == "generate_image":
                                            tool_result = await generate_image(**rfunc_args)
                                        elif rfunc_name == "generate_video":
                                            tool_result = await generate_video(**rfunc_args)
                                        elif rfunc_name == "generate_speech":
                                            tool_result = await generate_speech(**rfunc_args)
                                        else:
                                            tool_result = f"Error: Unknown function '{rfunc_name}'"

                                        tool_was_actually_called = True
                                        last_tool_result = tool_result
                                        await self.session_manager.set_just_generated(self.conversation_uuid)
                                        await self.session_manager.clear_workflow_state(self.conversation_uuid)
                                        state = None
                                        response_text = self._generate_contextual_success_message(tool_result, tool_was_called=True, lang=self._lang_for(message))
                                        logger.debug(f"Recovery tool execution success: {rfunc_name}")
                                    except Exception as te:
                                        logger.error(f"Recovery tool execution failed: {te}")
                                        response_text = f"⚠️ There was a problem: {str(te)}"
                            elif recovery_response.text:
                                response_text = recovery_response.text
                            else:
                                response_text = "⚠️ I couldn't process your request. Could you try again with more details?"
                        except Exception as e:
                            logger.debug(f"Failed to get recovery response: {e}")
                            response_text = "⚠️ I couldn't process your request. Could you try again with more details?"

            except ValueError as e:
                # Handle Gemini safety or malformed content errors
                error_str = str(e)
                if "MALFORMED_FUNCTION_CALL" in error_str:
                    logger.warning(f"Gemini MALFORMED_FUNCTION_CALL: {error_str}")
                    
                    # Try to execute pending action if we have one (user already confirmed)
                    pending_action = await self.get_pending_action()
                    if pending_action:
                        logger.debug(f"MALFORMED recovery - executing pending action: {pending_action.get('function')}")
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
                            response_text = "⚠️ I couldn't process your request. Could you try again with more details?"
                    except Exception as e:
                        logger.debug(f"Failed to get recovery response after MALFORMED_FUNCTION_CALL: {e}")
                        response_text = "⚠️ I couldn't process your request. Could you try again with more details?"
                elif "finish_reason" in error_str or "response.text" in error_str:
                    logger.warning(f"Gemini response error: {error_str}")
                    if tool_was_actually_called and last_tool_result:
                        # Tool was called - show its result
                        result_msg = self._generate_contextual_success_message(last_tool_result, tool_was_called=True, lang=self._lang_for(message))
                        response_text = result_msg if result_msg else str(last_tool_result)
                    else:
                        # No tool was called - ask user to retry
                        response_text = "⚠️ There was a problem processing your request. Please try again."
                else:
                    raise e
            
            # === Detect cost confirmation and save pending action (state-driven) ===
            # Skipped when a tool already ran this turn: a success message that
            # happens to mention "tokens" must never re-create a pending action.
            if response_text and not tool_was_actually_called and COST_CONFIRMATION_PATTERN.search(response_text):
                ref_urls = await self._get_image_ref_urls()
                built = (
                    build_action_args(state, ref_urls)
                    if is_ready_for_confirmation(state) else None
                )
                if built is None:
                    # State incomplete (e.g. conversation predates the state
                    # machine) → one-shot LLM extraction over Gemini's own
                    # confirmation message. Failure ⇒ {} ⇒ nothing saved.
                    extracted = await extract_generation_params(response_text, "confirmation")
                    if extracted:
                        state = merge_extracted(state, extracted)
                        if is_ready_for_confirmation(state):
                            built = build_action_args(state, ref_urls)
                if built:
                    function_name, action_args = built
                    logger.debug("Saving pending %s action from state: %s", function_name, action_args)
                    blocked = await self._save_pending_or_block(
                        function_name, action_args, response_text
                    )
                    if blocked:
                        response_text = blocked

            # Capture a refined-prompt proposal (✨ block) and persist the state
            if state is not None:
                if response_text and "✨" in response_text:
                    state = capture_refined_prompt(state, response_text)
                await self.session_manager.save_workflow_state(self.conversation_uuid, state)

            # Guardar respuesta del asistente en Redis
            await self.session_manager.add_message(
                self.conversation_uuid,
                "assistant",
                response_text
            )

            return response_text
            
        except Exception as e:
            # Log the technical detail server-side; show the user a localized,
            # generic failure message (never a raw exception in a fixed language).
            logger.error("send_message failed: %s", e, exc_info=True)
            error_msg = fallback_error_message("unknown", self._lang_for(message))
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
    
    def _generate_contextual_success_message(self, tool_result: str, tool_was_called: bool = False, lang: str = "en") -> str:
        """Generate a contextual success message based on tool result.
        
        CRITICAL: Only returns success messages if tool_was_called is True AND
        the tool_result indicates actual success (contains 'exitosamente' / 'successfully').
        Otherwise returns an informative error/status message.
        """
        if not tool_result or not tool_was_called:
            # No tool was executed - NEVER return a success message
            return None

        tool_result_lower = tool_result.lower()

        # Accepted-but-still-processing (HTTP 202): not an error and not a
        # finished result — tell the user it's on the way and they'll be notified.
        if is_generation_processing(tool_result):
            return processing_message(lang, generation_processing_type(tool_result))

        # Structured generator error: return the friendly per-category message
        # instead of dumping the technical string on the user.
        if tool_result.startswith(GENERATION_ERROR_PREFIX):
            parsed = parse_generation_error(tool_result) or {"category": "unknown"}
            return fallback_error_message(parsed["category"], lang)

        # Check if the tool result indicates an actual error. Return the
        # localized generic failure message (raw technical detail is logged
        # server-side) instead of an English-only prefix + raw dump.
        if tool_result_lower.startswith("error"):
            logger.warning("Unstructured tool error surfaced to user: %s", tool_result[:300])
            return fallback_error_message("unknown", lang)
        
        # Only return success if the tool result confirms success
        is_success = any(word in tool_result_lower for word in [
            'exitosamente', 'successfully', 'generado', 'generated', 'generada'
        ])

        if not is_success:
            # Tool returned but result is unclear - return the raw result
            return tool_result

        # Success detected. This function is only reached on terminal paths
        # (confirmed pending actions, or when the chat model is unavailable), so
        # its return value is shown DIRECTLY to the user — never re-localized by
        # the LLM. Return a message in the conversation language instead of the
        # internal English success signal from tools.py.
        if 'video' in tool_result_lower:
            gen_type = 'video'
        elif 'audio' in tool_result_lower:
            gen_type = 'audio'
        else:
            gen_type = 'image'
        return success_message(lang, gen_type)


# TTL-based LRU cache: max 200 concurrent sessions, evict after 30 minutes of inactivity.
# Accessing a key resets the TTL, so active sessions are never evicted mid-conversation.
_chatbot_cache: TTLCache = TTLCache(maxsize=200, ttl=1800)


def get_chatbot(conversation_uuid: str = "default") -> GeminiChatbot:
    """Get or create a chatbot instance for a specific conversation."""
    if conversation_uuid not in _chatbot_cache:
        _chatbot_cache[conversation_uuid] = GeminiChatbot(conversation_uuid=conversation_uuid)
        logger.debug("Created new GeminiChatbot for uuid='%s'", conversation_uuid)
    return _chatbot_cache[conversation_uuid]
