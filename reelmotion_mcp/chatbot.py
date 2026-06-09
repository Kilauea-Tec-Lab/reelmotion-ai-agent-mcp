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
)
from generation_errors import (
    GENERATION_ERROR_PREFIX,
    parse_generation_error,
    fallback_error_message,
)
from logging_config import setup_logging
from moderation import (
    is_disallowed_content,
    is_disallowed_content_full,
    get_refusal_message,
)

# Load environment variables
load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)

# Pattern matching for confirmations - ONLY words that mean "yes, proceed/execute"
CONFIRMATION_PATTERNS = re.compile(
    r'^(?:'
    # Multi-word patterns (more specific, checked first)
    r'ok dale|si dale|yes\s+please|si\s+por\s+favor|sí\s+por\s+favor|'
    r'go ahead|lets go|let\'s go|do it|'
    r'me parece bien|suena bien|sounds good|that works|'
    r'that\'s good|that\'s fine|'
    r'está bien|esta bien|de acuerdo|'
    r'así está bien|'
    # Single/short word confirmations - ONLY execution confirmations
    r'ok|okey|okay|yes|sure|yep|yeah|go|agreed|confirm|accept|'
    r'proceed|approve|approved|'
    r'si|sí|dale|confirmo|confirmar|procede|hazlo|adelante|claro|afirmativo|'
    r'correcto|exacto|va|venga|vamos|hecho|'
    r'acepto|apruebo|aprobado|anda|órale|sale|'
    r'y|s|1|👍|✅'
    r')[\s.,!?]*$',
    re.IGNORECASE
)

# Pattern matching for REACTIONS to results - these are NOT confirmations
# These should NEVER trigger tool re-execution
REACTION_PATTERNS = re.compile(
    r'^(?:'
    # English reactions
    r'perfect|amazing|awesome|cool|nice|great|love it|like it|looks? great|'
    r'looks? good|looks? amazing|looks? awesome|looks? perfect|'
    r'beautiful|wonderful|excellent|fantastic|incredible|brilliant|'
    r'stunning|gorgeous|impressive|magnificent|superb|lovely|'
    r'wow|omg|oh my god|so good|so cool|so nice|too good|'
    r'thats? perfect|thats? great|thats? amazing|thats? awesome|'
    r'thats? beautiful|thats? wonderful|thats? incredible|'
    r'i love it|i like it|love this|like this|'
    r'well done|good job|great job|nice work|'
    r'thank you|thanks|thx|ty|thank u|'
    # Spanish reactions
    r'perfecto|perfecta|genial|excelente|fantástico|fantástica|'
    r'increíble|increible|hermoso|hermosa|precioso|preciosa|'
    r'espectacular|maravilloso|maravillosa|impresionante|'
    r'bellísimo|bellísima|bellisimo|bellisima|'
    r'qué lindo|que lindo|qué bonito|que bonito|qué bello|que bello|'
    r'qué bien|que bien|qué chido|que chido|qué chévere|que chevere|'
    r'me gusta|me encanta|me fascina|me parece genial|'
    r'está genial|esta genial|está increíble|esta increible|'
    r'quedó genial|quedo genial|quedó increíble|quedo increible|'
    r'quedó perfecto|quedo perfecto|quedó bien|quedo bien|'
    r'listo|bueno|bien|done|ready|'
    r'gracias|muchas gracias|'
    # Reactions with punctuation emphasis
    r'wow+|woo+|yay|siii+|sí+'
    r')[\s.,!?]*$',
    re.IGNORECASE
)

# Pattern to detect when Gemini is asking for confirmation (cost question)
COST_CONFIRMATION_PATTERN = re.compile(
    r'(?:costará|cost|costar[aá]n|tokens?|créditos?|credits?|¿confirmas?|confirm|proceder|proceed)',
    re.IGNORECASE
)

# Patterns to detect video model in conversation.
# NOTE: order matters — more specific variants (e.g. *-fast, *-pro) are listed
# BEFORE the base model so detection loops that break on first match pick them.
VIDEO_MODEL_PATTERNS = {
    'seedance-2.0-fast': re.compile(r'\bseedance[-\s]?2(?:\.0)?[-\s]?fast\b', re.IGNORECASE),
    'seedance-2.0': re.compile(r'\bseedance[-\s]?2(?:\.0)?(?![\d.])(?![-\s]?fast)', re.IGNORECASE),
    'veo-3.1-flash': re.compile(r'\bveo[-\s]?3\.?1[-\s]?flash\b', re.IGNORECASE),
    'veo-3.1-ultra': re.compile(r'\bveo[-\s]?3\.?1[-\s]?ultra\b', re.IGNORECASE),
    'veo-3.1': re.compile(r'\bveo[-\s]?3\.?1(?!\s*(?:flash|ultra))\b', re.IGNORECASE),
    'runway-aleph': re.compile(r'\brunway[-\s]?aleph\b', re.IGNORECASE),
    'runway-4.5': re.compile(r'\brunway[-\s]?4\.?5\b', re.IGNORECASE),
    'kling-v3-omni-pro': re.compile(r'\bkling[-\s]?v?3[-\s]?omni[-\s]?pro\b', re.IGNORECASE),
    'kling-v3-omni-std': re.compile(r'\bkling[-\s]?v?3[-\s]?omni[-\s]?std\b', re.IGNORECASE),
}

# Pattern to extract a chosen resolution (Seedance pricing depends on it)
RESOLUTION_PATTERN = re.compile(r'\b(480p|720p|1080p)\b', re.IGNORECASE)

# Patterns to extract duration
DURATION_PATTERN = re.compile(r'(\d+)\s*(?:segundos?|seconds?|sec|s\b)', re.IGNORECASE)

# Patterns to detect image model in conversation
IMAGE_MODEL_PATTERNS = {
    'Nano Banana 2': re.compile(r'\bnano[-\s]?banana(?:\s*2)?\b', re.IGNORECASE),
    'Freepik': re.compile(r'\bfreepik\b', re.IGNORECASE),
    'GPT': re.compile(r'\bgpt\b', re.IGNORECASE),
}

# Pattern to detect validation/acceptance of a refined prompt
REFINED_PROMPT_ACCEPTANCE_PATTERN = re.compile(
    r'(?:me gusta|i like|prefiero|prefer|usa|use|utiliza|usar)\s+(?:ese|esa|el|la|that|this)\s+(?:prompt|descripci[óo]n|versión|version|text)',
    re.IGNORECASE
)

def is_confirmation(message: str) -> bool:
    """Check if the message is a simple confirmation (intent to execute/proceed)."""
    cleaned = message.strip().lower()
    return bool(CONFIRMATION_PATTERNS.match(cleaned))

def is_reaction(message: str) -> bool:
    """Check if the message is a reaction/compliment to a result (NOT a confirmation to execute)."""
    cleaned = message.strip().lower()
    return bool(REACTION_PATTERNS.match(cleaned))

def needs_clarification(message: str, has_ref_files: bool) -> tuple[bool, str]:
    """
    Check if a message is ambiguous and needs clarification.
    Returns (needs_clarification, suggested_question)
    """
    msg_lower = message.lower().strip()
    
    # === CLEAR INTENT: VIDEO MODEL MENTIONED ===
    video_model_keywords = ['veo', 'kling', 'runway', 'haiper', 'minimax', 'aleph', 'seedance']
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

def _is_model_listing_message(content: str) -> bool:
    """Check if an assistant message is a model listing (shows available models).
    Handles multiple markdown formats: '- Model', '* Model', '**Model**', '* **Model**', etc.
    """
    content_lower = content.lower()
    model_keywords = ['runway', 'veo', 'kling']
    # Count how many distinct model names appear in the message
    model_count = sum(1 for kw in model_keywords if kw in content_lower)
    # If 3+ model names appear, it's very likely a listing (not a single model mention)
    if model_count >= 3:
        return True
    # Also detect by bullet-point patterns with model names
    bullet_patterns = [
        r'[\*\-]\s+\*{0,2}(?:Runway|Veo|Kling)',  # * Model, - Model, * **Model**, - **Model**
        r'[\*\-]\s+\*{0,2}(?:GPT|Nano Banana 2|Freepik)',  # Image model listings
    ]
    bullet_matches = sum(1 for p in bullet_patterns if re.search(p, content))
    if bullet_matches >= 1 and model_count >= 2:
        return True
    return False


def detect_video_params_from_history(history: list) -> dict:
    """
    Try to extract video generation parameters from conversation history.
    Returns dict with 'model', 'duration', 'prompt' if found.
    """
    params = {}
    
    # Search recent messages (last 14) for model and duration
    recent = history[-14:] if len(history) > 14 else history
    
    # Priority 0: Check the MOST RECENT assistant cost/confirmation message for the model
    # This is the MOST AUTHORITATIVE source as it reflects the final agreed-upon model
    # (e.g., when user changes model mid-conversation like "make it std")
    for msg in reversed(recent):
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            content_lower = content.lower()
            # Only check messages that look like cost confirmations (not model listings)
            is_cost_confirm = any(word in content_lower for word in ['cost', 'costará', '×', 'tokens/se'])
            has_model_listing = _is_model_listing_message(content)
            if is_cost_confirm and not has_model_listing:
                for model_name, pattern in VIDEO_MODEL_PATTERNS.items():
                    if pattern.search(content):
                        params['model'] = model_name
                        break
            if 'model' in params:
                break
            break  # Only check the MOST RECENT assistant message
    
    # Priority 1: Detect model from MOST RECENT USER messages
    # This prevents assistant messages (which list ALL models) from matching the wrong one
    if 'model' not in params:
        for msg in reversed(recent):
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                for model_name, pattern in VIDEO_MODEL_PATTERNS.items():
                    if pattern.search(content):
                        params['model'] = model_name
                        break
                if 'model' in params:
                    break
    
    # Priority 2: If no model found in user messages, check assistant's CONFIRMATION messages
    # (e.g., "You've selected Kling V3 Omni Std") but NOT model listing messages
    if 'model' not in params:
        for msg in reversed(recent):
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                # Only check confirmation-style messages, not model listings
                confirmation_phrases = ['selected', 'chosen', 'elegido', 'seleccionado', 'usaremos', 'you\'ve chosen', 'using the', 'you want']
                if any(phrase in content.lower() for phrase in confirmation_phrases):
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
            # Also matches abbreviated forms: sec, seg, secs, segs, s
            cost_match = re.search(r'[×x\*]\s*(\d+)\s*(?:segundos?|seconds?|secs?|segs?|s\b)', content, re.IGNORECASE)
            if cost_match:
                params['duration'] = int(cost_match.group(1))
                break
            # Priority 1.5: Direct "Duration: X seconds" or "Duración: X segundos" from cost confirmation
            duration_label_match = re.search(r'(?:duration|duración)\s*:\s*(\d+)\s*(?:segundos?|seconds?|secs?|segs?|s\b)', content, re.IGNORECASE)
            if duration_label_match:
                params['duration'] = int(duration_label_match.group(1))
                break
            # Priority 2: Look for "video de X segundos" pattern
            video_dur_match = re.search(r'(?:video\s+de|duración\s+de?)\s*(\d+)\s*(?:segundos?|seconds?|secs?|segs?|s\b)', content, re.IGNORECASE)
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
    
    # === PRIORITY 0 for prompt: Extract from cost confirmation message (most reliable) ===
    # The cost confirmation message has format: "Prompt: [actual prompt]\nModel: ...\nDuration: ..."
    recent_reversed = list(reversed(recent))
    for msg in recent_reversed:
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            content_lower = content.lower()
            # Check if this is a cost confirmation message (has cost/tokens AND confirm)
            is_cost_msg = bool(re.search(r'(?:cost|costo|costará|tokens?).*(?:confirm|¿confirma)', content_lower, re.DOTALL))
            if not is_cost_msg:
                is_cost_msg = bool(re.search(r'(?:confirm|¿confirma).*(?:cost|costo|costará|tokens?)', content_lower, re.DOTALL))
            if is_cost_msg:
                # Extract prompt from "Prompt: ..." line
                prompt_match = re.search(r'(?:prompt|descripción)\s*:\s*(.+?)(?=\n(?:model|modelo|duration|duración|cost|costo)|$)', content, re.IGNORECASE | re.DOTALL)
                if prompt_match:
                    extracted = prompt_match.group(1).strip()
                    # Remove surrounding quotes if present
                    extracted = re.sub(r'^["\u201c]|["\u201d]$', '', extracted).strip()
                    
                    # Detect truncation (prevents using "..." summaries as actual prompts)
                    if extracted.endswith('...') or extracted.endswith('…'):
                        logger.debug(f"Prompt in cost confirmation is truncated: {extracted[:80]}... SKIPPING to avoid cut-off prompt.")
                        extracted = None
                    
                    if extracted and len(extracted) > 10:
                        params['prompt'] = extracted
                        logger.debug(f"Extracted prompt from cost confirmation: {extracted[:80]}...")
                break  # Only check the most recent assistant message
    
    # Get the prompt (most recent user message that's not a confirmation or model/duration selection)
    if 'prompt' not in params:
      for i, msg in enumerate(recent_reversed):
        if msg.get('role') == 'user':
            content = msg.get('content', '')
            
            # --- INTELLIGENT CONTEXTUAL ACCEPTANCE CHECK ---
            # Check if this message is accepting/approving a refined prompt proposed by the assistant
            # This handles explicit patterns ("use that prompt") AND contextual ones ("sure", "looks good", "ok")
            
            # 1. Is explicit pattern?
            is_explicit = REFINED_PROMPT_ACCEPTANCE_PATTERN.search(content)
            
            # 2. Is contextual acceptance? (Assistant asked to approve/refine + User says something short/positive)
            is_contextual = False
            # Exclude model selections and duration-only messages from contextual acceptance
            # Match both standalone model names AND with preamble: "lets do Veo 3.1", "use runway aleph", "go with seedance 2.0"
            has_video_model_ref = any(p.search(content) for p in VIDEO_MODEL_PATTERNS.values())
            is_video_model_selection = bool(re.match(r'^\s*(?:(?:lets?\s+(?:do|go\s+with|use)|(?:go|use|i\'?ll?\s+(?:go|do|use))\s+(?:with\s+)?|i\s+want\s+|quiero\s+|usa\s+|vamos\s+(?:con\s+)?)\s*)?(?:seedance[-\s]?2(?:\.0)?[-\s]?(?:fast)?|veo[-\s]?3\.?1[-\s]?(?:flash|ultra)?|kling[-\s]?(?:v?3[-\s]?omni[-\s]?(?:pro|std))?|runway[-\s]?(?:aleph|4\.?5)?)\s*[.,!?]*$', content, re.IGNORECASE))
            is_duration_selection = re.match(r'^\s*\d+\s*(?:segundos?|seconds?|seg|sec|s)?\s*[.,!?]*$', content, re.IGNORECASE)
            # If message contains a video model name, it's a model selection, NOT prompt acceptance
            if not is_video_model_selection and not is_duration_selection and not has_video_model_ref:
                if i + 1 < len(recent_reversed):
                    prev_msg = recent_reversed[i+1]
                    if prev_msg.get('role') == 'assistant':
                        prev_text = prev_msg.get('content', '').lower()
                        # Did assistant propose something? (covers both English and Spanish)
                        # BUT NOT model listings (which also contain "suggest")
                        is_prev_model_listing = _is_model_listing_message(prev_msg.get('content', ''))
                        if not is_prev_model_listing:
                            proposal_phrases = [
                                'refined prompt', 'prompt refinado', 'improved prompt',
                                'approve', 'te parece', 'do you like', 'how about',
                                'te gusta', 'prefieres', 'refinar', 'mejorar',
                                'podríamos', 'algo como', 'something like',
                                'sugiero', 'suggest', 'te gustaría', 'would you like',
                                'aquí tienes', 'here is a', 'quieres que',
                                'usar el tuyo', 'use your', 'tu original'
                            ]
                            if any(x in prev_text for x in proposal_phrases):
                                 # Only SHORT messages (<=5 words) qualify as contextual acceptance
                                 # Longer messages are likely descriptive prompts, not acceptance
                                 if len(content.split()) <= 5:
                                     # Check for negative words or change requests
                                     if not any(neg in content.lower() for neg in ['no', 'bad', 'wrong', 'mal', 'incorrect', 'change', 'cambia', 'don\'t', 'not']):
                                         is_contextual = True

            if is_explicit or is_contextual:
                # Search previous ASSISTANT messages for the refined prompt
                for j in range(i + 1, len(recent_reversed)):
                    prev_msg = recent_reversed[j]
                    if prev_msg.get('role') == 'assistant':
                         cand = prev_msg.get('content', '')
                         # Skip cost confirmations and model listings - keep searching deeper
                         if COST_CONFIRMATION_PATTERN.search(cand) and len(cand) < 80:
                             continue
                         if _is_model_listing_message(cand):
                             continue
                         # Priority 1: Text between quotes (refined prompts are usually quoted)
                         quote_match = re.search(r'["\u201c]([^"\u201d]{15,})["\u201d]', cand)
                         if quote_match:
                             params['prompt'] = quote_match.group(1)
                         else:
                             # Priority 2: Cleaning heuristics if not quoted
                             cleaned = re.sub(r'^(?:Here is|Aquí tienes|Esta es|Propuesta|Aquí hay).*:[\r\n\s]*', '', cand, flags=re.IGNORECASE)
                             cleaned = re.sub(r'[\r\n\s]*(?:Do you like|Te gusta|Te parece|Qué te parece|¿|Confirmas).*$', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
                             cleaned = cleaned.strip()
                             if len(cleaned) > 10:
                                params['prompt'] = cleaned
                         break
                if 'prompt' in params:
                    break
            
            # Skip confirmation messages - but check for refined prompt in preceding assistant message
            if is_confirmation(content):
                # When user confirms/approves, the preceding assistant message may contain the refined prompt
                for j in range(i + 1, len(recent_reversed)):
                    prev_msg = recent_reversed[j]
                    if prev_msg.get('role') == 'assistant':
                        cand = prev_msg.get('content', '')
                        # Skip cost confirmations and model listings - search deeper
                        if COST_CONFIRMATION_PATTERN.search(cand) and len(cand) < 80:
                            continue
                        if _is_model_listing_message(cand):
                            continue
                        # Try to extract quoted prompt
                        quote_match = re.search(r'["\u201c]([^"\u201d]{15,})["\u201d]', cand)
                        if quote_match:
                            params['prompt'] = quote_match.group(1)
                        break
                if 'prompt' in params:
                    break
                continue
            # Skip STANDALONE model selection (just "veo 3.1" alone, "kling v3 omni pro", "lets do seedance 2.0", etc.)
            if re.match(r'^\s*(?:(?:lets?\s+(?:do|go\s+with|use)|(?:go|use|i\'?ll?\s+(?:go|do|use))\s+(?:with\s+)?|i\s+want\s+|quiero\s+|usa\s+|vamos\s+(?:con\s+)?)\s*)?(?:seedance[-\s]?2(?:\.0)?[-\s]?(?:fast)?|veo[-\s]?3\.?1[-\s]?(?:flash|ultra)?|kling[-\s]?(?:v?3[-\s]?omni[-\s]?(?:pro|std))?|runway[-\s]?(?:aleph|4\.?5)?|haiper|minimax)\s*[.,!?]*$', content, re.IGNORECASE):
                continue
            # Skip duration-only messages like "5 seconds", "5s", or just "4"
            if re.match(r'^\s*\d+\s*(?:segundos?|seconds?|seg|sec|s)?\s*[\.!?]*$', content, re.IGNORECASE):
                continue
            # Skip generic video/image creation messages (too vague to be a prompt)
            if re.match(r'^\s*(?:i want to |quiero |me gustaría )?(?:create|make|genera[rt]?|crea[rt]?|haz(?:me)?|anima[rt]?|animate)\s+(?:a\s+|un\s+|una\s+)?(?:video|vídeo|imagen|image|clip)\s*[.,!?]*$', content, re.IGNORECASE):
                continue
            # Skip creation COMMANDS that include model names or durations (these are instructions, NOT descriptive prompts)
            # e.g., "Genera un video de esta imagen con veo 3.1 fast de 4s" or "Create a video with veo 3.1 ultra 8 seconds"
            if re.search(r'(?:crea[rt]?|genera[rt]?|make|create|haz(?:me)?|anima[rt]?|animate)\s+.*(?:video|vídeo|imagen|image|clip)', content, re.IGNORECASE):
                has_video_model = any(p.search(content) for p in VIDEO_MODEL_PATTERNS.values())
                has_image_model = any(p.search(content) for p in IMAGE_MODEL_PATTERNS.values())
                has_duration = bool(DURATION_PATTERN.search(content))
                if has_video_model or has_image_model or has_duration:
                    continue
            # Skip cost-related questions (not a prompt)
            if re.match(r'^.*(?:cost|token|cuánto|cuanto|precio|price|how much).*$', content, re.IGNORECASE) and len(content) < 60:
                continue
            # Skip language-change requests (not descriptive prompts)
            if re.search(r'\b(?:speak|talk|habla|responde)\s+(?:in\s+)?(?:english|español|spanish|inglés)\b', content, re.IGNORECASE) and len(content) < 60:
                continue
            # Skip model/parameter change requests (not descriptive prompts)
            if len(content) < 80 and re.search(r'\b(?:make\s+it|change\s+(?:it\s+)?to|switch\s+to|cambia|hazlo)\b', content, re.IGNORECASE) and re.search(r'\b(?:std|pro|flash|ultra|aleph|standard)\b', content, re.IGNORECASE):
                continue
            # Skip questions (not descriptive prompts)
            if content.strip().endswith('?') and len(content) < 60:
                continue
            # Skip very short messages that are likely answers to questions, not prompts
            if len(content) <= 3:
                continue
            # This is likely the actual prompt - USE IT AS IS, don't modify it
            params['prompt'] = content
            break
    
      # Fallback: If no prompt found from user messages, look in assistant messages for quoted text
      # (refined prompts are usually shown between quotes in assistant responses)
      if 'prompt' not in params:
        for msg in reversed(recent):
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                # Don't extract from model listing messages
                if _is_model_listing_message(content):
                    continue
                quote_match = re.search(r'["\u201c]([^"\u201d]{15,})["\u201d]', content)
                if quote_match:
                    params['prompt'] = quote_match.group(1)
                    break

    # Detect resolution (Seedance pricing depends on it). Prefer a labelled
    # "Resolution: X" line in an assistant message, then any 480p/720p/1080p token.
    for msg in recent_reversed:
        content = msg.get('content', '')
        res_label = re.search(r'(?:resolution|resoluci[\u00f3o]n)\s*:\s*(480p|720p|1080p)', content, re.IGNORECASE)
        if res_label:
            params['resolution'] = res_label.group(1).lower()
            break
    if 'resolution' not in params:
        for msg in recent_reversed:
            res_match = RESOLUTION_PATTERN.search(msg.get('content', ''))
            if res_match:
                params['resolution'] = res_match.group(1).lower()
                break

    return params

# Voice name to voice_id mapping for speech detection
VOICE_NAME_TO_ID = {
    'adam': 'pNInz6obpgDQGcFmaJgB',
    'alice': 'Xb7hH8MSUJpSbSDYk0k2',
    'antoni': 'ErXwobaYiN019PkySvjV',
    'bill': 'pqHfZKP75CvOlQylNhV4',
    'brian': 'nPczCjzI2devNBz1zQrb',
    'callum': 'N2lVS1w4EtoT3dr4eOWO',
    'charlie': 'IKne3meq5aSn9XLyUdCD',
    'chris': 'iP95p4xoKVk53GoZ742B',
    'daniel': 'onwK4e9ZLuTAKqWW03F9',
    'domi': 'AZnzlk1XvdvUeBnXmlld',
    'elli': 'MF3mGyEYCl7XYWbV9V6O',
    'eric': 'cjVigY5qzO86Huf0OWal',
    'george': 'JBFqnCBsd6RMkjVDRZzb',
    'harry': 'SOYHLrjzK2X1ezoPC6cr',
    'jessica': 'cgSgspJ2msm6clMCkdW9',
    'josh': 'TxGEqnHWrfWFTfGW9XjX',
    'laura': 'FGY2WhTYpPnrIDTdsKH5',
    'liam': 'TX3LPaxmHKxFdv7VOQHJ',
    'lily': 'pFZP5JQG7iQjIQuC4Bku',
    'matilda': 'XrExE9yKIg1WjnnlVkGX',
    'rachel': '21m00Tcm4TlvDq8ikWAM',
    'river': 'SAz9YHcvj6GT2YYXdXww',
    'roger': 'CwhRBWXzGAHq8TQ4Fs17',
    'sarah': 'EXAVITQu4vr4xnSDxMaL',
    'will': 'bIHbv24MWmeRgasZH58o',
}

def detect_speech_params_from_history(history: list) -> dict:
    """
    Try to extract speech generation parameters from conversation history.
    Returns dict with 'text', 'voice_id' if found.
    """
    params = {}
    
    recent = history[-14:] if len(history) > 14 else history
    recent_reversed = list(reversed(recent))
    
    # Detect voice from user messages (e.g., "lets go with Rachel", "Rachel", "Adam")
    for msg in recent_reversed:
        if msg.get('role') == 'user':
            content = msg.get('content', '').strip().lower()
            # Skip confirmations and very short messages
            if is_confirmation(content) and len(content) < 10:
                continue
            for voice_name, voice_id in VOICE_NAME_TO_ID.items():
                if voice_name in content:
                    params['voice_id'] = voice_id
                    break
            if 'voice_id' in params:
                break
    
    # If no voice found in user messages, check assistant confirmation messages
    if 'voice_id' not in params:
        for msg in recent_reversed:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                content_lower = content.lower()
                # Only check confirmation-style messages (Voice: Rachel)
                voice_match = re.search(r'(?:voice|voz)\s*:\s*(\w+)', content, re.IGNORECASE)
                if voice_match:
                    voice_name = voice_match.group(1).strip().lower()
                    if voice_name in VOICE_NAME_TO_ID:
                        params['voice_id'] = VOICE_NAME_TO_ID[voice_name]
                        break
    
    # Default to Rachel if no voice detected
    if 'voice_id' not in params:
        params['voice_id'] = '21m00Tcm4TlvDq8ikWAM'
    
    # Detect speech text - look for quoted text in assistant messages (the confirmed text)
    for msg in recent_reversed:
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            # Look for the confirmed text pattern: Text: "..."
            text_match = re.search(r'(?:text|texto)\s*:\s*["\u201c]([^"\u201d]{5,})["\u201d]', content, re.IGNORECASE)
            if text_match:
                params['text'] = text_match.group(1)
                break
            # Fallback: look for quoted text that looks like user's speech text
            quote_match = re.search(r'["\u201c]([^"\u201d]{15,})["\u201d]', content)
            if quote_match:
                params['text'] = quote_match.group(1)
                break
    
    # If no text from assistant, get from user messages (the original speech text)
    if 'text' not in params:
        for msg in recent_reversed:
            if msg.get('role') == 'user':
                content = msg.get('content', '').strip()
                # Skip short messages (confirmations, voice selections)
                if len(content) > 20 and not is_confirmation(content):
                    # Skip messages that are just voice names
                    content_lower = content.lower()
                    if not any(content_lower.strip().startswith(v) for v in VOICE_NAME_TO_ID.keys()):
                        params['text'] = content
                        break
    
    return params


def _extract_prompt_from_confirmation(confirmation_text: str) -> str:
    """
    Extract the exact prompt from a cost confirmation message.
    This ensures the prompt shown to the user is exactly what gets executed.
    
    Matches patterns like:
    - "I'm going to generate: [PROMPT]"
    - "Voy a generar: [PROMPT]"
    - "Prompt: [PROMPT]"
    - "Prompt: \"[PROMPT]\""
    - "Descripción: [PROMPT]"
    """
    if not confirmation_text:
        return None
    
    # Pattern 1: "Prompt:" or "Descripción:" followed by the text (possibly quoted)
    prompt_match = re.search(
        r'(?:prompt|descripci[oó]n)\s*:\s*(?:["\u201c])?(.+?)(?:["\u201d])?\s*(?=\n(?:model|modelo|cost|costo|precio)|$)',
        confirmation_text, re.IGNORECASE | re.DOTALL
    )
    if prompt_match:
        extracted = prompt_match.group(1).strip()
        # Remove surrounding quotes if present
        extracted = re.sub(r'^["\u201c]|["\u201d]$', '', extracted).strip()
        # Remove leading ** from markdown bold
        extracted = re.sub(r'^\*{1,2}\s*', '', extracted).strip()
        extracted = re.sub(r'\s*\*{1,2}$', '', extracted).strip()
        if extracted and len(extracted) > 10 and not extracted.endswith('...') and not extracted.endswith('…'):
            return extracted
    
    # Pattern 2: "I'm going to generate:" / "Voy a generar:" followed by the prompt
    gen_match = re.search(
        r'(?:I\'m going to generate|voy a generar|generaré)\s*:?\s*(?:["\u201c])?(.+?)(?:["\u201d])?\s*(?=\n(?:model|modelo|cost|costo|precio)|$)',
        confirmation_text, re.IGNORECASE | re.DOTALL
    )
    if gen_match:
        extracted = gen_match.group(1).strip()
        extracted = re.sub(r'^["\u201c]|["\u201d]$', '', extracted).strip()
        extracted = re.sub(r'^\*{1,2}\s*', '', extracted).strip()
        extracted = re.sub(r'\s*\*{1,2}$', '', extracted).strip()
        if extracted and len(extracted) > 10 and not extracted.endswith('...') and not extracted.endswith('…'):
            return extracted
    
    # Pattern 3: Look for text between quotes (most common for refined prompts)
    quote_match = re.search(r'["\u201c]([^"\u201d]{15,})["\u201d]', confirmation_text)
    if quote_match:
        extracted = quote_match.group(1).strip()
        if not extracted.endswith('...') and not extracted.endswith('…'):
            return extracted
    
    return None


def _extract_all_params_from_confirmation(text: str) -> dict:
    """
    Extract the action type AND all parameters from a structured cost confirmation message.

    The confirmation always has a known format:
        Prompt: [...]
        Model: [...]
        Duration: [X] seconds   (video only)
        Cost: [Y] tokens
        Do you confirm?

    This avoids scanning 14 messages of history with fragile regex.
    Returns dict with keys: 'type', 'prompt', 'model', 'duration' (as available).
    """
    params = {}
    if not text:
        return params

    text_lower = text.lower()

    # Detect action type from keyword presence
    if any(w in text_lower for w in ["video", "vídeo", "veo", "runway", "kling", "seedance", "tokens/se"]):
        params["type"] = "video"
    elif any(w in text_lower for w in ["speech", "voice", "voz", "audio", "narración", "narration"]):
        params["type"] = "speech"
    elif any(w in text_lower for w in ["imagen", "image", "foto", "picture", "gpt", "nano banana", "freepik"]):
        params["type"] = "image"

    # Extract prompt/text from structured label line
    for label in ["prompt", "descripción", "descripcion", "edit", "texto", "text"]:
        match = re.search(
            rf"(?:{label})\s*:\s*[\"\\u201c]?\*{{0,2}}(.+?)\*{{0,2}}[\"\\u201d]?\s*(?=\n|$)",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if match:
            extracted = match.group(1).strip()
            extracted = re.sub(r'^["\u201c]|["\u201d]$', "", extracted).strip()
            extracted = re.sub(r"^\*{1,2}\s*|\s*\*{1,2}$", "", extracted).strip()
            if extracted and len(extracted) > 5 and not extracted.endswith("...") and not extracted.endswith("\u2026"):
                params["prompt"] = extracted
                break

    # Extract model from "Model: ..." line (sort longest keys first to prevent partial matches)
    model_match = re.search(r"(?:model|modelo)\s*:\s*\*{0,2}(.+?)\*{0,2}\s*(?:\n|$)", text, re.IGNORECASE)
    if model_match:
        raw_model = model_match.group(1).strip().lower()
        model_map = {
            "seedance 2.0 fast": "seedance-2.0-fast",
            "seedance 2 fast": "seedance-2.0-fast",
            "seedance2 fast": "seedance-2.0-fast",
            "seedance-2.0-fast": "seedance-2.0-fast",
            "seedance 2.0": "seedance-2.0",
            "seedance 2": "seedance-2.0",
            "seedance2": "seedance-2.0",
            "seedance-2.0": "seedance-2.0",
            "veo 3.1 ultra": "veo-3.1-ultra",
            "veo-3.1-ultra": "veo-3.1-ultra",
            "veo 3.1 flash": "veo-3.1-flash",
            "veo-3.1-flash": "veo-3.1-flash",
            "veo 3.1": "veo-3.1",
            "veo-3.1": "veo-3.1",
            "runway aleph": "runway-aleph",
            "runway-aleph": "runway-aleph",
            "runway 4.5": "runway-4.5",
            "runway-4.5": "runway-4.5",
            "kling v3 omni pro": "kling-v3-omni-pro",
            "kling-v3-omni-pro": "kling-v3-omni-pro",
            "kling v3 omni std": "kling-v3-omni-std",
            "kling-v3-omni-std": "kling-v3-omni-std",
            "nano banana 2": "Nano Banana 2",
            "nano banana": "Nano Banana 2",
            "gpt": "GPT",
            "freepik": "Freepik",
        }
        for k in sorted(model_map.keys(), key=len, reverse=True):
            if k in raw_model:
                params["model"] = model_map[k]
                break

    # Extract duration: "Duration: X seconds" or from cost "Y tokens/sec × X sec"
    dur_label = re.search(
        r"(?:duration|duración)\s*:\s*(\d+)\s*(?:segundos?|seconds?|sec|s)?",
        text,
        re.IGNORECASE,
    )
    if dur_label:
        params["duration"] = int(dur_label.group(1))
    else:
        cost_match = re.search(r"[×x\*]\s*(\d+)\s*(?:segundos?|seconds?|sec|s)\b", text, re.IGNORECASE)
        if cost_match:
            params["duration"] = int(cost_match.group(1))

    # Extract resolution (Seedance only): "Resolution: X" line, else any token.
    res_label = re.search(r"(?:resolution|resoluci[óo]n)\s*:\s*(480p|720p|1080p)", text, re.IGNORECASE)
    if res_label:
        params["resolution"] = res_label.group(1).lower()
    else:
        res_match = re.search(r"\b(480p|720p|1080p)\b", text, re.IGNORECASE)
        if res_match:
            params["resolution"] = res_match.group(1).lower()

    return params


def detect_image_params_from_history(history: list) -> dict:
    """
    Try to extract image generation parameters from conversation history.
    Returns dict with 'model', 'prompt' if found.
    """
    params = {}
    
    # Search recent messages for model
    recent = history[-14:] if len(history) > 14 else history
    
    # Detect model from MOST RECENT USER messages first (priority to user's explicit selection)
    # This prevents matching 'GPT' from the bot's model listing message
    for msg in reversed(recent):
        if msg.get('role') == 'user':
            content = msg.get('content', '').strip()
            for model_name, pattern in IMAGE_MODEL_PATTERNS.items():
                if pattern.search(content):
                    params['model'] = model_name
                    break
            if 'model' in params:
                break
    
    # If no model found in user messages, check assistant CONFIRMATION messages only
    if 'model' not in params:
        for msg in reversed(recent):
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                # Only check confirmation-style messages, not model listings
                confirmation_phrases = ['chosen', 'selected', 'elegido', 'seleccionado', 'usaremos', 'using', 'usaré']
                if any(phrase in content.lower() for phrase in confirmation_phrases):
                    for model_name, pattern in IMAGE_MODEL_PATTERNS.items():
                        if pattern.search(content):
                            params['model'] = model_name
                            break
                    if 'model' in params:
                        break
    
    # === PRIORITY 0 for prompt: Extract from cost confirmation message (most reliable) ===
    recent_reversed = list(reversed(recent))
    for msg in recent_reversed:
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            content_lower = content.lower()
            is_cost_msg = bool(re.search(r'(?:cost|costo|costar\u00e1|tokens?).*(?:confirm|\u00bfconfirma)', content_lower, re.DOTALL))
            if not is_cost_msg:
                is_cost_msg = bool(re.search(r'(?:confirm|\u00bfconfirma).*(?:cost|costo|costar\u00e1|tokens?)', content_lower, re.DOTALL))
            if is_cost_msg:
                prompt_match = re.search(r'(?:prompt|descripción)\s*:\s*(.+?)(?=\n(?:model|modelo|cost|costo)|$)', content, re.IGNORECASE | re.DOTALL)
                if prompt_match:
                    extracted = prompt_match.group(1).strip()
                    extracted = re.sub(r'^["\u201c]|["\u201d]$', '', extracted).strip()
                    
                    # Detect truncation (prevents using "..." summaries as actual prompts)
                    if extracted.endswith('...') or extracted.endswith('…'):
                        logger.debug(f"Prompt in cost confirmation is truncated: {extracted[:80]}... SKIPPING to avoid cut-off prompt.")
                        extracted = None
                    
                    if extracted and len(extracted) > 10:
                        params['prompt'] = extracted
                        logger.debug(f"Extracted prompt from cost confirmation: {extracted[:80]}...")
                break
    
    # Get the prompt (most recent user message that's not a confirmation or model selection)
    if 'prompt' not in params:
      for i, msg in enumerate(recent_reversed):
        if msg.get('role') == 'user':
            content = msg.get('content', '')
            
            # --- INTELLIGENT CONTEXTUAL ACCEPTANCE CHECK ---
            # Check if this message is accepting/approving a refined prompt proposed by the assistant
            # This handles explicit patterns ("use that prompt") AND contextual ones ("sure", "looks good", "ok")
            
            # 1. Is explicit pattern?
            is_explicit = REFINED_PROMPT_ACCEPTANCE_PATTERN.search(content)
            
            # 2. Is contextual acceptance? (Assistant asked to approve/refine + User says something short/positive)
            is_contextual = False
            # Exclude model selections from contextual acceptance
            # Match both standalone and with preamble: "lets do gpt", "use freepik", "go with nano banana 2"
            has_image_model_ref = any(p.search(content) for p in IMAGE_MODEL_PATTERNS.values())
            is_image_model_selection = bool(re.match(r'^\s*(?:(?:lets?\s+(?:do|go\s+with|use)|(?:go|use|i\'?ll?\s+(?:go|do|use))\s+(?:with\s+)?|i\s+want\s+|quiero\s+|usa\s+|vamos\s+(?:con\s+)?)\s*)?(?:gpt|nano[-\s]?banana(?:\s*2)?|freepik)\s*[.,!?]*$', content, re.IGNORECASE))
            if not is_image_model_selection and not has_image_model_ref:
                if i + 1 < len(recent_reversed):
                    prev_msg = recent_reversed[i+1]
                    if prev_msg.get('role') == 'assistant':
                        prev_text = prev_msg.get('content', '').lower()
                        # Did assistant propose something? (covers both English and Spanish)
                        # BUT NOT model listings (which also contain "suggest")
                        is_prev_model_listing = _is_model_listing_message(prev_msg.get('content', ''))
                        if not is_prev_model_listing:
                            proposal_phrases = [
                                'refined prompt', 'prompt refinado', 'improved prompt',
                                'approve', 'te parece', 'do you like', 'how about',
                                'te gusta', 'prefieres', 'refinar', 'mejorar',
                                'podríamos', 'algo como', 'something like',
                                'sugiero', 'suggest', 'te gustaría', 'would you like',
                                'aquí tienes', 'here is a', 'quieres que',
                                'usar el tuyo', 'use your', 'tu original'
                            ]
                            if any(x in prev_text for x in proposal_phrases):
                             # Only SHORT messages (<=5 words) qualify as contextual acceptance
                             # Longer messages are likely descriptive prompts, not acceptance
                             if len(content.split()) <= 5:
                                 # Check for negative words or change requests
                                 if not any(neg in content.lower() for neg in ['no', 'bad', 'wrong', 'mal', 'incorrect', 'change', 'cambia', 'don\'t', 'not']):
                                     is_contextual = True

            if is_explicit or is_contextual:
                # Search previous ASSISTANT messages for the refined prompt
                for j in range(i + 1, len(recent_reversed)):
                    prev_msg = recent_reversed[j]
                    if prev_msg.get('role') == 'assistant':
                         cand = prev_msg.get('content', '')
                         # Skip cost confirmations and model listings - keep searching deeper
                         if COST_CONFIRMATION_PATTERN.search(cand) and len(cand) < 80:
                             continue
                         if _is_model_listing_message(cand):
                             continue
                         # Priority 1: Text between quotes (refined prompts are usually quoted)
                         quote_match = re.search(r'["\u201c]([^"\u201d]{15,})["\u201d]', cand)
                         if quote_match:
                             params['prompt'] = quote_match.group(1)
                         else:
                             # Priority 2: Cleaning heuristics if not quoted
                             cleaned = re.sub(r'^(?:Here is|Aquí tienes|Esta es|Propuesta|Aquí hay).*:[\r\n\s]*', '', cand, flags=re.IGNORECASE)
                             cleaned = re.sub(r'[\r\n\s]*(?:Do you like|Te gusta|Te parece|Qué te parece|¿|Confirmas).*$', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
                             cleaned = cleaned.strip()
                             if len(cleaned) > 10:
                                params['prompt'] = cleaned
                         break
                if 'prompt' in params:
                    break
            
            # Skip confirmation messages - but check for refined prompt in preceding assistant message
            if is_confirmation(content):
                # When user confirms/approves, the preceding assistant message may contain the refined prompt
                for j in range(i + 1, len(recent_reversed)):
                    prev_msg = recent_reversed[j]
                    if prev_msg.get('role') == 'assistant':
                        cand = prev_msg.get('content', '')
                        # Skip cost confirmations and model listings - search deeper
                        if COST_CONFIRMATION_PATTERN.search(cand) and len(cand) < 80:
                            continue
                        if _is_model_listing_message(cand):
                            continue
                        # Try to extract quoted prompt
                        quote_match = re.search(r'["\u201c]([^"\u201d]{15,})["\u201d]', cand)
                        if quote_match:
                            params['prompt'] = quote_match.group(1)
                        break
                if 'prompt' in params:
                    break
                continue
            # Skip model selection messages (standalone model names)
            if re.match(r'^\s*(?:gpt|nano[-\s]?banana|freepik)\s*[.,!?]*$', content, re.IGNORECASE):
                continue
            # Skip generic "create image" type messages (too vague)
            if re.match(r'^\s*(?:i want to |quiero |me gustaría )?(?:create|make|genera[rt]?|crea[rt]?|haz(?:me)?)\s+(?:a\s+|an\s+|un\s+|una\s+)?(?:image|imagen|picture|foto|photo)s?\s*[.,!?]*$', content, re.IGNORECASE):
                continue
            # Skip creation COMMANDS that include model names or durations (instructions, NOT descriptive prompts)
            # e.g., "Genera una imagen con GPT de una rana" or "Create an image with Freepik"
            if re.search(r'(?:crea[rt]?|genera[rt]?|make|create|haz(?:me)?)\s+.*(?:image|imagen|picture|foto|photo)', content, re.IGNORECASE):
                has_image_model = any(p.search(content) for p in IMAGE_MODEL_PATTERNS.values())
                has_video_model = any(p.search(content) for p in VIDEO_MODEL_PATTERNS.values())
                if has_image_model or has_video_model:
                    continue
            # Skip cost-related questions
            if re.match(r'^.*(?:cost|token|cuánto|cuanto|precio|price|how much).*$', content, re.IGNORECASE) and len(content) < 60:
                continue
            # Skip language-change requests (not descriptive prompts)
            if re.search(r'\b(?:speak|talk|habla|responde)\s+(?:in\s+)?(?:english|español|spanish|inglés)\b', content, re.IGNORECASE) and len(content) < 60:
                continue
            # Skip questions (not descriptive prompts)
            if content.strip().endswith('?') and len(content) < 60:
                continue
            # Skip very short messages that are likely answers to questions, not prompts
            if len(content) <= 3:
                continue
            # This is likely the actual prompt (use as-is)
            if len(content) > 3:
                params['prompt'] = content
                break
    
    # Fallback: If no prompt found from user messages, look in assistant messages for quoted text
    # (refined prompts are usually shown between quotes in assistant responses)
    if 'prompt' not in params:
        for msg in reversed(recent):
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                # Don't extract from model listing messages
                if _is_model_listing_message(content):
                    continue
                quote_match = re.search(r'["\u201c]([^"\u201d]{15,})["\u201d]', content)
                if quote_match:
                    params['prompt'] = quote_match.group(1)
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
        LANGUAGE ADAPTATION (CRITICAL - HIGHEST PRIORITY):
        - DEFAULT LANGUAGE: English. If you have NO prior conversation history or cannot determine the user's language, ALWAYS use English.
        - You MUST respond in the SAME language the user is writing in.
        - Detect the language from EACH user message. Track the LAST CLEARLY IDENTIFIABLE language.
        - If the user writes in English, respond in English. If the user writes in Spanish, respond in Spanish. Same for any other language.
        - AMBIGUOUS MESSAGES: Short replies like "no", "ok", "yes", "gpt", "veo 3.1", "5s", model names, or single words that exist in multiple languages are NOT a language switch. KEEP the last clearly detected language (default: English).
        - LANGUAGE SWITCH: Only change language if the user writes a CLEAR sentence in a different language or explicitly requests it.
        - This applies to ALL messages: questions, confirmations, cost info, errors, EVERYTHING.
        - Keep technical terms and model names in their original form (e.g., "Nano Banana 2", "GPT", "Freepik").
        - IMPORTANT: All instructions below are written in English for clarity, but you MUST always respond to the user in THEIR language (default: English).
        
        ⛔ MANDATORY WORKFLOW - NEVER SKIP STEPS:
        You MUST NEVER call generate_image or generate_video unless ALL workflow steps have been completed.
        Each step MUST happen in a SEPARATE message exchange (user sends message → you respond → user sends next message → you respond).
        You CANNOT complete multiple steps in a single response.
        If ANY step is missing, you MUST ask for it before proceeding.
        
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
             Supported models for video editing: Runway Aleph, Kling V3 Omni Std, Kling V3 Omni Pro.
        
        2. IMAGE GENERATION/EDITING INTENT:
           - "Generate image" = IMAGE workflow
           - "Create an image" = IMAGE workflow
           - "Edit image" + reference image = IMAGE-TO-IMAGE workflow
           - Mentions image models: GPT, Nano Banana 2, Freepik
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
           - Example: "Okay, let's start with Scene 1. We need an image of the hero. Which model do you want to use: Nano Banana 2, GPT, or Freepik?"
        
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
        
        STEP 3 - ASK FOR THE MODEL (WITH SUGGESTION):
        - Based on THE_PROMPT, suggest a model and explain why.
        - ⚠️ ALWAYS present the models as a FORMATTED LIST (one model per line), never as inline text.
        - Available models:
          → GPT (6 tokens): Best for detailed, realistic, complex images. Recommended for most cases.
          → Nano Banana 2 (7 tokens): Great for artistic, stylized, creative images.
          → Freepik (1 token): Good for clean, commercial-style images.
        - Ask: "I suggest using [model] because [reason]. Which model would you like to use?" (in user's language)
        - Wait for the user to choose.
        - Token costs per image: Nano Banana 2 = 7 tokens, GPT = 6 tokens, Freepik = 1 token.
        
        STEP 4 - CONFIRM COST AND EXECUTE:
        - Summarize what will be generated:
          → "I'm going to generate: [brief description of THE_PROMPT]"
          → "Model: [chosen model]"
          → "Cost: [X] tokens" (Nano Banana 2=7, GPT=6, Freepik=1)
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
        4. If there are attached images, always pass them in 'reference_images'.
        5. Available models are: 'Nano Banana 2' (7 tokens), 'GPT' (6 tokens), and 'Freepik' (1 token).
        6. NEVER mention URLs in your responses - images are sent automatically to the user.
        7. IF THERE'S AN ERROR: Inform the user. If user says "try again"/"retry", execute the tool again without hesitation.
        
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
          → Seedance 2.0 Fast (resolution-based: 480p=12, 720p=26 tokens/sec) - 4 to 15 sec - fast & economical (max 720p)
          → Seedance 2.0 (resolution-based: 480p=15, 720p=32, 1080p=72 tokens/sec) - 4 to 15 sec - supports 1080p
          → Runway 4.5 (14 tokens/sec) - 5, 8 or 10 sec - high quality
          → Runway Aleph (17 tokens/sec) - 5 or 10 sec - versatile
          → Veo 3.1 Flash (17 tokens/sec) - 8 sec only - fast and good quality
          → Kling V3 Omni Std (19 tokens/sec) - 3 to 15 sec - text/image-to-video
          → Kling V3 Omni Pro (26 tokens/sec) - 3 to 15 sec - text/image-to-video, better quality
          → Veo 3.1 (44 tokens/sec) - 8 sec only - high quality
          → Veo 3.1 Ultra (65 tokens/sec) - 8 sec only - maximum Veo quality
        - Say: "I suggest [model] because [reason]. Which model would you like to use?" (in user's language)
        - Wait for user to choose model.

        STEP 3.5 - ASK FOR RESOLUTION (ONLY for Seedance 2.0 / Seedance 2.0 Fast):
        - ⚠️ This step applies ONLY when the chosen model is Seedance 2.0 or Seedance 2.0 Fast. For ALL OTHER models, SKIP this step entirely.
        - Seedance pricing depends on the resolution, so you MUST ask for it before quoting the cost.
          → Seedance 2.0: offer 480p, 720p, or 1080p.
          → Seedance 2.0 Fast: offer ONLY 480p or 720p. If the user asks for 1080p, tell them the Fast tier does not support it and it will use 720p (or suggest switching to Seedance 2.0).
        - Ask: "Which resolution? Options: [valid resolutions for the chosen model]" (in user's language)
        - Wait for the user to choose. SAVE as THE_RESOLUTION.

        STEP 4 - ASK FOR DURATION:
        - Based on the chosen model, tell the user the valid durations:
          → Seedance 2.0 / Seedance 2.0 Fast: any whole number from 4 to 15 seconds (default 5)
          → Veo 3.1 / Veo 3.1 Flash / Veo 3.1 Ultra: ONLY 8 seconds (auto-set, just inform)
          → Runway Aleph: 5 or 10 seconds
          → Runway 4.5: 5, 8 or 10 seconds
          → Kling V3 Omni Pro / Std: 3 to 15 seconds
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
          → **Runway Aleph** (17 tokens/sec) - 5 or 10 sec - High quality editing
          → **Kling V3 Omni Std** (19 tokens/sec) - 3 to 15 sec - Flexible duration ⭐ Recommended
          → **Kling V3 Omni Pro** (26 tokens/sec) - 3 to 15 sec - Better quality
        - ⛔ DO NOT show any other models (Veo, Runway 4.5, Seedance, etc.) - they do NOT support video-to-video.
        - Suggest Kling V3 Omni Std as the most economical option.
        - Ask: "Which model would you like to use?" (in user's language)
        - Wait for user to choose. SAVE as THE_MODEL.
        
        STEP 3 - ASK FOR DURATION:
        - Based on THE_MODEL:
          → Kling V3 Omni Std / Pro: 3 to 15 seconds
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
          → ai_model = the chosen model name (exact: 'kling-v3-omni-std', 'kling-v3-omni-pro', or 'runway-aleph')
          → video_duration = THE_DURATION
          → reference_video = the attached video URL
        
        ═══════════════════════════════════════════════════
        
        ADDITIONAL VIDEO RULES (apply to BOTH workflows):
        1. ⚠️ PROMPT PARAMETER RULE: Same as image - NEVER use conversational replies as prompt.
        2. FORBIDDEN to modify the agreed-upon prompt without consent.
        3. When calling the tool, use EXACT model names:
           - 'seedance-2.0', 'seedance-2.0-fast'
           - 'veo-3.1', 'veo-3.1-flash', 'veo-3.1-ultra'
           - 'runway-aleph', 'runway-4.5'
           - 'kling-v3-omni-pro', 'kling-v3-omni-std'
           For Seedance, also pass resolution ('480p'/'720p'/'1080p'). Seedance auto-detects
           the mode: a reference video → reference mode (discounted), an image → image mode, prompt only → text mode.
        4. If there are attached images, use them as reference automatically (image-to-video).
        5. NEVER mention video URLs - they are sent automatically.
        6. IF THERE'S AN ERROR: Inform user. If they say "try again"/"retry", execute again without hesitation.
        7. Reference files are NOT lost after errors - they persist in the session.
        
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
            lang = "es" if is_spanish(action.get("cost_message", "")) else "en"
            return None, fallback_error_message("unknown", lang)

        # === BALANCE GATE (fresh balance from the confirming request) ===
        balance = get_token_balance()
        cost = action.get("estimated_cost")
        if cost is None:
            cost = estimate_generation_cost(function_name, args)
        if balance is not None and cost is not None and cost > balance:
            set_insufficient_block({"required": cost, "available": balance})
            lang = "es" if is_spanish(action.get("cost_message", "")) else "en"
            message = build_insufficient_balance_message(
                cost, balance, lang, affordable_options(balance)
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
                lang = "es" if is_spanish(action.get("cost_message", "")) else "en"
                return None, fallback_error_message("unknown", lang)

            # NOTE: no clear_pending_action() needed — the atomic claim above
            # already removed the action from Redis.

            # Generator failed: explain WHY in plain language (LLM with code fallback)
            if tool_result and (
                tool_result.startswith(GENERATION_ERROR_PREFIX)
                or tool_result.lower().startswith("error")
            ):
                lang = "es" if is_spanish(action.get("cost_message", "")) else "en"
                friendly = await self._explain_generation_error(tool_result, lang)
                # tool_result=None so callers never set the just_generated flag
                return None, friendly

            # Generate success message - tool WAS actually called here
            response_text = self._generate_contextual_success_message(tool_result, tool_was_called=True)
            if not response_text:
                # Fallback if message generation returns None (shouldn't happen for pending actions)
                response_text = tool_result if tool_result else "⚠️ Could not determine the operation result."
            return tool_result, response_text

        except Exception as e:
            import traceback
            logger.error("Failed to execute pending action: %s\n%s", e, traceback.format_exc())
            lang = "es" if is_spanish(action.get("cost_message", "")) else "en"
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
            "top up tokens, or contact support — whichever fits the error). "
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
            lang = "es" if is_spanish(confirmation_text) else "en"
            logger.debug(
                "Blocked saving pending '%s': cost=%d > balance=%d",
                function_name, cost, balance,
            )
            return build_insufficient_balance_message(
                cost, balance, lang, affordable_options(balance)
            )

        await self.save_pending_action(function_name, action_args, confirmation_text)
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
                # Drop any pending generation so a later "yes" cannot execute
                # something that relied on the blocked prompt.
                await self.clear_pending_action()
                return refusal

            if not self.chat_session:
                await self.start_chat()

            # === CHECK: Was something just generated? Reset workflow on reactions ===
            just_generated = await self.session_manager.get_just_generated(self.conversation_uuid)
            if just_generated:
                # Clear the flag regardless of what the user says
                await self.session_manager.clear_just_generated(self.conversation_uuid)
                logger.debug(f"Post-generation state detected. Clearing workflow state.")
                # Also clear any stale pending action that might have been re-created
                await self.clear_pending_action()
                
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
                            response_text = "Glad you liked it! Let me know if you need anything else. 😊"
                    except Exception:
                        response_text = "Glad you liked it! Let me know if you need anything else. 😊"
                    
                    await self.session_manager.add_message(
                        self.conversation_uuid,
                        "assistant",
                        response_text
                    )
                    return response_text
            
            # === FAST PATH: Check for confirmation with pending action ===
            if is_confirmation(message):
                pending_action = await self.get_pending_action()
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
                        # Save response
                        await self.session_manager.add_message(
                            self.conversation_uuid,
                            "assistant",
                            response_text
                        )
                        return response_text
                    # If execution failed, fall through to normal Gemini flow
                else:
                    # === FALLBACK: No pending action saved, but user confirmed something ===
                    # Try to reconstruct pending action from conversation history
                    logger.debug(f"User confirmed but NO pending action found. Trying to reconstruct from history...")
                    session = await self.session_manager.get_session(self.conversation_uuid)
                    history = session.get("messages", []) if session else []
                    
                    if history:
                        # Check last assistant message for cost confirmation pattern
                        last_assistant = None
                        for msg in reversed(history):
                            if msg.get('role') == 'assistant':
                                last_assistant = msg.get('content', '')
                                break
                        
                        if last_assistant and COST_CONFIRMATION_PATTERN.search(last_assistant):
                            logger.debug(f"Found cost confirmation in last assistant message. Reconstructing action...")
                            history_with_current = history + [{'role': 'assistant', 'content': last_assistant}]
                            response_lower = last_assistant.lower()
                            
                            ref_files = await self.get_reference_files()
                            # Filter out blob: URLs
                            ref_urls = [f["url"] for f in ref_files if f.get("type") == "image" and not f.get("url", "").startswith("blob:")] if ref_files else []
                            blob_ref_urls = [f["url"] for f in ref_files if f.get("type") == "image" and f.get("url", "").startswith("blob:")] if ref_files else []
                            if blob_ref_urls:
                                logger.warning(f"Filtered out {len(blob_ref_urls)} blob: URLs during reconstruction")
                            
                            is_video = any(w in response_lower for w in ['video', 'vídeo', 'animar', 'animate', 'veo', 'runway', 'kling', 'seedance', 'tokens/se'])
                            is_speech = any(w in response_lower for w in ['speech', 'voice', 'voz', 'audio', 'narración'])
                            is_image = any(w in response_lower for w in ['imagen', 'image', 'foto', 'picture', 'gpt', 'nano banana 2', 'nano banana', 'freepik'])
                            
                            reconstructed = False
                            if is_video:
                                params = detect_video_params_from_history(history_with_current)
                                if params.get('model') and params.get('prompt'):
                                    action_args = {
                                        "prompt": params['prompt'],
                                        "model": params['model'],
                                        "duration": params.get('duration', 8)
                                    }
                                    if params.get('resolution'):
                                        action_args["resolution"] = params['resolution']
                                    if ref_urls:
                                        action_args["reference_image"] = ref_urls[0]
                                    logger.debug(f"Reconstructed VIDEO action: model={params['model']}, duration={action_args['duration']}, resolution={action_args.get('resolution')}")
                                    await self.save_pending_action("generate_video", action_args, last_assistant)
                                    reconstructed = True
                            elif is_speech:
                                params = detect_speech_params_from_history(history_with_current)
                                if params.get('text'):
                                    action_args = {"text": params['text'], "voice_id": params.get('voice_id', '21m00Tcm4TlvDq8ikWAM')}
                                    logger.debug(f"Reconstructed SPEECH action")
                                    await self.save_pending_action("generate_speech", action_args, last_assistant)
                                    reconstructed = True
                            elif is_image:
                                params = detect_image_params_from_history(history_with_current)
                                # PRIORITY: Extract prompt directly from confirmation message
                                confirmed_prompt = _extract_prompt_from_confirmation(last_assistant)
                                if confirmed_prompt:
                                    params['prompt'] = confirmed_prompt
                                    logger.debug(f"Extracted prompt from confirmation for reconstruction: {confirmed_prompt[:80]}...")
                                if params.get('model'):
                                    action_args = {"prompt": params.get('prompt', 'generate image'), "model": params['model']}
                                    if ref_urls:
                                        action_args["reference_images"] = ref_urls
                                        action_args["image_type"] = 2 if len(ref_urls) == 1 else 3
                                    logger.debug(f"Reconstructed IMAGE action: model={params['model']}")
                                    await self.save_pending_action("generate_image", action_args, last_assistant)
                                    reconstructed = True
                            
                            if reconstructed:
                                # Save user message and execute
                                await self.session_manager.add_message(self.conversation_uuid, "user", message)
                                tool_result, response_text = await self.execute_pending_action()
                                if response_text:
                                    if tool_result:
                                        # Only flag real generations (not blocks/errors)
                                        await self.session_manager.set_just_generated(self.conversation_uuid)
                                    await self.session_manager.add_message(self.conversation_uuid, "assistant", response_text)
                                    return response_text
            
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
                logger.debug(f"Ambiguous request detected, asking for clarification")
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
                    
                    logger.debug(f"Handling function call: {func_name}")
                    
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
                            response_text = self._generate_contextual_success_message(tool_result, tool_was_called=True, lang="es" if is_spanish(message) else "en")
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
                                last_tool_result, "es" if is_spanish(message) else "en"
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
                                    response_text = self._generate_contextual_success_message(last_tool_result, tool_was_called=True, lang="es" if is_spanish(message) else "en")
                            except Exception as e:
                                logger.debug(f"Failed to get followup response: {e}")
                                response_text = self._generate_contextual_success_message(last_tool_result, tool_was_called=True, lang="es" if is_spanish(message) else "en")
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
                                rfunc_args = dict(recovery_fc.args)
                                logger.debug(f"Recovery found function_call: {rfunc_name}({rfunc_args})")
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
                                    response_text = self._generate_contextual_success_message(tool_result, tool_was_called=True, lang="es" if is_spanish(message) else "en")
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
                        result_msg = self._generate_contextual_success_message(last_tool_result, tool_was_called=True, lang="es" if is_spanish(message) else "en")
                        response_text = result_msg if result_msg else str(last_tool_result)
                    else:
                        # No tool was called - ask user to retry
                        response_text = "⚠️ There was a problem processing your request. Please try again."
                else:
                    raise e
            
            # === Detect cost confirmation and save pending action ===
            if response_text and COST_CONFIRMATION_PATTERN.search(response_text):
                # Extract ALL params directly from the structured confirmation message.
                # This is far more reliable than scanning history with regex.
                conf_params = _extract_all_params_from_confirmation(response_text)
                action_type = conf_params.get("type")

                # Get reference images (filter out browser-only blob: URLs)
                ref_files = await self.get_reference_files()
                ref_urls = [
                    f["url"] for f in ref_files
                    if f.get("type") == "image" and not f.get("url", "").startswith("blob:")
                ] if ref_files else []
                blob_ref_urls = [
                    f["url"] for f in ref_files
                    if f.get("type") == "image" and f.get("url", "").startswith("blob:")
                ] if ref_files else []
                if blob_ref_urls:
                    logger.warning(
                        "Filtered out %d blob: URLs when saving pending action", len(blob_ref_urls)
                    )

                # Fall back to history scanning only when extraction from the confirmation
                # message itself was incomplete (e.g., model or prompt missing).
                if action_type == "video":
                    model = conf_params.get("model")
                    duration = conf_params.get("duration")
                    prompt = conf_params.get("prompt")
                    resolution = conf_params.get("resolution")

                    if not model or not duration or not resolution:
                        session = await self.session_manager.get_session(self.conversation_uuid)
                        history = session.get("messages", []) if session else []
                        history_with_current = history + [{"role": "assistant", "content": response_text}]
                        fallback = detect_video_params_from_history(history_with_current)
                        model = model or fallback.get("model")
                        duration = duration or fallback.get("duration")
                        prompt = prompt or fallback.get("prompt")
                        resolution = resolution or fallback.get("resolution")

                    if model and duration:
                        action_args = {
                            "prompt": prompt or "animate the image",
                            "model": model,
                            "duration": duration,
                        }
                        # Resolution only affects Seedance tiers; pass it through when known.
                        if model in ("seedance-2.0", "seedance-2.0-fast") and resolution:
                            action_args["resolution"] = resolution
                        if ref_urls:
                            action_args["reference_image"] = ref_urls[0]
                        logger.debug("Saving pending VIDEO action: %s", action_args)
                        blocked = await self._save_pending_or_block(
                            "generate_video", action_args, response_text
                        )
                        if blocked:
                            response_text = blocked

                elif action_type == "speech":
                    text_val = conf_params.get("prompt")  # "prompt" key holds speech text too
                    voice_id = "21m00Tcm4TlvDq8ikWAM"  # default Rachel

                    if not text_val:
                        session = await self.session_manager.get_session(self.conversation_uuid)
                        history = session.get("messages", []) if session else []
                        history_with_current = history + [{"role": "assistant", "content": response_text}]
                        fallback = detect_speech_params_from_history(history_with_current)
                        text_val = fallback.get("text")
                        voice_id = fallback.get("voice_id", voice_id)

                    if text_val:
                        action_args = {"text": text_val, "voice_id": voice_id}
                        logger.debug("Saving pending SPEECH action")
                        blocked = await self._save_pending_or_block(
                            "generate_speech", action_args, response_text
                        )
                        if blocked:
                            response_text = blocked

                elif action_type == "image":
                    model = conf_params.get("model")
                    prompt = conf_params.get("prompt")

                    if not model or not prompt:
                        session = await self.session_manager.get_session(self.conversation_uuid)
                        history = session.get("messages", []) if session else []
                        history_with_current = history + [{"role": "assistant", "content": response_text}]
                        fallback = detect_image_params_from_history(history_with_current)
                        model = model or fallback.get("model")
                        prompt = prompt or fallback.get("prompt")

                    if model:
                        action_args = {"prompt": prompt or "generate image", "model": model}
                        if ref_urls:
                            action_args["reference_images"] = ref_urls
                            action_args["image_type"] = 2 if len(ref_urls) == 1 else 3
                            logger.debug(
                                "Set image_type=%d (%d reference images)",
                                action_args["image_type"],
                                len(ref_urls),
                            )
                        logger.debug("Saving pending IMAGE action: %s", action_args)
                        blocked = await self._save_pending_or_block(
                            "generate_image", action_args, response_text
                        )
                        if blocked:
                            response_text = blocked
            
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

        # Structured generator error: return the friendly per-category message
        # instead of dumping the technical string on the user.
        if tool_result.startswith(GENERATION_ERROR_PREFIX):
            parsed = parse_generation_error(tool_result) or {"category": "unknown"}
            return fallback_error_message(parsed["category"], lang)

        # Check if the tool result indicates an actual error
        if tool_result_lower.startswith("error"):
            return f"⚠️ There was a problem processing your request: {tool_result}"
        
        # Only return success if the tool result confirms success
        is_success = any(word in tool_result_lower for word in [
            'exitosamente', 'successfully', 'generado', 'generated', 'generada'
        ])
        
        if not is_success:
            # Tool returned but result is unclear - return the raw result
            return tool_result
            
        # Success detected - return raw result so LLM can generate localized message
        # DO NOT return hardcoded English strings like "Your image is ready!"
        return tool_result


# TTL-based LRU cache: max 200 concurrent sessions, evict after 30 minutes of inactivity.
# Accessing a key resets the TTL, so active sessions are never evicted mid-conversation.
_chatbot_cache: TTLCache = TTLCache(maxsize=200, ttl=1800)


def get_chatbot(conversation_uuid: str = "default") -> GeminiChatbot:
    """Get or create a chatbot instance for a specific conversation."""
    if conversation_uuid not in _chatbot_cache:
        _chatbot_cache[conversation_uuid] = GeminiChatbot(conversation_uuid=conversation_uuid)
        logger.debug("Created new GeminiChatbot for uuid='%s'", conversation_uuid)
    return _chatbot_cache[conversation_uuid]
