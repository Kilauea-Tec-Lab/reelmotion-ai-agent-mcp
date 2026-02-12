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
    r'^(?:'
    # Multi-word patterns (more specific, checked first)
    r'ok dale|si dale|yes\s+please|si\s+por\s+favor|sí\s+por\s+favor|'
    r'go ahead|lets go|let\'s go|do it|'
    r'me gusta\s+(?:ese|esa|eso|este|esta)|me encanta\s+(?:ese|esa|eso|este|esta)|'
    r'me parece bien|suena bien|sounds good|looks good|that works|love it|like it|'
    r'that\'s good|that\'s great|that\'s fine|'
    r'está bien|esta bien|se ve bien|de acuerdo|'
    r'está genial|esta genial|así está bien|así mero|eso mero|'
    # Single/short word confirmations
    r'ok|okey|okay|yes|sure|yep|yeah|go|agreed|confirm|accept|done|ready|'
    r'proceed|approve|approved|nice|cool|awesome|amazing|'
    r'si|sí|dale|confirmo|confirmar|procede|hazlo|adelante|claro|afirmativo|'
    r'correcto|eso|exacto|perfecto|listo|va|venga|vamos|bueno|bien|hecho|'
    r'acepto|apruebo|aprobado|ya|anda|órale|sale|'
    # Acceptance/approval phrases (critical for refined prompt acceptance)
    r'me gusta|me encanta|genial|excelente|fantástico|así|le doy|'
    r'y|s|1|👍|✅'
    r')[\s.,!?]*$',
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
    'Nano Banana': re.compile(r'\bnano[-\s]?banana\b', re.IGNORECASE),
    'Freepik': re.compile(r'\bfreepik\b', re.IGNORECASE),
    'GPT': re.compile(r'\bgpt\b', re.IGNORECASE),
}

# Pattern to detect validation/acceptance of a refined prompt
REFINED_PROMPT_ACCEPTANCE_PATTERN = re.compile(
    r'(?:me gusta|i like|prefiero|prefer|usa|use|utiliza|usar)\s+(?:ese|esa|el|la|that|this)\s+(?:prompt|descripci[óo]n|versión|version|text)',
    re.IGNORECASE
)

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
            has_model_listing = any(listing in content for listing in ['- Runway', '- Veo', '- Sora', '- Kling'])
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
    recent_reversed = list(reversed(recent))
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
            is_video_model_selection = re.match(r'^\s*(?:sora[-\s]?2[-\s]?(?:pro)?|veo[-\s]?3\.?1[-\s]?(?:flash|ultra)?|kling[-\s]?(?:v?3[-\s]?omni[-\s]?(?:pro|std))?|runway[-\s]?(?:aleph|4\.?5)?)\s*[.,!?]*$', content, re.IGNORECASE)
            is_duration_selection = re.match(r'^\s*\d+\s*(?:segundos?|seconds?|seg|sec|s)?\s*[.,!?]*$', content, re.IGNORECASE)
            if not is_video_model_selection and not is_duration_selection:
                if i + 1 < len(recent_reversed):
                    prev_msg = recent_reversed[i+1]
                    if prev_msg.get('role') == 'assistant':
                        prev_text = prev_msg.get('content', '').lower()
                        # Did assistant propose something? (covers both English and Spanish)
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
                         if any(listing in cand for listing in ['- Runway', '- Veo', '- Sora', '- Kling']):
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
                        if any(listing in cand for listing in ['- Runway', '- Veo', '- Sora', '- Kling']):
                            continue
                        # Try to extract quoted prompt
                        quote_match = re.search(r'["\u201c]([^"\u201d]{15,})["\u201d]', cand)
                        if quote_match:
                            params['prompt'] = quote_match.group(1)
                        break
                if 'prompt' in params:
                    break
                continue
            # Skip STANDALONE model selection (just "sora 2" alone, "kling v3 omni pro", etc.)
            if re.match(r'^\s*(sora[-\s]?2[-\s]?(?:pro)?|veo[-\s]?3\.?1[-\s]?(?:flash|ultra)?|kling[-\s]?(?:v?3[-\s]?omni[-\s]?(?:pro|std))?|runway[-\s]?(?:aleph|4\.?5)?|haiper|minimax)\s*[.,!?]*$', content, re.IGNORECASE):
                continue
            # Skip duration-only messages like "5 seconds", "5s", or just "4"
            if re.match(r'^\s*\d+\s*(?:segundos?|seconds?|seg|sec|s)?\s*[\.!?]*$', content, re.IGNORECASE):
                continue
            # Skip generic video/image creation messages (too vague to be a prompt)
            if re.match(r'^\s*(?:i want to |quiero |me gustaría )?(?:create|make|genera[rt]?|crea[rt]?|haz(?:me)?|anima[rt]?|animate)\s+(?:a\s+|un\s+|una\s+)?(?:video|vídeo|imagen|image|clip)\s*[.,!?]*$', content, re.IGNORECASE):
                continue
            # Skip creation COMMANDS that include model names or durations (these are instructions, NOT descriptive prompts)
            # e.g., "Genera un video de esta imagen con veo 3.1 fast de 4s" or "Create a video with sora 2 pro 8 seconds"
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
                if any(listing in content for listing in ['- Runway', '- Veo', '- Sora', '- Kling']):
                    continue
                quote_match = re.search(r'["\u201c]([^"\u201d]{15,})["\u201d]', content)
                if quote_match:
                    params['prompt'] = quote_match.group(1)
                    break
    
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
    
    # Get the prompt (most recent user message that's not a confirmation or model selection)
    recent_reversed = list(reversed(recent))
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
            is_image_model_selection = re.match(r'^\s*(?:gpt|nano[-\s]?banana|freepik)\s*[.,!?]*$', content, re.IGNORECASE)
            if not is_image_model_selection:
                if i + 1 < len(recent_reversed):
                    prev_msg = recent_reversed[i+1]
                    if prev_msg.get('role') == 'assistant':
                        prev_text = prev_msg.get('content', '').lower()
                        # Did assistant propose something? (covers both English and Spanish)
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
                         if any(listing in cand for listing in ['Nano Banana', 'GPT', 'Freepik']):
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
                        if any(listing in cand for listing in ['Nano Banana', 'GPT', 'Freepik']):
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
                if 'Nano Banana' in content and 'GPT' in content and 'Freepik' in content:
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
        - You MUST respond in the SAME language the user is writing in.
        - Detect the language from EACH user message. Track the LAST CLEARLY IDENTIFIABLE language.
        - If the user writes in English, respond in English. If the user writes in Spanish, respond in Spanish. Same for any other language.
        - AMBIGUOUS MESSAGES: Short replies like "no", "ok", "yes", "gpt", "sora 2", "5s", model names, or single words that exist in multiple languages are NOT a language switch. KEEP the last clearly detected language.
        - LANGUAGE SWITCH: Only change language if the user writes a CLEAR sentence in a different language or explicitly requests it.
        - This applies to ALL messages: questions, confirmations, cost info, errors, EVERYTHING.
        - Keep technical terms and model names in their original form (e.g., "Nano Banana", "GPT", "Freepik").
        - IMPORTANT: All instructions below are written in English for clarity, but you MUST always respond to the user in THEIR language.
        
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
           - Mentions video models: Sora 2, Sora 2 Pro, Veo 3.1, Runway Aleph, Runway 4.5, etc.
           - IMPORTANT: Start the VIDEO WORKFLOW, do NOT call the tool directly.
           - For video EDITING (video-to-video), the user MUST provide a reference video.
             Supported models for video editing: Runway Aleph, Kling V3 Omni Std, Kling V3 Omni Pro.
        
        2. IMAGE GENERATION/EDITING INTENT:
           - "Generate image" = IMAGE workflow
           - "Create an image" = IMAGE workflow
           - "Edit image" + reference image = IMAGE-TO-IMAGE workflow
           - Mentions image models: GPT, Nano Banana, Freepik
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
           - Example: "Okay, let's start with Scene 1. We need an image of the hero. Which model do you want to use: Nano Banana, GPT, or Freepik?"
        
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
        - Based on THE_PROMPT, suggest a model and explain why:
          → GPT: Best for detailed, realistic, complex images. Recommended for most cases.
          → Nano Banana: Great for artistic, stylized, creative images.
          → Freepik: Good for clean, commercial-style images.
        - Ask: "I suggest using [model] because [reason]. Which model do you want to use: Nano Banana, GPT, or Freepik?" (in user's language)
        - Wait for the user to choose.
        - All models cost 10 tokens per image.
        
        STEP 4 - CONFIRM COST AND EXECUTE:
        - Summarize what will be generated:
          → "I'm going to generate: [brief description of THE_PROMPT]"
          → "Model: [chosen model]"
          → "Cost: 10 tokens"
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
        5. Available models are: 'Nano Banana', 'GPT', and 'Freepik' (all cost 10 tokens per image).
        6. NEVER mention URLs in your responses - images are sent automatically to the user.
        7. IF THERE'S AN ERROR: Inform the user. If user says "try again"/"retry", execute the tool again without hesitation.
        
        ═══════════════════════════════════════════════════
        CRITICAL RULES FOR 'generate_video' TOOL - MANDATORY WORKFLOW
        ═══════════════════════════════════════════════════
        
        You MUST follow these steps IN ORDER. Each step requires a SEPARATE user response.
        NEVER skip a step. NEVER combine steps. NEVER call the tool until Step 5 is confirmed.
        
        STEP 1 - IDENTIFY INTENT AND ASK FOR THE PROMPT:
        - When you detect the user wants to create a VIDEO, ask: "What do you want the video to show? Describe the action, scene, or animation." (in user's language)
        - If the user already provided a clear DESCRIPTIVE prompt, take it and IMMEDIATELY move to Step 2 in the SAME response.
        - ⚠️ COMMAND vs PROMPT: "Create a video with sora 2" is a COMMAND (not a prompt). "A frog jumping in the jungle" IS a prompt.
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
        - Show available models with costs AND valid durations:
          → Kling V3 Omni Pro (8 tokens/sec) - 3 to 15 sec - text/image-to-video, economical
          → Kling V3 Omni Std (6 tokens/sec) - 3 to 15 sec - video-to-video, most economical
          → Sora 2 (15 tokens/sec) - 4, 8 or 12 sec - good quality
          → Veo 3.1 Flash (21 tokens/sec) - 8 sec only - fast and good quality
          → Runway Aleph (19 tokens/sec) - 5 or 10 sec - video editing
          → Runway 4.5 (25 tokens/sec) - 5, 8 or 10 sec - high quality
          → Sora 2 Pro (30 tokens/sec) - 4, 8 or 12 sec - maximum Sora quality
          → Veo 3.1 (48 tokens/sec) - 8 sec only - high quality
          → Veo 3.1 Ultra (60 tokens/sec) - 8 sec only - maximum Veo quality
        - Say: "I suggest [model] because [reason]. Which model would you like to use?" (in user's language)
        - Wait for user to choose model.
        
        STEP 4 - ASK FOR DURATION:
        - Based on the chosen model, tell the user the valid durations:
          → Sora 2 / Sora 2 Pro: ONLY 4, 8 or 12 seconds
          → Veo 3.1 / Veo 3.1 Flash / Veo 3.1 Ultra: ONLY 8 seconds (auto-set, just inform)
          → Runway Aleph: 5 or 10 seconds
          → Runway 4.5: 5, 8 or 10 seconds
          → Kling V3 Omni Pro / Std: 3 to 15 seconds
        - If the model only allows ONE duration (e.g., Veo 3.1 = 8s), inform the user and auto-set it. Move to Step 5 in the SAME response.
        - Otherwise ask: "How many seconds? Options: [valid durations]" (in user's language)
        - Wait for the user to choose. VALIDATE the duration is valid for the model.
        
        STEP 5 - CONFIRM COST AND EXECUTE:
        - Calculate cost: tokens_per_second × duration
        - Summarize what will be generated:
          → "I'm going to generate a video:"
          → "Prompt: [brief description of THE_PROMPT]"
          → "Model: [model]"
          → "Duration: [X] seconds"
          → "Cost: [Y] tokens ([Z] tokens/sec × [X] sec)"
          → "Do you confirm?" (in user's language)
        - ⛔ DO NOT call the tool until the user explicitly confirms this step.
        - Once confirmed, CALL generate_video immediately using THE_PROMPT.
        
        ADDITIONAL VIDEO RULES:
        1. ⚠️ PROMPT PARAMETER RULE: Same as image - NEVER use conversational replies as prompt.
        2. FORBIDDEN to modify the agreed-upon prompt without consent.
        3. When calling the tool, use EXACT model names:
           - 'veo-3.1', 'veo-3.1-flash', 'veo-3.1-ultra'
           - 'runway-aleph', 'runway-4.5', 'sora-2', 'sora-2-pro'
           - 'kling-v3-omni-pro', 'kling-v3-omni-std'
        4. If there are attached images, use them as reference automatically (image-to-video).
        5. VIDEO-TO-VIDEO EDITING: If user wants to EDIT a video, they MUST attach the reference video.
           → Supported models: Runway Aleph, Kling V3 Omni Std, Kling V3 Omni Pro.
           → The prompt should describe the EDITING instructions (e.g., "change style to anime", "add rain effect").
           → Pass the reference video URL in the reference_video parameter.
        6. NEVER mention video URLs - they are sent automatically.
        7. IF THERE'S AN ERROR: Inform user. If they say "try again"/"retry", execute again without hesitation.
        8. Reference images are NOT lost after errors - they persist in the session.
        
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
          → "Cost: 5 tokens"
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
        FINAL REMINDER (READ THIS LAST - HIGHEST PRIORITY)
        ═══════════════════════════════════════════════════
        LANGUAGE: Your response MUST be in the SAME language as the user's message. If the user writes in English, respond ONLY in English. If in Spanish, respond ONLY in Spanish. NO EXCEPTIONS.
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
                response_text = tool_result if tool_result else "⚠️ Could not determine the operation result."
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
            
            # Send to Gemini with timeout
            GEMINI_TIMEOUT = 180  # seconds - increased for large system prompt + slow first responses
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
                clarification = "Sorry, I didn't quite understand what you want to do. Could you be more specific? For example:\n" \
                               "- Do you want to create a **video** or an **image**?\n" \
                               "- If you have a reference image: Do you want to **animate it** or **generate a similar image**?"
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
                            print(f"DEBUG [parts]: Part {idx}: has_fc={has_fc}, fc_name={fc_name}, has_text={has_text}")
                    
                    if parts_to_check:
                        for part in parts_to_check:
                            if hasattr(part, "function_call"):
                                try:
                                    fc_candidate = part.function_call
                                    if fc_candidate and hasattr(fc_candidate, 'name') and fc_candidate.name:
                                        fc = fc_candidate
                                        break
                                except Exception as e:
                                    print(f"DEBUG [parts]: Error accessing function_call: {e}")

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
                            timeout=900  # 15min - video generation (Kling) can take 10+ minutes
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
                    print(f"DEBUG: No text in Gemini response. tool_was_actually_called={tool_was_actually_called}")
                    
                    if tool_was_actually_called and last_tool_result:
                        # Tool WAS called - check if it succeeded or failed
                        tool_result_lower = last_tool_result.lower() if last_tool_result else ""
                        is_error = tool_result_lower.startswith("error")
                        
                        if is_error:
                            # Tool was called but FAILED - tell the user what went wrong
                            response_text = f"⚠️ There was a problem generating your content: {last_tool_result}\nPlease try again or adjust the parameters."
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
                                print(f"DEBUG: Recovery found function_call: {rfunc_name}({rfunc_args})")
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
                                    response_text = self._generate_contextual_success_message(tool_result, tool_was_called=True)
                                    print(f"DEBUG: Recovery tool execution success: {rfunc_name}")
                                except Exception as te:
                                    print(f"ERROR: Recovery tool execution failed: {te}")
                                    response_text = f"⚠️ There was a problem: {str(te)}"
                            elif recovery_response.text:
                                response_text = recovery_response.text
                            else:
                                response_text = "⚠️ I couldn't process your request. Could you try again with more details?"
                        except Exception as e:
                            print(f"DEBUG: Failed to get recovery response: {e}")
                            response_text = "⚠️ I couldn't process your request. Could you try again with more details?"

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
                            response_text = "⚠️ I couldn't process your request. Could you try again with more details?"
                    except Exception as e:
                        print(f"DEBUG: Failed to get recovery response after MALFORMED_FUNCTION_CALL: {e}")
                        response_text = "⚠️ I couldn't process your request. Could you try again with more details?"
                elif "finish_reason" in error_str or "response.text" in error_str:
                    print(f"WARNING: Gemini response error: {error_str}")
                    if tool_was_actually_called and last_tool_result:
                        # Tool was called - show its result
                        result_msg = self._generate_contextual_success_message(last_tool_result, tool_was_called=True)
                        response_text = result_msg if result_msg else str(last_tool_result)
                    else:
                        # No tool was called - ask user to retry
                        response_text = "⚠️ There was a problem processing your request. Please try again."
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
                
                # Include the current response_text in history for better param detection
                history_with_current = history + [{'role': 'assistant', 'content': response_text}]
                
                # Detect if this is video or image by checking BOTH response AND conversation history
                response_lower = response_text.lower()
                history_text = ' '.join([m.get('content', '') for m in history[-10:]]).lower()
                
                is_video_context = any(word in response_lower for word in ['video', 'vídeo', 'animar', 'animate', 'sora', 'veo', 'runway', 'kling'])
                is_image_context = any(word in response_lower for word in ['imagen', 'image', 'foto', 'picture'])
                
                # If response doesn't have clear keywords, check conversation history
                if not is_video_context and not is_image_context:
                    # Check if tokens/sec pattern exists (video-specific)
                    if re.search(r'tokens?/se[cg]', response_lower) or re.search(r'tokens?/se[cg]', history_text):
                        is_video_context = True
                    # Check conversation history for video model mentions
                    elif any(word in history_text for word in ['video', 'vídeo', 'sora', 'veo', 'runway', 'kling']):
                        is_video_context = True
                    elif any(word in history_text for word in ['imagen', 'image', 'gpt', 'nano banana', 'freepik']):
                        is_image_context = True
                
                if is_video_context:
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
                elif is_image_context:
                    # Detect image generation
                    params = detect_image_params_from_history(history_with_current)
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
