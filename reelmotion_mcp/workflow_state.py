"""
Explicit workflow state machine for the generation workflows.

Replaces the regex-over-history inference that used to live in chatbot.py:
the current workflow (image / video_gen / video_edit / speech), the step the
user is on, and every collected parameter are stored as a single JSON document
in Redis (workflow_state:{uuid}) and mutated ONLY through the deterministic
functions in this module.

Key invariants:
- `step` is always DERIVED from params via compute_step(); it is never trusted
  from storage or set ad hoc.
- A JSON-formatted prompt (Veo 3 style) is stored VERBATIM — byte-identical to
  what the user pasted — and is never re-serialized, paraphrased, or trimmed.
- No LLM calls and no I/O here: stdlib + pricing.py only, so tests can import
  this module without Redis/genai/env side effects.
"""
import copy
import json
import re
from typing import Dict, List, Optional, Tuple

from pricing import (
    IMAGE_COSTS,
    VIDEO_TOKEN_RATES,
    VIDEO_DURATION_RULES,
    SEEDANCE2_MODELS,
    KLING_MODELS,
    KLING_ROUTE_BASE,
    KLING_ROUTE_EDIT,
    KLING_ROUTE_TURBO,
    normalize_seedance_resolution,
    normalize_kling_quality,
)

# Models that price/behave by RESOLUTION (an explicit resolution step is asked
# before quoting cost): Seedance tiers + the Kling v3/o3 tiers.
RESOLUTION_MODELS = tuple(SEEDANCE2_MODELS) + tuple(KLING_MODELS)

STATE_VERSION = 1
WORKFLOW_STATE_TTL = 7200  # 2h — outlives pending_action (300s), shorter than session (24h)

WORKFLOW_IMAGE = "image"
WORKFLOW_VIDEO_GEN = "video_gen"
WORKFLOW_VIDEO_EDIT = "video_edit"
WORKFLOW_SPEECH = "speech"
WORKFLOW_UNKNOWN = "unknown"

VIDEO_WORKFLOWS = (WORKFLOW_VIDEO_GEN, WORKFLOW_VIDEO_EDIT)

# Models allowed for video-to-video editing (the rest don't support it).
# kling-o3 covers both the "edit an existing video" and "reference-to-video"
# routes; runway-aleph remains a high-quality editor.
VIDEO_EDIT_MODELS = ("runway-aleph", "kling-o3")

RACHEL_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

# Voice name -> ElevenLabs voice_id (moved from chatbot.py so code and the
# system prompt share one table)
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
    'rachel': RACHEL_VOICE_ID,
    'river': 'SAz9YHcvj6GT2YYXdXww',
    'roger': 'CwhRBWXzGAHq8TQ4Fs17',
    'sarah': 'EXAVITQu4vr4xnSDxMaL',
    'will': 'bIHbv24MWmeRgasZH58o',
}

# ---------------------------------------------------------------------------
# Confirmation / reaction patterns (moved from chatbot.py — the state machine
# needs them for step transitions, and chatbot.py imports them from here to
# avoid a circular import)
# ---------------------------------------------------------------------------
CONFIRMATION_PATTERNS = re.compile(
    # Repeatable group so multi-word confirmations ("yes confirm", "ok yes",
    # "si dale confirmo") match, not just a single word. Each alternative below
    # is still an exact confirmation token; a non-confirmation word ("yes no")
    # breaks the chain and fails to reach `$`.
    r'^(?:(?:'
    r'ok dale|si dale|yes\s+please|si\s+por\s+favor|sí\s+por\s+favor|'
    r'go ahead|lets go|let\'s go|do it|'
    r'me parece bien|suena bien|sounds good|that works|'
    r'that\'s good|that\'s fine|'
    r'está bien|esta bien|de acuerdo|'
    r'así está bien|'
    r'ok|okey|okay|yes|sure|yep|yeah|go|agreed|confirm|accept|'
    r'proceed|approve|approved|'
    r'si|sí|dale|confirmo|confirmar|procede|hazlo|adelante|claro|afirmativo|'
    r'correcto|exacto|va|venga|vamos|hecho|'
    r'acepto|apruebo|aprobado|anda|órale|sale|'
    r'y|s|1|👍|✅'
    r')[\s.,!?]*)+$',
    re.IGNORECASE
)

REACTION_PATTERNS = re.compile(
    r'^(?:'
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
    r'wow+|woo+|yay|siii+|sí+'
    r')[\s.,!?]*$',
    re.IGNORECASE
)

_DECLINE_PATTERNS = re.compile(
    r'^(?:no|nope|nah|skip|no gracias|no thanks|d[ée]jalo|tal cual|'
    r'as[ií] (?:est[áa]) bien|keep it|use it as is|use mine|usa el m[ií]o)\b',
    re.IGNORECASE
)

# A message that only EXPRESSES creation intent ("quiero crear un video") is
# not a prompt — the descriptive text comes in a later message.
_INTENT_ONLY_PATTERN = re.compile(
    r'^\s*(?:i want to |i\'d like to |quiero |me gustar[ií]a |por favor )?'
    r'(?:create|make|genera[rt]?|crea[rt]?|haz(?:me)?|anima[rt]?|animate|edit|edita[rt]?)'
    r'\s+(?:a\s+|an\s+|un\s+|una\s+|el\s+|la\s+|this\s+|este\s+|esta\s+)?'
    r'(?:video|v[ií]deo|imagen|image|picture|foto|photo|clip|speech|voz|voice|audio)s?\s*[.,!?]*\s*$',
    re.IGNORECASE
)


def is_confirmation(message: str) -> bool:
    """Check if the message is a simple confirmation (intent to execute/proceed)."""
    return bool(CONFIRMATION_PATTERNS.match((message or "").strip().lower()))


def is_reaction(message: str) -> bool:
    """Check if the message is a reaction/compliment to a result (NOT a confirmation)."""
    return bool(REACTION_PATTERNS.match((message or "").strip().lower()))


def is_refine_decline(message: str) -> bool:
    """Check if the message declines the prompt-refinement offer."""
    return bool(_DECLINE_PATTERNS.match((message or "").strip().lower()))


# ---------------------------------------------------------------------------
# JSON prompt detection (Veo 3 style structured prompts)
# ---------------------------------------------------------------------------
_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)

# Keys that strongly suggest the JSON prompt targets a VIDEO model
_VIDEO_JSON_KEYS = frozenset(
    ("camera", "motion", "movement", "duration", "audio", "sfx", "music",
     "dialogue", "cinematography", "shot", "transition", "fps")
)


def _try_parse_json_object(candidate: str) -> Optional[dict]:
    """json.loads that only accepts a non-empty JSON object."""
    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError):
        return None
    if isinstance(parsed, dict) and parsed:
        return parsed
    return None


def _scan_balanced_braces(text: str, start: int) -> Optional[int]:
    """Return the index of the brace closing text[start]=='{', respecting strings."""
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def detect_json_prompt(message: str) -> Optional[str]:
    """
    Return the EXACT substring of the message that is a valid JSON object
    prompt, or None.

    Handles: a raw {...} object, ```json fenced blocks, fenced blocks without
    a language tag, and prose before/after the JSON. The returned string is a
    verbatim slice of the input — never re-serialized — so it can be sent to
    the generation backend byte-identical.
    """
    if not message or "{" not in message:
        return None

    # Fenced blocks first: the fence is an explicit signal.
    for match in _FENCED_BLOCK_RE.finditer(message):
        candidate = match.group(1).strip()
        if _try_parse_json_object(candidate) is not None:
            return candidate

    # Bare JSON: brace-depth scan from each opening brace (first valid wins).
    start = message.find("{")
    while start != -1:
        end = _scan_balanced_braces(message, start)
        if end is not None:
            candidate = message[start:end + 1]
            if _try_parse_json_object(candidate) is not None:
                return candidate
        start = message.find("{", start + 1)
    return None


def json_prompt_summary(json_prompt: str) -> Tuple[int, bool]:
    """Return (field_count, looks_like_video) for a detected JSON prompt."""
    parsed = _try_parse_json_object(json_prompt) or {}
    keys = {str(k).lower() for k in parsed.keys()}
    return len(parsed), bool(keys & _VIDEO_JSON_KEYS)


# ---------------------------------------------------------------------------
# Parameter parsers (code-first, no LLM)
# ---------------------------------------------------------------------------
def _build_model_alias_table() -> Dict[str, str]:
    """alias (lowercase) -> canonical pricing.py key."""
    table: Dict[str, str] = {}
    for canonical in (
        list(IMAGE_COSTS) + list(VIDEO_TOKEN_RATES)
        + list(SEEDANCE2_MODELS) + list(KLING_MODELS)
    ):
        table[canonical.lower()] = canonical
    table.update({
        "seedance 2.0 fast": "seedance-2.0-fast",
        "seedance 2 fast": "seedance-2.0-fast",
        "seedance2 fast": "seedance-2.0-fast",
        "seedance fast": "seedance-2.0-fast",
        "seedance-2-fast": "seedance-2.0-fast",
        "seedance 2.0": "seedance-2.0",
        "seedance 2": "seedance-2.0",
        "seedance2": "seedance-2.0",
        "seedance-2": "seedance-2.0",
        "veo 3.1 ultra": "veo-3.1-ultra",
        "veo 3.1 flash": "veo-3.1-flash",
        "veo 3.1": "veo-3.1",
        "veo3.1": "veo-3.1",
        "veo 3,1": "veo-3.1",
        "runway aleph": "runway-aleph",
        "runway 4.5": "runway-4.5",
        "runway 4,5": "runway-4.5",
        # Kling v3 / v3-turbo / o3 (Evolink). The old kling-v3-omni-*/kling-v1
        # keys no longer exist on the backend (they 400) — these aliases map
        # legacy phrasings onto the closest current tier.
        "kling v3 turbo": "kling-v3-turbo",
        "kling-v3 turbo": "kling-v3-turbo",
        "kling v3-turbo": "kling-v3-turbo",
        "kling turbo": "kling-v3-turbo",
        "kling v3": "kling-v3",
        "kling o3": "kling-o3",
        "kling-o3": "kling-o3",
        "kling pro": "kling-v3",
        "kling std": "kling-v3-turbo",
        "kling standard": "kling-v3-turbo",
        "kling v3 omni pro": "kling-v3",
        "kling v3 omni std": "kling-v3-turbo",
        "kling omni pro": "kling-v3",
        "kling omni std": "kling-v3-turbo",
        "nano banana 2": "Nano Banana 2",
        "nano banana": "Nano Banana 2",
        "nanobanana": "Nano Banana 2",
        "gpt": "GPT",
        "seedream": "Seedream",
        "midjourney": "Midjourney",
        "mid journey": "Midjourney",
    })
    return table


_MODEL_ALIASES = _build_model_alias_table()
# Longest alias first so "veo 3.1 ultra" wins over "veo 3.1"
_MODEL_ALIAS_RE = re.compile(
    r"(?<![\w.])(" + "|".join(
        re.escape(alias) for alias in sorted(_MODEL_ALIASES, key=len, reverse=True)
    ) + r")(?![\w.])",
    re.IGNORECASE,
)


def normalize_model_name(text: str) -> Optional[str]:
    """Find a known model mention in free text and return its canonical name."""
    if not text:
        return None
    match = _MODEL_ALIAS_RE.search(text)
    if not match:
        return None
    return _MODEL_ALIASES[match.group(1).lower()]


_DURATION_SUFFIX_RE = re.compile(
    r"\b(\d+)\s*(?:s|sec|secs|seconds?|seg|segs|segundos?)\b", re.IGNORECASE
)
_BARE_INT_RE = re.compile(r"^\s*(\d+)\s*[.,!?]*\s*$")
_RESOLUTION_RE = re.compile(r"\b(480p|720p|1080p|4k)\b", re.IGNORECASE)
# Bare pixel heights ("1080" instead of "1080p") are only read as a resolution
# in the awaiting_resolution step — users routinely drop the "p" there. Not read
# elsewhere so a stray number never gets mistaken for a resolution.
_BARE_RESOLUTION_RE = re.compile(r"\b(480|720|1080)\b")
_VOICE_RE = re.compile(
    r"\b(" + "|".join(sorted(VOICE_NAME_TO_ID, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def parse_duration(text: str, model: Optional[str] = None, allow_bare: bool = True) -> Optional[int]:
    """
    Extract a duration in seconds. Bare integers ("8") are only accepted when
    allow_bare is True (callers pass it only in the awaiting_duration step).
    Validates against VIDEO_DURATION_RULES when the model is known.
    """
    duration = None
    if allow_bare:
        bare = _BARE_INT_RE.match(text or "")
        if bare:
            duration = int(bare.group(1))
    if duration is None:
        suffixed = _DURATION_SUFFIX_RE.search(text or "")
        if suffixed:
            duration = int(suffixed.group(1))
    if duration is None:
        return None
    if model and model in VIDEO_DURATION_RULES and duration not in VIDEO_DURATION_RULES[model]:
        return None
    return duration


def parse_resolution(text: str, allow_bare: bool = False) -> Optional[str]:
    match = _RESOLUTION_RE.search(text or "")
    if match:
        return match.group(1).lower()
    if allow_bare:
        bare = _BARE_RESOLUTION_RE.search(text or "")
        if bare:
            return bare.group(1) + "p"
    return None


def parse_voice(text: str) -> Optional[Tuple[str, str]]:
    """Return (voice_name, voice_id) when a known voice name appears in the text."""
    match = _VOICE_RE.search(text or "")
    if not match:
        return None
    name = match.group(1).lower()
    return name.capitalize(), VOICE_NAME_TO_ID[name]


# ---------------------------------------------------------------------------
# Workflow intent detection (replaces keyword checks scattered in chatbot.py)
# ---------------------------------------------------------------------------
_CREATION_VERBS = ("crea", "genera", "haz", "make", "create", "generate",
                   "anima", "animate", "edita", "edit", "modifica", "modify")
_VIDEO_MARKERS = ("video", "vídeo", "anima", "animate", "movimiento", "movement", "clip")
_IMAGE_MARKERS = ("imagen", "image", "foto", "photo", "picture", "draw", "dibuj")
_SPEECH_MARKERS = ("speech", "voice", "voz", "narra", "narración", "narracion",
                   "narration", "audio", "habla")
_EDIT_MARKERS = ("edit", "edita", "modifica", "modify", "cambia el video", "change the video")

# "crea una imagen" / "make a video": the object right after a creation verb is
# the strongest intent signal (beats loose markers elsewhere in the message,
# e.g. "olvida el video, mejor crea una imagen").
_VERB_OBJECT_RE = re.compile(
    r'(?:crea\w*|genera\w*|haz(?:me)?|make|create|generate|draw|dibuja\w*|'
    r'anima\w*|animate|edit\w*|edita\w*)'
    r'\s+(?:\w+\s+){0,2}?'
    r'(video|v[ií]deo|imagen|image|foto|photo|picture|clip|speech|voz|voice|audio)',
    re.IGNORECASE,
)
_OBJECT_TO_WORKFLOW = {
    "video": WORKFLOW_VIDEO_GEN, "vídeo": WORKFLOW_VIDEO_GEN,
    "clip": WORKFLOW_VIDEO_GEN,
    "imagen": WORKFLOW_IMAGE, "image": WORKFLOW_IMAGE, "foto": WORKFLOW_IMAGE,
    "photo": WORKFLOW_IMAGE, "picture": WORKFLOW_IMAGE,
    "speech": WORKFLOW_SPEECH, "voz": WORKFLOW_SPEECH, "voice": WORKFLOW_SPEECH,
    "audio": WORKFLOW_SPEECH,
}


def detect_workflow_intent(message: str, has_reference_video: bool = False) -> Optional[str]:
    """
    Detect which workflow a message starts. Returns a WORKFLOW_* constant or
    None when the message carries no clear generation intent.
    """
    msg = (message or "").lower()
    if not msg:
        return None

    # Verb+object ("crea una imagen") is the strongest signal — check it first
    # so "olvida el video, mejor crea una imagen" resolves to image.
    verb_object = _VERB_OBJECT_RE.search(msg)
    if verb_object:
        intent = _OBJECT_TO_WORKFLOW.get(verb_object.group(1).lower())
        if intent == WORKFLOW_VIDEO_GEN:
            has_edit = any(m in msg for m in _EDIT_MARKERS)
            return WORKFLOW_VIDEO_EDIT if (has_edit and has_reference_video) else WORKFLOW_VIDEO_GEN
        if intent:
            return intent

    model = normalize_model_name(msg)
    has_video_marker = any(m in msg for m in _VIDEO_MARKERS)
    has_image_marker = any(m in msg for m in _IMAGE_MARKERS)
    has_speech_marker = any(m in msg for m in _SPEECH_MARKERS)
    has_edit_marker = any(m in msg for m in _EDIT_MARKERS)

    if model is not None:
        if model in IMAGE_COSTS:
            return WORKFLOW_IMAGE
        if has_edit_marker and has_reference_video:
            return WORKFLOW_VIDEO_EDIT
        return WORKFLOW_VIDEO_GEN
    if has_video_marker:
        if has_edit_marker and has_reference_video:
            return WORKFLOW_VIDEO_EDIT
        return WORKFLOW_VIDEO_GEN
    if has_image_marker:
        return WORKFLOW_IMAGE
    if has_speech_marker:
        return WORKFLOW_SPEECH
    return None


def _has_creation_verb(message: str) -> bool:
    msg = (message or "").lower()
    return any(verb in msg for verb in _CREATION_VERBS)


# ---------------------------------------------------------------------------
# State construction and step derivation
# ---------------------------------------------------------------------------
def new_state(workflow_type: str = WORKFLOW_UNKNOWN) -> dict:
    state = {
        "version": STATE_VERSION,
        "workflow_type": workflow_type,
        "step": "",
        "params": {
            "prompt": None,
            "prompt_format": "text",
            "candidate_prompt": None,
            # Starts resolved: the agent no longer OFFERS refinement on its own.
            # The unsolicited "would you like me to refine your prompt?" turn was
            # the single biggest drop-off in the funnel (124 abandoned chats), so
            # the workflow goes prompt -> model directly. Refinement still runs
            # when the user explicitly asks for it: Gemini emits a ✨ candidate,
            # capture_refined_prompt() stores it, and compute_step() routes to
            # "refining" on the strength of that candidate alone.
            "refine_resolved": True,
            "model": None,
            "duration": None,
            "resolution": None,
            "voice_id": None,
            "voice_name": None,
            "quantity": 1,
            "speech_text": None,
            "text_confirmed": False,
        },
        "updated_at": "",
    }
    state["step"] = compute_step(state)
    return state


def compute_step(state: dict) -> str:
    """Derive the current step from params. Single source of truth for `step`."""
    workflow_type = state.get("workflow_type", WORKFLOW_UNKNOWN)
    params = state.get("params", {})

    if workflow_type == WORKFLOW_SPEECH:
        if not params.get("speech_text"):
            return "awaiting_text"
        if not params.get("text_confirmed"):
            return "confirm_text"
        if not params.get("voice_id"):
            return "awaiting_voice"
        return "awaiting_confirmation"

    if not params.get("prompt"):
        return "awaiting_prompt"
    # A pending candidate means the user ASKED for a rewrite and Gemini proposed
    # one — wait for them to accept it regardless of refine_resolved.
    if params.get("candidate_prompt"):
        return "refining"
    if not params.get("refine_resolved"):
        return "offer_refine"
    if workflow_type == WORKFLOW_UNKNOWN:
        return "awaiting_type"
    if not params.get("model"):
        return "awaiting_model"
    if workflow_type in VIDEO_WORKFLOWS:
        model = params["model"]
        if model in RESOLUTION_MODELS and not params.get("resolution"):
            return "awaiting_resolution"
        if not params.get("duration"):
            return "awaiting_duration"
    return "awaiting_confirmation"


# ---------------------------------------------------------------------------
# The deterministic transition function
# ---------------------------------------------------------------------------
def apply_user_message(state: dict, message: str, has_reference_video: bool = False) -> dict:
    """
    Apply a user message to the workflow state. Pure function: returns a new
    state, never mutates the input, never calls an LLM.
    """
    state = copy.deepcopy(state)
    params = state["params"]
    msg = (message or "").strip()
    if not msg:
        state["step"] = compute_step(state)
        return state
    step = compute_step(state)

    # JSON detection runs FIRST so intent/model/duration parsers never read
    # inside the JSON prompt's values.
    json_prompt = detect_json_prompt(msg)
    scan_text = msg.replace(json_prompt, " ") if json_prompt else msg

    # --- Workflow type switch / resolution -------------------------------
    intent = detect_workflow_intent(scan_text, has_reference_video)
    if intent:
        if state["workflow_type"] == WORKFLOW_UNKNOWN:
            # Resolves "awaiting_type" ("video" answering "image or video?")
            # while keeping the already-captured prompt (e.g. a pasted JSON).
            state["workflow_type"] = intent
        elif intent != state["workflow_type"] and _has_creation_verb(scan_text):
            # Explicit new request of a different type → restart the workflow.
            state = new_state(intent)
            params = state["params"]

    # --- JSON prompt (always wins, skips text refinement) ------------------
    if json_prompt:
        params["prompt"] = json_prompt
        params["prompt_format"] = "json"
        params["candidate_prompt"] = None
        params["refine_resolved"] = True
        if state["workflow_type"] == WORKFLOW_SPEECH:
            # JSON prompts don't apply to speech; treat as a fresh unknown flow
            state["workflow_type"] = WORKFLOW_UNKNOWN

    # --- Model (accepted at any step; mid-flow changes allowed) ------------
    model = normalize_model_name(scan_text)
    if model:
        is_image_model = model in IMAGE_COSTS
        if state["workflow_type"] == WORKFLOW_UNKNOWN:
            state["workflow_type"] = WORKFLOW_IMAGE if is_image_model else WORKFLOW_VIDEO_GEN
        type_matches = (
            (is_image_model and state["workflow_type"] == WORKFLOW_IMAGE)
            or (not is_image_model and state["workflow_type"] in VIDEO_WORKFLOWS)
        )
        edit_allowed = (
            state["workflow_type"] != WORKFLOW_VIDEO_EDIT or model in VIDEO_EDIT_MODELS
        )
        if type_matches and edit_allowed:
            params["model"] = model
            if model.startswith("veo-"):
                params["duration"] = 8  # Veo family is fixed at 8s
            elif params.get("duration") is not None:
                rules = VIDEO_DURATION_RULES.get(model)
                if rules and params["duration"] not in rules:
                    params["duration"] = None  # previous duration invalid for new model
            if model not in RESOLUTION_MODELS:
                params["resolution"] = None

    # --- Duration / resolution (video workflows) ----------------------------
    if state["workflow_type"] in VIDEO_WORKFLOWS:
        duration = parse_duration(
            scan_text, params.get("model"), allow_bare=(step == "awaiting_duration")
        )
        if duration is not None:
            params["duration"] = duration
        resolution = parse_resolution(
            scan_text, allow_bare=(step == "awaiting_resolution")
        )
        if resolution is not None:
            model_for_res = params.get("model")
            if model_for_res in SEEDANCE2_MODELS:
                params["resolution"] = normalize_seedance_resolution(model_for_res, resolution)
            elif model_for_res in KLING_MODELS:
                params["resolution"] = normalize_kling_quality(model_for_res, None, resolution)
            else:
                params["resolution"] = resolution

    # --- Speech -------------------------------------------------------------
    if state["workflow_type"] == WORKFLOW_SPEECH:
        voice = parse_voice(scan_text)
        if voice:
            params["voice_name"], params["voice_id"] = voice
        if step == "awaiting_text":
            if not is_confirmation(msg) and not _INTENT_ONLY_PATTERN.match(msg) and len(msg) > 3 and not voice:
                params["speech_text"] = msg
        elif step == "confirm_text":
            if is_confirmation(msg):
                params["text_confirmed"] = True
            elif len(msg) > 20 and not voice:
                params["speech_text"] = msg  # user replaced the text
        elif step == "awaiting_voice":
            if voice is None and is_confirmation(msg):
                params["voice_name"], params["voice_id"] = "Rachel", RACHEL_VOICE_ID
            if voice or is_confirmation(msg):
                params["text_confirmed"] = True

    # --- Prompt capture / refinement flow (image & video) -------------------
    elif not json_prompt:
        if step == "awaiting_prompt":
            is_substantive = (
                len(msg) > 15
                and not is_confirmation(msg)
                and not _INTENT_ONLY_PATTERN.match(msg)
                # A short message that names a model is a model pick, not a
                # prompt; a long one ("a frog jumping..., use veo 3.1") is both.
                and (model is None or len(msg) > 40)
            )
            if is_substantive:
                params["prompt"] = msg
                params["prompt_format"] = "text"
        elif step == "offer_refine":
            if is_refine_decline(msg):
                params["refine_resolved"] = True
            # is_confirmation → user wants help; Gemini will produce the ✨
            # candidate which capture_refined_prompt() picks up.
        elif step == "refining":
            accepts = is_confirmation(msg) or is_reaction(msg)
            if accepts or is_refine_decline(msg):
                if accepts and params.get("candidate_prompt"):
                    params["prompt"] = params["candidate_prompt"]
                params["candidate_prompt"] = None
                params["refine_resolved"] = True
            # otherwise the user is iterating; Gemini refreshes the candidate.

    state["step"] = compute_step(state)
    return state


_REFINED_QUOTE_RE = re.compile(r'["“](.{10,}?)["”]', re.DOTALL)


def capture_refined_prompt(state: dict, assistant_text: str) -> dict:
    """
    Capture the refined prompt Gemini proposes (the standardized
    '✨ **Refined Prompt:** "..."' block) into candidate_prompt.
    Returns the (possibly unchanged) state; pure function.
    """
    if not assistant_text or "✨" not in assistant_text:
        return state
    after_marker = assistant_text[assistant_text.find("✨"):]

    candidate = None
    if state["params"].get("prompt_format") == "json":
        candidate = detect_json_prompt(after_marker)
    if candidate is None:
        quote_match = _REFINED_QUOTE_RE.search(after_marker)
        if quote_match:
            candidate = quote_match.group(1).strip()
    if not candidate:
        return state

    state = copy.deepcopy(state)
    state["params"]["candidate_prompt"] = candidate
    state["step"] = compute_step(state)
    return state


# ---------------------------------------------------------------------------
# Extractor merge (LLM fallback results → state)
# ---------------------------------------------------------------------------
_EXTRACTOR_INTENTS = {
    "image": WORKFLOW_IMAGE,
    "video_gen": WORKFLOW_VIDEO_GEN,
    "video_edit": WORKFLOW_VIDEO_EDIT,
    "speech": WORKFLOW_SPEECH,
}


def merge_extracted(state: Optional[dict], extracted: dict) -> dict:
    """
    Merge a param_extractor result into the state. Extracted values only fill
    gaps — they never overwrite params already captured deterministically.
    """
    state = copy.deepcopy(state) if state else new_state()
    params = state["params"]
    extracted = extracted or {}

    intent = _EXTRACTOR_INTENTS.get(extracted.get("intent"))
    if intent and state["workflow_type"] == WORKFLOW_UNKNOWN:
        state["workflow_type"] = intent

    prompt = (extracted.get("prompt") or "").strip()
    if prompt and len(prompt) > 5:
        if state["workflow_type"] == WORKFLOW_SPEECH:
            if not params.get("speech_text"):
                params["speech_text"] = prompt
                params["text_confirmed"] = True
        elif not params.get("prompt"):
            params["prompt"] = prompt
            params["prompt_format"] = "json" if detect_json_prompt(prompt) == prompt else "text"
            params["refine_resolved"] = True

    model = normalize_model_name(extracted.get("model") or "")
    if model and not params.get("model"):
        params["model"] = model
        if model.startswith("veo-"):
            params["duration"] = 8

    duration = extracted.get("duration")
    if isinstance(duration, int) and duration > 0 and not params.get("duration"):
        params["duration"] = duration

    resolution = (extracted.get("resolution") or "").lower()
    if resolution in ("480p", "720p", "1080p", "4k") and not params.get("resolution"):
        params["resolution"] = resolution

    voice = parse_voice(extracted.get("voice_name") or "")
    if voice and not params.get("voice_id"):
        params["voice_name"], params["voice_id"] = voice

    if state["workflow_type"] == WORKFLOW_SPEECH and params.get("speech_text"):
        params["text_confirmed"] = True
        if not params.get("voice_id"):
            params["voice_name"], params["voice_id"] = "Rachel", RACHEL_VOICE_ID

    state["step"] = compute_step(state)
    return state


# ---------------------------------------------------------------------------
# Readiness and action construction
# ---------------------------------------------------------------------------
def missing_fields(state: dict) -> List[str]:
    """Fields still required before the cost-confirmation step."""
    workflow_type = state.get("workflow_type", WORKFLOW_UNKNOWN)
    params = state.get("params", {})
    missing = []

    if workflow_type == WORKFLOW_SPEECH:
        if not params.get("speech_text"):
            missing.append("speech_text")
        return missing  # voice defaults to Rachel

    if not params.get("prompt"):
        missing.append("prompt")
    if workflow_type == WORKFLOW_UNKNOWN:
        missing.append("workflow_type")
        return missing
    if not params.get("model"):
        missing.append("model")
    if workflow_type in VIDEO_WORKFLOWS:
        if not params.get("duration"):
            missing.append("duration")
        if params.get("model") in RESOLUTION_MODELS and not params.get("resolution"):
            missing.append("resolution")
    return missing


def is_ready_for_confirmation(state: Optional[dict]) -> bool:
    return bool(state) and not missing_fields(state)


def build_action_args(state: dict, ref_urls: Optional[List[str]] = None) -> Optional[Tuple[str, dict]]:
    """
    Build (function_name, args) for a pending action from the state. This is
    the ONLY place pending-action args are constructed. Returns None when
    required fields are missing.
    """
    if not is_ready_for_confirmation(state):
        return None
    workflow_type = state["workflow_type"]
    params = state["params"]
    ref_urls = ref_urls or []

    if workflow_type == WORKFLOW_IMAGE:
        args = {
            "prompt": params["prompt"],
            "model": params["model"],
            "quantity": params.get("quantity") or 1,
        }
        if len(ref_urls) == 1:
            args["image_type"] = 2
            args["reference_image"] = ref_urls[0]
        elif ref_urls:
            args["image_type"] = 3
            args["reference_images"] = ref_urls
        else:
            args["image_type"] = 1
        return "generate_image", args

    if workflow_type in VIDEO_WORKFLOWS:
        model = params["model"]
        args = {
            "prompt": params["prompt"],
            "model": model,
            "duration": params["duration"],
        }
        if model in SEEDANCE2_MODELS:
            args["resolution"] = normalize_seedance_resolution(model, params.get("resolution"))
        elif model in KLING_MODELS:
            route = (
                KLING_ROUTE_EDIT
                if (workflow_type == WORKFLOW_VIDEO_EDIT and model == "kling-o3")
                else (KLING_ROUTE_TURBO if model == "kling-v3-turbo" else KLING_ROUTE_BASE)
            )
            args["resolution"] = normalize_kling_quality(model, route, params.get("resolution"))
            # mode=edit makes the cost estimate use the edit rate; the tool also
            # auto-detects the edit route from the attached reference video.
            if workflow_type == WORKFLOW_VIDEO_EDIT and model == "kling-o3":
                args["mode"] = "edit"
        # Reference files are attached by execute_pending_action from refs:{uuid}
        return "generate_video", args

    if workflow_type == WORKFLOW_SPEECH:
        return "generate_speech", {
            "text": params["speech_text"],
            "voice_id": params.get("voice_id") or RACHEL_VOICE_ID,
        }

    return None


# ---------------------------------------------------------------------------
# Gemini context note
# ---------------------------------------------------------------------------
def state_context_note(state: dict) -> str:
    """Render the [SYSTEM CONTEXT — workflow state] note injected into Gemini parts."""
    params = state.get("params", {})
    parts = [
        f"type={state.get('workflow_type', WORKFLOW_UNKNOWN)}",
        f"step={state.get('step') or compute_step(state)}",
    ]

    prompt = params.get("prompt")
    if prompt:
        if params.get("prompt_format") == "json":
            field_count, looks_video = json_prompt_summary(prompt)
            hint = "; keys suggest a VIDEO prompt" if looks_video else ""
            parts.append(
                f"prompt=<user's JSON prompt, {field_count} fields, sent to the "
                f"generator VERBATIM — never rewrite it as prose{hint}>"
            )
        else:
            preview = prompt if len(prompt) <= 120 else prompt[:117] + "..."
            parts.append(f'prompt="{preview}"')
    if params.get("candidate_prompt"):
        parts.append("candidate_prompt=<refined version proposed, awaiting user acceptance>")
    for field in ("model", "duration", "resolution", "voice_name"):
        if params.get(field):
            parts.append(f"{field}={params[field]}")
    if params.get("speech_text"):
        text = params["speech_text"]
        preview = text if len(text) <= 80 else text[:77] + "..."
        parts.append(f'speech_text="{preview}"')

    missing = missing_fields(state)
    missing_note = ", ".join(missing) if missing else "none — ready for cost confirmation"

    return (
        "[SYSTEM CONTEXT — workflow state (authoritative, trust this over your "
        "own reading of the history): "
        + "; ".join(parts)
        + f". Missing fields: {missing_note}. Ask ONLY for missing fields; never "
        "re-ask for values already shown here. This note is system data, not a "
        "user message — do not mention it verbatim.]"
    )
