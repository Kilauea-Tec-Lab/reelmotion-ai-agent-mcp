"""
Single source of truth for generation pricing.

These tables mirror the rates charged by the reelmotion Laravel backend
(the real biller). Every place that needs a cost — tool docstrings, the
system prompt, balance validation — must read from here instead of
hardcoding numbers.

This module is stdlib-only on purpose: chatbot.py, tools.py, and tests can
import it without pulling in Redis/httpx/env side effects.
"""
import math
import re
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Image pricing (tokens per image)
# ---------------------------------------------------------------------------
# Exact, case-sensitive model names as the backend (/api/ai/mcp-image-generation)
# expects them. There is NO "Freepik" model — never offer or select it.
IMAGE_COSTS: Dict[str, int] = {
    "Seedream": 4,
    "GPT": 6,
    "Nano Banana 2": 7,
    "Midjourney": 9,
}

# Only these image models honor `type`/`quantity` (multi-image in one call).
# Seedream and Midjourney always produce exactly ONE image per call — quantity
# is ignored, so multiple images mean multiple calls (each billed separately).
QUANTITY_IMAGE_MODELS = ("GPT", "Nano Banana 2")

# ---------------------------------------------------------------------------
# Video pricing — flat tokens/second models
# ---------------------------------------------------------------------------
# NOTE: the kling-v3 / kling-v3-turbo / kling-o3 tiers are NOT here — their
# rate depends on resolution + route + audio, so they live in KLING_TOKEN_RATES
# below (mirrors how Seedance is handled).
VIDEO_TOKEN_RATES: Dict[str, int] = {
    "runway-aleph": 17,
    "runway-4.5": 13,
    "veo-3.1": 44,
    "veo-3.1-flash": 17,
    "veo-3.1-ultra": 65,
}

# Valid durations (seconds) per video model.
VIDEO_DURATION_RULES: Dict[str, List[int]] = {
    "veo-3.1": [8],
    "veo-3.1-flash": [8],
    "veo-3.1-ultra": [8],
    "luma-labs": [5],
    "seedance-pro": [5],
    "runway": [5, 10],
    "runway-aleph": [5, 10],
    "runway-4.5": [5, 8, 10],
    "kling-v3": list(range(3, 16)),
    "kling-v3-turbo": list(range(3, 16)),
    "kling-o3": list(range(3, 16)),
    "seedance-2.0": list(range(4, 16)),
    "seedance-2.0-fast": list(range(4, 16)),
}

# ---------------------------------------------------------------------------
# Seedance 2.0 pricing & helpers
# ---------------------------------------------------------------------------
# Seedance is the only video tier whose price depends on RESOLUTION (not just
# duration). The backend (/api/ai/generate-video) does the real charging; these
# tables mirror its rates so the agent can quote the cost before generating.
SEEDANCE2_MODELS = ("seedance-2.0", "seedance-2.0-fast")

# tokens per second, indexed by resolution. "fast" tier has no 1080p.
SEEDANCE2_TOKEN_RATES = {
    "normal": {
        "seedance-2.0": {"480p": 15, "720p": 32, "1080p": 72},
        "seedance-2.0-fast": {"480p": 12, "720p": 26},
    },
    # Discounted rate applies ONLY in reference mode when reference_videos are
    # sent (fal.ai charges ×0.6 in that case).
    "reference_discount": {
        "seedance-2.0": {"480p": 9, "720p": 20, "1080p": 43},
        "seedance-2.0-fast": {"480p": 7, "720p": 16},
    },
}

SEEDANCE2_VALID_ASPECT_RATIOS = ("auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16")


def normalize_seedance_resolution(model: str, resolution: Optional[str]) -> str:
    """Return a valid resolution for the given Seedance tier, falling back to 720p."""
    res = (resolution or "720p").lower().strip()
    valid_for_model = SEEDANCE2_TOKEN_RATES["normal"].get(model, {})
    if res not in valid_for_model:
        # Invalid resolution for this tier (e.g., 1080p on the fast tier) -> 720p
        res = "720p"
    return res


def normalize_seedance_duration(duration) -> int:
    """Normalize a Seedance duration to an int in the valid 4-15 range (default 5)."""
    if duration in (None, "", "auto"):
        return 5
    try:
        dur = int(duration)
    except (TypeError, ValueError):
        return 5
    return max(4, min(15, dur))


def compute_seedance2_cost(
    model: str,
    resolution: Optional[str],
    duration,
    has_reference_videos: bool = False,
) -> int:
    """Total token cost for a Seedance 2.0 generation = rate(resolution) × duration."""
    res = normalize_seedance_resolution(model, resolution)
    table_key = "reference_discount" if has_reference_videos else "normal"
    per_sec = SEEDANCE2_TOKEN_RATES[table_key].get(model, {}).get(res)
    if per_sec is None:
        per_sec = SEEDANCE2_TOKEN_RATES["normal"].get(model, {}).get(res, 0)
    return per_sec * normalize_seedance_duration(duration)


# ---------------------------------------------------------------------------
# Kling v3 / o3 pricing & helpers (Evolink — /api/ai/mcp-video-generation)
# ---------------------------------------------------------------------------
# kling-v3 / kling-v3-turbo / kling-o3 price per SECOND, and the per-second rate
# depends on the ROUTE (text/image vs reference/edit vs motion-control), the
# RESOLUTION (720p/1080p/4k) and whether native AUDIO is on. tokens = rate ×
# duration. The backend is the real biller (it trims quality/audio/duration and
# reserves atomically); these tables mirror its rates so the agent can quote a
# cost BEFORE generating.
KLING_MODELS = ("kling-v3", "kling-v3-turbo", "kling-o3")

# Routes (also used by tools.py to shape the request payload)
KLING_ROUTE_BASE = "base"            # v3/o3 text-to-video or image-to-video
KLING_ROUTE_TURBO = "turbo"          # kling-v3-turbo (text/image-to-video only)
KLING_ROUTE_REFERENCE = "reference"  # kling-o3 reference-to-video (consistency)
KLING_ROUTE_EDIT = "edit"            # kling-o3 video-edit
KLING_ROUTE_MOTION = "motion"        # kling-v3 motion-control (guide video)

# Per-second token rates, indexed by route then resolution.
KLING_TOKEN_RATES = {
    KLING_ROUTE_BASE: {"720p": 9, "1080p": 12, "4k": 42},
    KLING_ROUTE_TURBO: {"720p": 12, "1080p": 14},
    KLING_ROUTE_REFERENCE: {"720p": 13, "1080p": 17},
    KLING_ROUTE_EDIT: {"720p": 13, "1080p": 17},
    KLING_ROUTE_MOTION: {"720p": 13, "1080p": 17},
}

# Audio surcharge applies ONLY to the base (v3/o3 text/image) route, at
# 720p/1080p. Audio is ignored (and not charged) on turbo/reference/edit/motion.
KLING_AUDIO_RATES = {
    KLING_ROUTE_BASE: {"720p": 12, "1080p": 14},
}

KLING_VALID_QUALITIES = ("720p", "1080p", "4k")
KLING_DURATION_MIN = 3
KLING_DURATION_MAX = 15
KLING_REFERENCE_DURATION_MAX = 10  # reference route is capped at 10s

# Cheapest reachable per-second rate per model (the base/turbo route), used by
# the affordability helpers. Advanced routes (reference/edit/motion) cost more
# and need extra media, so "what can I afford?" quotes the base route.
KLING_BASE_RATES_BY_MODEL = {
    "kling-v3": KLING_TOKEN_RATES[KLING_ROUTE_BASE],
    "kling-o3": KLING_TOKEN_RATES[KLING_ROUTE_BASE],
    "kling-v3-turbo": KLING_TOKEN_RATES[KLING_ROUTE_TURBO],
}


def default_kling_route(provider: str) -> str:
    """The route a Kling provider falls into for plain text/image-to-video."""
    return KLING_ROUTE_TURBO if provider == "kling-v3-turbo" else KLING_ROUTE_BASE


def normalize_kling_quality(provider: str, route: Optional[str], quality: Optional[str]) -> str:
    """
    Clamp a requested quality to what the provider/route actually supports,
    defaulting to 720p. 4K exists ONLY on the base (text/image) route of
    kling-v3 / kling-o3; every other case caps at 1080p.
    """
    q = (quality or "720p").lower().strip()
    if q not in KLING_VALID_QUALITIES:
        q = "720p"
    route = route or default_kling_route(provider)
    if q == "4k" and (route != KLING_ROUTE_BASE or provider == "kling-v3-turbo"):
        q = "1080p"
    return q


def normalize_kling_duration(duration, route: Optional[str] = None) -> int:
    """Clamp a Kling duration to its valid range (3-15s, reference 3-10s; default 5)."""
    if duration in (None, "", "auto"):
        return 5
    try:
        dur = int(duration)
    except (TypeError, ValueError):
        return 5
    hi = KLING_REFERENCE_DURATION_MAX if route == KLING_ROUTE_REFERENCE else KLING_DURATION_MAX
    return max(KLING_DURATION_MIN, min(hi, dur))


def kling_route_from_args(provider: str, args: Optional[dict]) -> str:
    """
    Derive the Kling route from a generation-args dict (used for cost estimation
    and to decide which media field the payload should carry).
    """
    args = args or {}
    mode = str(args.get("mode") or "").lower().strip()
    if provider == "kling-v3-turbo":
        return KLING_ROUTE_TURBO
    has_ref_video = bool(args.get("reference_videos") or args.get("reference_video"))
    if provider == "kling-o3":
        if mode == "edit" or args.get("edit_video") or has_ref_video:
            return KLING_ROUTE_EDIT
        if mode == "reference" or args.get("reference_images"):
            return KLING_ROUTE_REFERENCE
        return KLING_ROUTE_BASE
    # kling-v3
    if mode == "motion" or args.get("motion_video") or has_ref_video:
        return KLING_ROUTE_MOTION
    return KLING_ROUTE_BASE


def _kling_sound_on(args: dict, route: str) -> bool:
    """
    Audio only matters (and is only charged) on the base (v3/o3 text/image)
    route, and is opt-in via sound="on" (default off — drafts skip the
    surcharge). It is ignored on turbo/reference/edit/motion.
    """
    if route != KLING_ROUTE_BASE:
        return False
    return str(args.get("sound") or "off").lower() == "on"


def compute_kling_cost(
    provider: str,
    route: Optional[str] = None,
    quality: Optional[str] = None,
    duration=None,
    sound: bool = False,
) -> int:
    """Total token cost for a Kling generation = rate(route, resolution, audio) × duration."""
    route = route or default_kling_route(provider)
    q = normalize_kling_quality(provider, route, quality)
    rate = None
    if sound and route == KLING_ROUTE_BASE:
        rate = KLING_AUDIO_RATES.get(KLING_ROUTE_BASE, {}).get(q)
    if rate is None:
        rate = KLING_TOKEN_RATES.get(route, {}).get(q)
    if rate is None:
        rate = KLING_TOKEN_RATES[KLING_ROUTE_BASE]["720p"]
    return rate * normalize_kling_duration(duration, route)


# ---------------------------------------------------------------------------
# Speech pricing (tiered by character count)
# ---------------------------------------------------------------------------
SPEECH_TIER_1_MAX = 500   # 1-500 chars -> 1 token
SPEECH_TIER_2_MAX = 999   # 501-999 chars -> 8 tokens
SPEECH_TIER_1_COST = 1
SPEECH_TIER_2_COST = 8
SPEECH_RATE_PER_1000 = 13  # 1000+ chars -> 13 tokens per 1000 chars (rounded up)


def speech_cost(text_length: int) -> int:
    """Token cost for a TTS generation of the given character count."""
    if text_length <= 0:
        return 0
    if text_length <= SPEECH_TIER_1_MAX:
        return SPEECH_TIER_1_COST
    if text_length <= SPEECH_TIER_2_MAX:
        return SPEECH_TIER_2_COST
    return math.ceil(text_length / 1000) * SPEECH_RATE_PER_1000


# ---------------------------------------------------------------------------
# Cost estimation (used for balance validation before executing a tool)
# ---------------------------------------------------------------------------
def _normalize_image_model(model: str) -> Optional[str]:
    """Map loose image model names to the canonical IMAGE_COSTS keys."""
    if model in IMAGE_COSTS:
        return model
    lowered = (model or "").lower()
    if "seedream" in lowered:
        return "Seedream"
    if "midjourney" in lowered or lowered.strip() == "mj":
        return "Midjourney"
    if "nano" in lowered or "banana" in lowered:
        return "Nano Banana 2"
    if "gpt" in lowered:
        return "GPT"
    return None


def estimate_generation_cost(function_name: str, args: dict) -> Optional[int]:
    """
    Estimate the token cost of a generation WITHOUT executing it.

    Returns None when the cost cannot be estimated (unknown/legacy model,
    missing text) — callers must NEVER block on an unknown cost; the Laravel
    backend remains the final biller.
    """
    args = args or {}

    if function_name == "generate_image":
        model = _normalize_image_model(args.get("model", "Seedream"))
        if model is None:
            return None
        # Seedream/Midjourney always bill for exactly one image per call;
        # quantity only multiplies the cost for the models that honor it.
        if model in QUANTITY_IMAGE_MODELS:
            try:
                quantity = max(1, int(args.get("quantity", 1)))
            except (TypeError, ValueError):
                quantity = 1
        else:
            quantity = 1
        return IMAGE_COSTS[model] * quantity

    if function_name == "generate_video":
        model = args.get("model")
        duration = args.get("duration")
        if model in SEEDANCE2_MODELS:
            has_ref_videos = bool(args.get("reference_videos") or args.get("reference_video"))
            return compute_seedance2_cost(
                model, args.get("resolution"), duration, has_reference_videos=has_ref_videos
            )
        if model in KLING_MODELS:
            if duration is None:
                return None
            route = kling_route_from_args(model, args)
            quality = args.get("quality") or args.get("resolution")
            sound = _kling_sound_on(args, route)
            return compute_kling_cost(model, route, quality, duration, sound)
        rate = VIDEO_TOKEN_RATES.get(model)
        if rate is None:
            # legacy/unknown or unpriced model (runway, luma-labs,
            # seedance-pro) — backend remains the final biller.
            return None
        try:
            return rate * int(duration)
        except (TypeError, ValueError):
            return None

    if function_name == "generate_speech":
        text = args.get("text")
        if not text:
            return None
        return speech_cost(len(text))

    return None


# ---------------------------------------------------------------------------
# Affordability helpers (what can the user generate with their balance?)
# ---------------------------------------------------------------------------
def affordable_options(balance: int) -> Dict:
    """
    Compute, in code, what the given balance can buy.

    Returns:
        {
          "images": [{"model": str, "cost": int}, ...],            # sorted cheap -> expensive
          "videos": [{"model": str, "resolution": Optional[str],
                      "max_duration": int, "cost": int}, ...],     # best affordable per model/res
          "speech_max_chars": int,                                  # 0 if nothing affordable
        }
    """
    images = [
        {"model": model, "cost": cost}
        for model, cost in sorted(IMAGE_COSTS.items(), key=lambda item: item[1])
        if cost <= balance
    ]

    videos = []
    for model, rate in VIDEO_TOKEN_RATES.items():
        durations = [d for d in VIDEO_DURATION_RULES.get(model, []) if rate * d <= balance]
        if durations:
            best = max(durations)
            videos.append({"model": model, "resolution": None, "max_duration": best, "cost": rate * best})
    for model, res_table in SEEDANCE2_TOKEN_RATES["normal"].items():
        for res, rate in res_table.items():
            durations = [d for d in VIDEO_DURATION_RULES.get(model, []) if rate * d <= balance]
            if durations:
                best = max(durations)
                videos.append({"model": model, "resolution": res, "max_duration": best, "cost": rate * best})
    for model, res_table in KLING_BASE_RATES_BY_MODEL.items():
        for res, rate in res_table.items():
            durations = [d for d in VIDEO_DURATION_RULES.get(model, []) if rate * d <= balance]
            if durations:
                best = max(durations)
                videos.append({"model": model, "resolution": res, "max_duration": best, "cost": rate * best})
    # Most capable options first (longest duration, then cheapest)
    videos.sort(key=lambda v: (-v["max_duration"], v["cost"]))

    if balance >= SPEECH_RATE_PER_1000:
        speech_max_chars = (balance // SPEECH_RATE_PER_1000) * 1000
    elif balance >= SPEECH_TIER_2_COST:
        speech_max_chars = SPEECH_TIER_2_MAX
    elif balance >= SPEECH_TIER_1_COST:
        speech_max_chars = SPEECH_TIER_1_MAX
    else:
        speech_max_chars = 0

    return {"images": images, "videos": videos, "speech_max_chars": speech_max_chars}


def min_video_cost() -> int:
    """
    Cheapest possible video generation across all priced models/resolutions.

    Used as the "low balance" threshold: below this the user cannot generate
    ANY video, so the frontend should surface a buy-tokens CTA.
    """
    costs = [
        rate * min(VIDEO_DURATION_RULES[model])
        for model, rate in VIDEO_TOKEN_RATES.items()
        if VIDEO_DURATION_RULES.get(model)
    ]
    for model, res_table in SEEDANCE2_TOKEN_RATES["normal"].items():
        durations = VIDEO_DURATION_RULES.get(model)
        if durations:
            costs.extend(rate * min(durations) for rate in res_table.values())
    for model, res_table in KLING_BASE_RATES_BY_MODEL.items():
        durations = VIDEO_DURATION_RULES.get(model)
        if durations:
            costs.extend(rate * min(durations) for rate in res_table.values())
    return min(costs) if costs else 0


# ---------------------------------------------------------------------------
# Language heuristic + insufficient-balance message (code-generated, not LLM)
# ---------------------------------------------------------------------------
_SPANISH_MARKERS = (
    "¿", "¡", "costo", "costará", "duración", "duracion", "confirmas", "confirmar",
    "genera", "imagen", "vídeo", "segundos", "resolución", "resolucion",
    " el ", " la ", " los ", " las ", " para ", " con ", " tu ",
)

def is_spanish(text: str) -> bool:
    """Keyword heuristic to pick the message language. Defaults to English."""
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _SPANISH_MARKERS)


# Dedicated, scoring-based marker sets for detect_language. Unlike
# _SPANISH_MARKERS (bare substring match, used only by the is_spanish
# fallback) these are matched on WORD BOUNDARIES so e.g. "crea"/"genera" can't
# falsely fire inside the English words "create"/"generate". Strong signals
# (accents, ¿/¡/ñ) never occur in English, so any hit is decisive. Ambiguous
# tokens shared by both languages ("a", "video", "no") are deliberately in
# neither set.
_ES_STRONG = ("¿", "¡", "ñ", "á", "é", "í", "ó", "ú")
_ES_WORDS = (
    "el", "la", "los", "las", "un", "una", "para", "con", "tu", "que", "por",
    "del", "de", "es", "mas", "esto", "este", "costo", "imagen", "genera",
    "segundos", "quiero", "necesito", "puedes", "hazme", "crea", "dame",
    "confirmas", "confirmar", "cuesta",
)
_EN_WORDS = (
    "the", "with", "your", "you", "for", "and", "please", "what", "would",
    "want", "generate", "create", "make", "yes", "okay", "thanks", "thank",
    "how", "where", "this", "that", "give", "more", "buy", "edit", "seconds",
    "i", "to", "of", "in", "is", "my", "can", "do", "me",
)


def detect_language(text: str) -> Optional[str]:
    """
    Best-effort, tri-state language detection for a SINGLE message.

    Returns 'es' or 'en' when one language clearly dominates, or None when the
    text carries no clear signal (e.g. a bare model name "veo 3.1", a number,
    "ok", an empty string). None lets the caller fall back to the running
    conversation language instead of forcing a switch on an ambiguous reply.
    """
    lowered = (text or "").strip().lower()
    if not lowered:
        return None
    padded = f" {lowered} "
    es = sum(1 for m in _ES_STRONG if m in lowered)
    es += sum(1 for w in _ES_WORDS if f" {w} " in padded)
    en = sum(1 for w in _EN_WORDS if f" {w} " in padded)
    if es > en:
        return "es"
    if en > es:
        return "en"
    return None


# Stable openings of the code-generated insufficient-balance message in each
# language. Used to recognise the block in conversation history so the agent
# can be made aware it was shown (the block bypasses Gemini's chat session).
_INSUFFICIENT_BALANCE_HEADERS = (
    "no tienes tokens suficientes",
    "you don't have enough tokens",
)


def is_insufficient_balance_message(text: str) -> bool:
    """True if `text` is (the start of) a code-generated balance-block message."""
    lowered = (text or "").lower()
    return any(header in lowered for header in _INSUFFICIENT_BALANCE_HEADERS)


_MAX_LISTED_VIDEOS = 4


def build_insufficient_balance_message(
    required: int,
    balance: int,
    lang: str,
    options: Dict,
) -> str:
    """Build the user-facing message shown when the balance can't cover a generation."""
    images = options.get("images", [])
    videos = options.get("videos", [])[:_MAX_LISTED_VIDEOS]
    speech_max_chars = options.get("speech_max_chars", 0)
    has_alternatives = bool(images or videos or speech_max_chars)

    if lang == "es":
        lines = [
            "⚠️ No tienes tokens suficientes para esta generación.",
            f"• Costo: {required} tokens   • Tu saldo: {balance} tokens",
            "",
        ]
        if has_alternatives:
            lines.append("Con tu saldo actual puedes generar, por ejemplo:")
            if videos:
                video_parts = [
                    f"{v['model']}{' ' + v['resolution'] if v['resolution'] else ''} "
                    f"hasta {v['max_duration']}s ({v['cost']} tokens)"
                    for v in videos
                ]
                lines.append(f"• Video: {', '.join(video_parts)}")
            if images:
                image_parts = [f"{i['model']} ({i['cost']})" for i in images]
                lines.append(f"• Imagen: {', '.join(image_parts)}")
            if speech_max_chars:
                lines.append(f"• Voz: hasta {speech_max_chars:,} caracteres")
            lines.append("")
            lines.append(
                "¿Quieres ajustar el modelo, la duración o la resolución? "
                "También puedes recargar tokens (+10% extra en recargas)."
            )
        else:
            lines.append(
                "Tu saldo no alcanza para ninguna generación por ahora. "
                "Puedes recargar tokens para continuar (+10% extra en recargas)."
            )
        return "\n".join(lines)

    lines = [
        "⚠️ You don't have enough tokens for this generation.",
        f"• Cost: {required} tokens   • Your balance: {balance} tokens",
        "",
    ]
    if has_alternatives:
        lines.append("With your current balance you could generate, for example:")
        if videos:
            video_parts = [
                f"{v['model']}{' ' + v['resolution'] if v['resolution'] else ''} "
                f"up to {v['max_duration']}s ({v['cost']} tokens)"
                for v in videos
            ]
            lines.append(f"• Video: {', '.join(video_parts)}")
        if images:
            image_parts = [f"{i['model']} ({i['cost']})" for i in images]
            lines.append(f"• Image: {', '.join(image_parts)}")
        if speech_max_chars:
            lines.append(f"• Speech: up to {speech_max_chars:,} characters")
        lines.append("")
        lines.append(
            "Would you like to adjust the model, duration, or resolution? "
            "You can also top up tokens (+10% bonus on top-ups)."
        )
    else:
        lines.append(
            "Your balance isn't enough for any generation right now. "
            "You can top up tokens to continue (+10% bonus on top-ups)."
        )
    return "\n".join(lines)
