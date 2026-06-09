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
IMAGE_COSTS: Dict[str, int] = {
    "Nano Banana 2": 7,
    "GPT": 6,
    "Freepik": 1,
}

# ---------------------------------------------------------------------------
# Video pricing — flat tokens/second models
# ---------------------------------------------------------------------------
VIDEO_TOKEN_RATES: Dict[str, int] = {
    "runway-aleph": 17,
    "runway-4.5": 14,
    "veo-3.1": 44,
    "veo-3.1-flash": 17,
    "veo-3.1-ultra": 65,
    "kling-v3-omni-pro": 26,
    "kling-v3-omni-std": 19,
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
    "kling-v1": [5, 10],
    "kling-v3-omni-pro": list(range(3, 16)),
    "kling-v3-omni-std": list(range(3, 16)),
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
    if "nano" in lowered:
        return "Nano Banana 2"
    if "gpt" in lowered:
        return "GPT"
    if "freepik" in lowered:
        return "Freepik"
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
        model = _normalize_image_model(args.get("model", "GPT"))
        if model is None:
            return None
        try:
            quantity = max(1, int(args.get("quantity", 1)))
        except (TypeError, ValueError):
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
        rate = VIDEO_TOKEN_RATES.get(model)
        if rate is None:
            return None  # legacy/unknown model (runway, luma-labs, seedance-pro, kling-v1)
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
