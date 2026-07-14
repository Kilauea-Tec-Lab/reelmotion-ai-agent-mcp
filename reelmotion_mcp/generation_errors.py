"""
Backend generation-error parsing and user-facing fallback messages.

The image/video generator endpoints (Laravel) return errors in several shapes
(JSON with message/error/errors keys, plain text, HTML error pages). This
module normalizes them into a stable, machine-readable string that:
  1. Gemini can interpret to explain the failure to the user (system prompt
     has instructions for the GENERATION_ERROR format), and
  2. the code can parse back to build a friendly fallback message when the
     LLM is unavailable.

Stdlib-only on purpose so tests can import it without side effects.
"""
import json
import re
from typing import Optional

GENERATION_ERROR_PREFIX = "GENERATION_ERROR"

# Marker returned when the backend accepted a generation but couldn't finish it
# within its synchronous window (HTTP 202). This is NOT an error: the video is
# still being produced and the user is notified (push/realtime) when it's ready.
# We do NOT poll — the agent just tells the user it's on the way.
GENERATION_PROCESSING_PREFIX = "GENERATION_PROCESSING"

_MAX_DETAIL_CHARS = 500

# Known categories, mapped from HTTP status codes.
CATEGORY_AUTH = "auth"
CATEGORY_INSUFFICIENT_TOKENS = "insufficient_tokens"
CATEGORY_PROVIDER_VALIDATION = "provider_validation"
CATEGORY_RATE_LIMIT = "rate_limit"
CATEGORY_BACKEND_UNAVAILABLE = "backend_unavailable"
CATEGORY_TIMEOUT = "timeout"
CATEGORY_UNKNOWN = "unknown"


def parse_backend_error(status_code: int, body_text: Optional[str]) -> dict:
    """
    Extract a clean, bounded detail message from a backend error response and
    classify it by status code.

    Returns {"category": str, "status": int, "detail": str}.
    """
    detail = ""
    if body_text:
        try:
            parsed = json.loads(body_text)
            if isinstance(parsed, dict):
                detail = str(parsed.get("message") or parsed.get("error") or "").strip()
                errors = parsed.get("errors")
                if isinstance(errors, dict):
                    flat = []
                    for value in errors.values():
                        items = value if isinstance(value, list) else [value]
                        flat.extend(str(item) for item in items)
                    if flat:
                        extra = "; ".join(flat[:5])
                        detail = f"{detail}: {extra}" if detail else extra
                if not detail:
                    detail = json.dumps(parsed)
            else:
                detail = str(parsed)
        except (ValueError, TypeError):
            # Not JSON — strip HTML tags and collapse whitespace
            detail = re.sub(r"<[^>]+>", " ", str(body_text))
            detail = re.sub(r"\s+", " ", detail).strip()

    detail = (detail or "No detail provided by the backend.")[:_MAX_DETAIL_CHARS]

    if status_code in (401, 403):
        category = CATEGORY_AUTH
    elif status_code == 402:
        category = CATEGORY_INSUFFICIENT_TOKENS
    elif status_code == 422:
        category = CATEGORY_PROVIDER_VALIDATION
    elif status_code == 429:
        category = CATEGORY_RATE_LIMIT
    elif status_code >= 500:
        category = CATEGORY_BACKEND_UNAVAILABLE
    else:
        category = CATEGORY_UNKNOWN

    return {"category": category, "status": status_code, "detail": detail}


def format_generation_error(gen_type: str, category: str, status: int, detail: str) -> str:
    """Render the stable error string returned by the generation tools."""
    detail = re.sub(r"\s+", " ", str(detail)).strip()[:_MAX_DETAIL_CHARS]
    return (
        f"{GENERATION_ERROR_PREFIX} | type={gen_type} | category={category} "
        f"| status={status} | detail={detail}"
    )


_GENERATION_ERROR_RE = re.compile(
    rf"^{GENERATION_ERROR_PREFIX}\s*\|\s*type=(?P<type>[\w-]+)\s*\|\s*category=(?P<category>[\w-]+)"
    rf"\s*\|\s*status=(?P<status>\d+)\s*\|\s*detail=(?P<detail>.*)$",
    re.DOTALL,
)


def parse_generation_error(text: str) -> Optional[dict]:
    """Parse a formatted GENERATION_ERROR string back into its fields."""
    if not text:
        return None
    match = _GENERATION_ERROR_RE.match(text.strip())
    if not match:
        return None
    return {
        "type": match.group("type"),
        "category": match.group("category"),
        "status": int(match.group("status")),
        "detail": match.group("detail").strip(),
    }


# Code-generated fallback messages, used when the LLM explanation fails.
_FALLBACK_MESSAGES = {
    CATEGORY_AUTH: {
        "es": "Hubo un problema de autenticación con el servicio de generación. "
              "Cierra y abre sesión de nuevo, y vuelve a intentarlo. No se descontaron tokens.",
        "en": "There was an authentication problem with the generation service. "
              "Sign out and back in, then try again. No tokens were charged.",
    },
    CATEGORY_INSUFFICIENT_TOKENS: {
        "es": "El servicio rechazó la generación porque tu saldo de tokens no es suficiente. "
              "Recarga tokens para continuar.",
        "en": "The service rejected the generation because your token balance isn't enough. "
              "Top up tokens to continue.",
    },
    CATEGORY_PROVIDER_VALIDATION: {
        "es": "El proveedor de IA rechazó la solicitud (posiblemente por el contenido del prompt "
              "o un parámetro no válido). Intenta reformular tu descripción y vuelve a intentarlo.",
        "en": "The AI provider rejected the request (possibly due to the prompt content or an "
              "invalid parameter). Try rephrasing your description and try again.",
    },
    CATEGORY_RATE_LIMIT: {
        "es": "El servicio está recibiendo demasiadas solicitudes en este momento. "
              "Espera un momento y vuelve a intentarlo.",
        "en": "The service is receiving too many requests right now. "
              "Wait a moment and try again.",
    },
    CATEGORY_BACKEND_UNAVAILABLE: {
        "es": "El servicio de generación no está disponible en este momento. "
              "Inténtalo de nuevo en unos minutos.",
        "en": "The generation service is unavailable right now. "
              "Please try again in a few minutes.",
    },
    CATEGORY_TIMEOUT: {
        "es": "La generación tardó demasiado y se agotó el tiempo de espera. "
              "Es posible que aún se complete; si no aparece, inténtalo de nuevo.",
        "en": "The generation took too long and timed out. "
              "It may still complete; if it doesn't show up, try again.",
    },
    CATEGORY_UNKNOWN: {
        "es": "Ocurrió un problema inesperado al generar tu contenido. "
              "Inténtalo de nuevo en un momento. Si una generación falla, tus tokens "
              "se reembolsan automáticamente, así que nunca pagas por algo que no recibiste.",
        "en": "An unexpected problem occurred while generating your content. "
              "Try again in a moment. If a generation ever fails, your tokens are "
              "refunded automatically, so you're never charged for something you didn't receive.",
    },
}


def fallback_error_message(category: str, lang: str = "en") -> str:
    """Friendly, code-generated explanation for a failure category."""
    messages = _FALLBACK_MESSAGES.get(category, _FALLBACK_MESSAGES[CATEGORY_UNKNOWN])
    return "⚠️ " + messages.get(lang, messages["en"])


# ---------------------------------------------------------------------------
# "Still processing" signal (HTTP 202) — not an error, not a finished result.
# ---------------------------------------------------------------------------
_PROCESSING_MESSAGES = {
    "video": {
        "es": "🎬 Tu video se está generando y puede tardar un poco más de lo normal. "
              "Te llegará una notificación cuando esté listo; no necesitas hacer nada más.",
        "en": "🎬 Your video is being generated and may take a little longer than usual. "
              "You'll get a notification as soon as it's ready — no further action needed.",
    },
    "image": {
        "es": "🎨 Tu imagen se está generando y puede tardar un poco más de lo normal. "
              "Te llegará una notificación cuando esté lista; no necesitas hacer nada más.",
        "en": "🎨 Your image is being generated and may take a little longer than usual. "
              "You'll get a notification as soon as it's ready — no further action needed.",
    },
}

_PROCESSING_TYPE_RE = re.compile(r"type=(?P<type>[\w-]+)")


def format_generation_processing(gen_type: str = "video") -> str:
    """Render the stable 'accepted, still processing' marker (HTTP 202)."""
    return f"{GENERATION_PROCESSING_PREFIX} | type={gen_type}"


def is_generation_processing(text: str) -> bool:
    """True if `text` is the 'still processing' marker (HTTP 202 path)."""
    return bool(text) and text.strip().startswith(GENERATION_PROCESSING_PREFIX)


def generation_processing_type(text: str) -> str:
    """Extract the gen_type from a GENERATION_PROCESSING marker (default 'video')."""
    if not text:
        return "video"
    match = _PROCESSING_TYPE_RE.search(text)
    return match.group("type") if match else "video"


def processing_message(lang: str = "en", gen_type: str = "video") -> str:
    """Friendly, localized 'your content is on the way' message (HTTP 202)."""
    by_type = _PROCESSING_MESSAGES.get(gen_type) or _PROCESSING_MESSAGES["video"]
    return by_type.get(lang) or by_type["en"]


# ---------------------------------------------------------------------------
# Synchronous success (HTTP 200) — localized, used when the result is returned
# to the user WITHOUT a further Gemini pass (confirmed pending actions, and
# fallbacks where the chat model is unavailable). The asset URL is delivered
# separately by the backend; this is just the friendly acknowledgment.
# ---------------------------------------------------------------------------
_SUCCESS_MESSAGES = {
    "video": {
        "es": "✅ ¡Tu video se generó correctamente!",
        "en": "✅ Your video was generated successfully!",
    },
    "image": {
        "es": "✅ ¡Tu imagen se generó correctamente!",
        "en": "✅ Your image was generated successfully!",
    },
    "audio": {
        "es": "✅ ¡Tu audio se generó correctamente!",
        "en": "✅ Your audio was generated successfully!",
    },
}


def success_message(lang: str = "en", gen_type: str = "image") -> str:
    """Friendly, localized 'your content is ready' message (HTTP 200 success)."""
    by_type = _SUCCESS_MESSAGES.get(gen_type) or _SUCCESS_MESSAGES["image"]
    return by_type.get(lang) or by_type["en"]
