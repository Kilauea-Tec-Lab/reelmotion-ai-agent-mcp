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
              "Intenta cerrar y abrir sesión de nuevo; si persiste, contacta a soporte.",
        "en": "There was an authentication problem with the generation service. "
              "Try signing out and back in; if it persists, contact support.",
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
              "Inténtalo de nuevo; si persiste, contacta a soporte.",
        "en": "An unexpected problem occurred while generating your content. "
              "Try again; if it persists, contact support.",
    },
}


def fallback_error_message(category: str, lang: str = "en") -> str:
    """Friendly, code-generated explanation for a failure category."""
    messages = _FALLBACK_MESSAGES.get(category, _FALLBACK_MESSAGES[CATEGORY_UNKNOWN])
    return "⚠️ " + messages.get(lang, messages["en"])
