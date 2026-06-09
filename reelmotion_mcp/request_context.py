from contextvars import ContextVar
from typing import Optional
import base64

_api_token_ctx: ContextVar[Optional[str]] = ContextVar("api_token", default=None)
_reference_images_ctx: ContextVar[Optional[list]] = ContextVar("reference_images", default=None)
_conversation_uuid_ctx: ContextVar[Optional[str]] = ContextVar("conversation_uuid", default=None)
# User token balance sent by Laravel on each request. None = unknown (never block).
_token_balance_ctx: ContextVar[Optional[int]] = ContextVar("token_balance", default=None)
_chat_id_ctx: ContextVar[Optional[str]] = ContextVar("chat_id", default=None)
# Set when a generation is blocked for insufficient balance: {"required": int, "available": int}
_insufficient_block_ctx: ContextVar[Optional[dict]] = ContextVar("insufficient_block", default=None)

def set_api_token(token: str):
    _api_token_ctx.set(token)

def get_api_token() -> Optional[str]:
    return _api_token_ctx.get()

def set_token_balance(balance: Optional[int]):
    _token_balance_ctx.set(balance)

def get_token_balance() -> Optional[int]:
    return _token_balance_ctx.get()

def set_chat_id(chat_id: str):
    _chat_id_ctx.set(chat_id)

def get_chat_id() -> Optional[str]:
    return _chat_id_ctx.get()

def set_insufficient_block(info: dict):
    _insufficient_block_ctx.set(info)

def get_insufficient_block() -> Optional[dict]:
    return _insufficient_block_ctx.get()

def clear_insufficient_block():
    _insufficient_block_ctx.set(None)

def set_conversation_uuid(uuid: str):
    _conversation_uuid_ctx.set(uuid)

def get_conversation_uuid() -> Optional[str]:
    return _conversation_uuid_ctx.get()

def set_reference_images(images: list):
    """Store reference images as base64 strings."""
    base64_images = []
    for img in images:
        if isinstance(img, dict) and "data" in img:
            # Convert bytes to base64 string
            b64 = base64.b64encode(img["data"]).decode("utf-8")
            mime_type = img.get("mime_type", "image/jpeg")
            base64_images.append(f"data:{mime_type};base64,{b64}")
    _reference_images_ctx.set(base64_images)

def get_reference_images() -> Optional[list]:
    return _reference_images_ctx.get()

def clear_reference_images():
    _reference_images_ctx.set(None)
