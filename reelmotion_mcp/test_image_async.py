"""
Unit tests for the image generation contract in tools.generate_image:
the new model catalog (Seedream / GPT / Nano Banana 2 / Midjourney, no Freepik),
per-model payload rules, reference handling, and the hybrid (200 completed /
202 processing / 422 failed / 402 insufficient) delivery path. The backend branch
is decided by HTTP STATUS, never the model name. No network or Redis — httpx, the
session manager, and the request context are faked, so the tool runs instantly
and deterministically.
"""
import asyncio
import json
import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import tools
from tools import normalize_image_model, _extract_image_urls
from generation_errors import (
    generation_processing_id,
    generation_processing_type,
    is_generation_processing,
    parse_generation_error,
    processing_message,
)


# ---------------------------------------------------------------------------
# normalize_image_model (pure)
# ---------------------------------------------------------------------------
class TestNormalizeImageModel:
    @pytest.mark.parametrize("raw,expected", [
        ("Seedream", "Seedream"),
        ("seedream 4.0", "Seedream"),
        ("Midjourney", "Midjourney"),
        ("midjourney v6", "Midjourney"),
        ("mj", "Midjourney"),
        ("Nano Banana 2", "Nano Banana 2"),
        ("nano-banana", "Nano Banana 2"),
        ("banana", "Nano Banana 2"),
        ("GPT", "GPT"),
        ("gpt-image", "GPT"),
    ])
    def test_known_models_normalize(self, raw, expected):
        assert normalize_image_model(raw) == expected

    @pytest.mark.parametrize("raw", ["Freepik", "freepik", "dall-e", "stable-diffusion", ""])
    def test_unknown_models_return_none(self, raw):
        assert normalize_image_model(raw) is None


# ---------------------------------------------------------------------------
# _extract_image_urls (pure)
# ---------------------------------------------------------------------------
class TestExtractImageUrls:
    def test_primary_image_url(self):
        assert _extract_image_urls({"image_url": "https://x/a.png"}) == ["https://x/a.png"]

    def test_midjourney_primary_first_then_variants_deduped(self):
        urls = _extract_image_urls({
            "image_url": "https://x/a.png",
            "all_result_urls": ["https://x/a.png", "https://x/b.png", "https://x/c.png"],
        })
        assert urls[0] == "https://x/a.png"
        assert urls == ["https://x/a.png", "https://x/b.png", "https://x/c.png"]

    def test_legacy_images_list(self):
        urls = _extract_image_urls({"images": [{"url": "https://x/a.png"}, "https://x/b.png"]})
        assert urls == ["https://x/a.png", "https://x/b.png"]

    def test_no_urls(self):
        assert _extract_image_urls({"status": "completed"}) == []


# ---------------------------------------------------------------------------
# processing message (image variant)
# ---------------------------------------------------------------------------
class TestImageProcessingMessage:
    def test_processing_type_parsed_from_marker(self):
        from generation_errors import format_generation_processing
        assert generation_processing_type(format_generation_processing("image")) == "image"
        assert generation_processing_type(format_generation_processing("video")) == "video"
        assert generation_processing_type("nonsense") == "video"

    def test_image_message_mentions_image_and_differs_from_video(self):
        en_image = processing_message("en", "image")
        en_video = processing_message("en", "video")
        assert "image" in en_image.lower()
        assert en_image != en_video

    def test_image_message_localized(self):
        es = processing_message("es", "image")
        assert "imagen" in es.lower()
        assert "chat" in es.lower()


# ---------------------------------------------------------------------------
# Fakes for generate_image
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error",
                request=httpx.Request("POST", "https://api.test/x"),
                response=httpx.Response(self.status_code, json=self._payload),
            )


class _FakeAsyncClient:
    def __init__(self, response, sink):
        self._response = response
        self._sink = sink

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        self._sink["url"] = url
        self._sink["payload"] = json
        self._sink["headers"] = headers
        return self._response


def _run_generate_image(response, model="Seedream", ref_files=None, prompt="a photoreal pug", **kwargs):
    sm = AsyncMock()
    sm.get_reference_files = AsyncMock(return_value=ref_files or [])
    sink = {}
    fake_client = _FakeAsyncClient(response, sink)
    env = {
        "BACKEND_URL": "https://api.test",
        "IMAGE_CREATION_ENDPOINT": "/api/ai/mcp-image-generation",
    }
    with patch.dict(os.environ, env, clear=False), \
         patch.object(tools, "get_api_token", lambda: "tok"), \
         patch.object(tools, "get_conversation_uuid", lambda: "conv-1"), \
         patch.object(tools, "get_session_manager", lambda: sm), \
         patch.object(tools.httpx, "AsyncClient", lambda *a, **k: fake_client):
        result = asyncio.run(tools.generate_image(prompt, model, **kwargs))
    return result, sm, sink


# ---------------------------------------------------------------------------
# generate_image — branch by HTTP STATUS
# ---------------------------------------------------------------------------
class TestGenerateImageHybrid:
    def test_sync_200_saves_primary_and_succeeds(self):
        body = {"success": True, "status": "completed", "image_url": "https://gcs/a.png"}
        result, sm, _ = _run_generate_image(_FakeResponse(body, 200))
        assert result == "Images generated successfully with Seedream."
        sm.save_generated_file.assert_awaited_once_with("conv-1", "https://gcs/a.png", "image")
        sm.clear_reference_files.assert_awaited_once_with("conv-1")

    def test_midjourney_saves_only_primary_not_variants(self):
        body = {
            "status": "completed",
            "image_url": "https://gcs/primary.png",
            "all_result_urls": [
                "https://gcs/primary.png", "https://gcs/v2.png",
                "https://gcs/v3.png", "https://gcs/v4.png",
            ],
        }
        result, sm, _ = _run_generate_image(_FakeResponse(body, 200), model="Midjourney")
        assert "Midjourney" in result
        sm.save_generated_file.assert_awaited_once_with("conv-1", "https://gcs/primary.png", "image")

    def test_legacy_images_list_is_saved(self):
        body = {"images": [{"url": "https://gcs/a.png"}]}
        result, sm, _ = _run_generate_image(_FakeResponse(body, 200), model="GPT")
        sm.save_generated_file.assert_awaited_once_with("conv-1", "https://gcs/a.png", "image")

    def test_202_returns_processing_marker_with_generation_id(self):
        body = {"success": True, "status": "processing", "generation_id": "uuid"}
        result, sm, _ = _run_generate_image(_FakeResponse(body, 202))
        assert is_generation_processing(result)
        assert generation_processing_type(result) == "image"
        assert generation_processing_id(result) == "uuid"
        sm.save_pending_generation.assert_awaited_once_with("conv-1", "uuid", "image")
        sm.save_generated_file.assert_not_awaited()
        sm.clear_reference_files.assert_not_awaited()

    def test_202_without_generation_id_still_processes(self):
        body = {"success": True, "status": "processing"}
        result, sm, _ = _run_generate_image(_FakeResponse(body, 202))
        assert is_generation_processing(result)
        assert generation_processing_id(result) is None
        sm.save_pending_generation.assert_not_awaited()

    def test_payload_includes_chat_id(self):
        body = {"image_url": "https://gcs/a.png"}
        _, _, sink = _run_generate_image(_FakeResponse(body, 200))
        # get_chat_id() is unset in tests, so it falls back to the conversation uuid.
        assert sink["payload"]["chat_id"] == "conv-1"

    def test_422_failed_surfaces_error_and_refund(self):
        body = {
            "success": False, "status": "failed",
            "error": "content blocked by provider", "refunded": True,
        }
        result, sm, _ = _run_generate_image(_FakeResponse(body, 422))
        parsed = parse_generation_error(result)
        assert parsed["type"] == "image"
        assert parsed["status"] == 422
        assert "content blocked by provider" in parsed["detail"]
        assert "refunded" in parsed["detail"].lower()
        sm.save_generated_file.assert_not_awaited()

    def test_422_failed_without_refund_flag_omits_refund_note(self):
        body = {"success": False, "status": "failed", "error": "boom", "refunded": False}
        result, _, _ = _run_generate_image(_FakeResponse(body, 422))
        parsed = parse_generation_error(result)
        assert "boom" in parsed["detail"]
        assert "refunded" not in parsed["detail"].lower()

    def test_402_insufficient_tokens_is_classified(self):
        body = {"message": "Insufficient token balance"}
        result, sm, _ = _run_generate_image(_FakeResponse(body, 402))
        parsed = parse_generation_error(result)
        assert parsed["category"] == "insufficient_tokens"
        assert parsed["status"] == 402
        sm.save_generated_file.assert_not_awaited()

    def test_200_without_url_does_not_claim_success(self):
        result, sm, _ = _run_generate_image(_FakeResponse({"status": "completed"}, 200))
        assert "not immediately available" in result.lower()
        sm.save_generated_file.assert_not_awaited()

    def test_invalid_model_is_rejected(self):
        result, sm, _ = _run_generate_image(_FakeResponse({}, 200), model="dall-e")
        assert result.startswith("Error: Invalid model")
        sm.save_generated_file.assert_not_awaited()


# ---------------------------------------------------------------------------
# generate_image — payload composition per model
# ---------------------------------------------------------------------------
class TestGenerateImagePayload:
    def test_seedream_payload_has_quality_no_type_quantity(self):
        body = {"image_url": "https://gcs/a.png"}
        _, _, sink = _run_generate_image(_FakeResponse(body, 200), model="Seedream", quality="3K")
        payload = sink["payload"]
        assert payload["model"] == "Seedream"
        assert payload["aspect_ratio"] == "16:9"
        assert payload["quality"] == "3K"
        assert "type" not in payload
        assert "quantity" not in payload

    def test_gpt_payload_has_type_quantity_no_quality(self):
        body = {"images": [{"url": "https://gcs/a.png"}]}
        _, _, sink = _run_generate_image(
            _FakeResponse(body, 200), model="GPT", image_type=1, quantity=2
        )
        payload = sink["payload"]
        assert payload["type"] == 1
        assert payload["quantity"] == 2
        assert "quality" not in payload

    def test_midjourney_payload_has_no_quality_or_quantity(self):
        body = {"image_url": "https://gcs/a.png"}
        _, _, sink = _run_generate_image(_FakeResponse(body, 200), model="Midjourney", quantity=4)
        payload = sink["payload"]
        assert "quality" not in payload
        assert "quantity" not in payload
        assert "type" not in payload

    def test_aspect_ratio_override(self):
        body = {"image_url": "https://gcs/a.png"}
        _, _, sink = _run_generate_image(_FakeResponse(body, 200), aspect_ratio="9:16")
        assert sink["payload"]["aspect_ratio"] == "9:16"


# ---------------------------------------------------------------------------
# generate_image — reference image handling
# ---------------------------------------------------------------------------
class TestGenerateImageReferences:
    def test_single_session_reference_is_attached(self):
        body = {"image_url": "https://gcs/a.png"}
        ref = [{"url": "https://x/ref.png", "type": "image"}]
        _, _, sink = _run_generate_image(_FakeResponse(body, 200), model="Seedream", ref_files=ref)
        assert sink["payload"]["reference_image"] == "https://x/ref.png"

    def test_multiple_references_use_list(self):
        body = {"image_url": "https://gcs/a.png"}
        ref = [
            {"url": "https://x/1.png", "type": "image"},
            {"url": "https://x/2.png", "type": "image"},
        ]
        _, _, sink = _run_generate_image(_FakeResponse(body, 200), model="Nano Banana 2", ref_files=ref)
        assert sink["payload"]["reference_images"] == ["https://x/1.png", "https://x/2.png"]

    def test_midjourney_drops_data_uri_references(self):
        body = {"image_url": "https://gcs/a.png"}
        ref = [{"url": "data:image/png;base64,AAAA", "type": "image"}]
        result, _, sink = _run_generate_image(_FakeResponse(body, 200), model="Midjourney", ref_files=ref)
        # data: URI is dropped for Midjourney, so it falls back to text-to-image.
        assert "reference_image" not in sink["payload"]
        assert "reference_images" not in sink["payload"]
        assert not result.startswith("Error")

    def test_seedream_keeps_data_uri_references(self):
        body = {"image_url": "https://gcs/a.png"}
        ref = [{"url": "data:image/png;base64,AAAA", "type": "image"}]
        _, _, sink = _run_generate_image(_FakeResponse(body, 200), model="Seedream", ref_files=ref)
        assert sink["payload"]["reference_image"] == "data:image/png;base64,AAAA"

    def test_blob_only_reference_errors(self):
        body = {"image_url": "https://gcs/a.png"}
        ref = [{"url": "blob:https://app/abc", "type": "image"}]
        result, sm, _ = _run_generate_image(_FakeResponse(body, 200), ref_files=ref)
        assert result.startswith("Error")
        assert "blob" in result.lower()
        sm.save_generated_file.assert_not_awaited()
