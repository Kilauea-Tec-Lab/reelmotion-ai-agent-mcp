"""
Unit tests for the hybrid (200 sync / 202 processing) video generation path in
tools.generate_video, plus the GENERATION_PROCESSING helpers in
generation_errors. The backend branch is decided by HTTP STATUS, never the
provider name. No network or Redis — httpx, the session manager, and the request
context are faked, so the tool runs instantly and deterministically.
"""
import asyncio
import json
import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import tools
from generation_errors import (
    GENERATION_PROCESSING_PREFIX,
    format_generation_processing,
    generation_processing_id,
    generation_processing_type,
    is_generation_processing,
    parse_generation_error,
    processing_message,
)


# ---------------------------------------------------------------------------
# GENERATION_PROCESSING helpers (pure)
# ---------------------------------------------------------------------------
class TestProcessingHelpers:
    def test_format_round_trips_through_is_processing(self):
        marker = format_generation_processing("video")
        assert marker.startswith(GENERATION_PROCESSING_PREFIX)
        assert is_generation_processing(marker) is True

    def test_marker_carries_optional_generation_id(self):
        # Without an id the marker still parses; with one, the id is extractable
        # and the type is unaffected (regex is order-independent).
        assert generation_processing_id(format_generation_processing("video")) is None
        marker = format_generation_processing("video", "gen-123")
        assert is_generation_processing(marker) is True
        assert generation_processing_type(marker) == "video"
        assert generation_processing_id(marker) == "gen-123"

    def test_is_processing_rejects_errors_and_success(self):
        assert is_generation_processing("GENERATION_ERROR | type=video | ...") is False
        assert is_generation_processing("Video generated successfully with runway-4.5.") is False
        assert is_generation_processing("") is False
        assert is_generation_processing(None) is False

    def test_processing_message_is_localized(self):
        es = processing_message("es")
        en = processing_message("en")
        assert "aparecerá" in es
        assert "appear" in en
        assert "chat" in es.lower()
        assert es != en

    def test_processing_message_defaults_to_english(self):
        assert processing_message("fr") == processing_message("en")


# ---------------------------------------------------------------------------
# Fakes for generate_video
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
    def __init__(self, response):
        self._response = response
        self.last_json = None  # capture the posted payload for assertions

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        self.last_json = json
        return self._response


def _run_generate_video(response, model="runway-4.5", duration=5, reference_files=None, **kwargs):
    sm = AsyncMock()
    sm.get_reference_files = AsyncMock(return_value=reference_files or [])
    fake_client = _FakeAsyncClient(response)
    env = {
        "BACKEND_URL": "https://api.test",
        "VIDEO_CREATION_ENDPOINT": "/api/ai/mcp-video-generation",
    }
    with patch.dict(os.environ, env, clear=False), \
         patch.object(tools, "get_api_token", lambda: "tok"), \
         patch.object(tools, "get_conversation_uuid", lambda: "conv-1"), \
         patch.object(tools, "get_session_manager", lambda: sm), \
         patch.object(tools.httpx, "AsyncClient", lambda *a, **k: fake_client):
        result = asyncio.run(
            tools.generate_video("a sunset over the sea", model, duration, **kwargs)
        )
    return result, sm, fake_client


# ---------------------------------------------------------------------------
# generate_video — branch by HTTP STATUS
# ---------------------------------------------------------------------------
class TestGenerateVideoHybrid:
    def test_sync_200_saves_and_succeeds(self):
        body = {
            "success": True,
            "status": "completed",
            "video_url": "https://gcs/v.mp4",
            "tokens_used": 65,
        }
        result, sm, _ = _run_generate_video(_FakeResponse(body, 200))
        assert result == "Video generated successfully with runway-4.5."
        sm.save_generated_file.assert_awaited_once_with("conv-1", "https://gcs/v.mp4", "video")
        sm.clear_reference_files.assert_awaited_once_with("conv-1")

    def test_202_returns_processing_marker_with_generation_id(self):
        body = {
            "success": True,
            "status": "processing",
            "generation_id": "uuid",
            "poll_url": "/api/ai/generation-status/uuid",
            "estimated_tokens": 65,
        }
        result, sm, _ = _run_generate_video(_FakeResponse(body, 202))
        assert is_generation_processing(result)
        assert generation_processing_id(result) == "uuid"
        # The id is stashed for the chat card, but no finished file is saved.
        sm.save_pending_generation.assert_awaited_once_with("conv-1", "uuid", "video")
        sm.save_generated_file.assert_not_awaited()
        sm.clear_reference_files.assert_not_awaited()

    def test_202_without_generation_id_still_processes(self):
        body = {"success": True, "status": "processing"}
        result, sm, _ = _run_generate_video(_FakeResponse(body, 202))
        assert is_generation_processing(result)
        assert generation_processing_id(result) is None
        sm.save_pending_generation.assert_not_awaited()

    def test_payload_includes_chat_id(self):
        body = {"success": True, "status": "processing", "generation_id": "uuid"}
        _, _, client = _run_generate_video(_FakeResponse(body, 202))
        # get_chat_id() is unset in tests, so it falls back to the conversation uuid.
        assert client.last_json["chat_id"] == "conv-1"

    def test_422_failed_surfaces_error_and_refund(self):
        body = {
            "success": False,
            "status": "failed",
            "generation_id": "uuid",
            "error": "content blocked by provider",
            "refunded": True,
        }
        result, sm, _ = _run_generate_video(_FakeResponse(body, 422))
        parsed = parse_generation_error(result)
        assert parsed["type"] == "video"
        assert parsed["status"] == 422
        assert "content blocked by provider" in parsed["detail"]
        assert "refunded" in parsed["detail"].lower()
        sm.save_generated_file.assert_not_awaited()

    def test_422_failed_without_refund_flag_omits_refund_note(self):
        body = {"success": False, "status": "failed", "error": "boom", "refunded": False}
        result, sm, _ = _run_generate_video(_FakeResponse(body, 422))
        parsed = parse_generation_error(result)
        assert "boom" in parsed["detail"]
        assert "refunded" not in parsed["detail"].lower()

    def test_402_insufficient_tokens_is_classified(self):
        body = {"message": "Insufficient token balance"}
        result, sm, _ = _run_generate_video(_FakeResponse(body, 402))
        parsed = parse_generation_error(result)
        assert parsed["category"] == "insufficient_tokens"
        assert parsed["status"] == 402
        sm.save_generated_file.assert_not_awaited()

    def test_200_without_url_does_not_claim_success(self):
        result, sm, _ = _run_generate_video(_FakeResponse({"status": "completed"}, 200))
        assert "not immediately available" in result.lower()
        sm.save_generated_file.assert_not_awaited()


# ---------------------------------------------------------------------------
# Payload contract — the backend now reads `provider` + Evolink routing fields
# ---------------------------------------------------------------------------
_OK = {"success": True, "status": "completed", "video_url": "https://gcs/v.mp4"}


class TestVideoPayloadContract:
    def test_model_is_sent_as_provider_not_ai_model(self):
        _, _, client = _run_generate_video(_FakeResponse(_OK, 200), model="kling-v3", duration=5)
        assert client.last_json["provider"] == "kling-v3"
        assert "ai_model" not in client.last_json
        assert client.last_json["video_duration"] == 5

    def test_kling_base_text_to_video_quality_and_sound(self):
        _, _, client = _run_generate_video(
            _FakeResponse(_OK, 200), model="kling-v3", duration=5, resolution="1080p"
        )
        body = client.last_json
        assert body["provider"] == "kling-v3"
        assert body["quality"] == "1080p"
        assert body["sound"] == "off"            # audio opt-in, default off
        assert "media_url" not in body           # text-to-video

    def test_kling_v3_4k_kept_on_base_route(self):
        _, _, client = _run_generate_video(
            _FakeResponse(_OK, 200), model="kling-v3", duration=4, quality="4k"
        )
        assert client.last_json["quality"] == "4k"

    def test_kling_turbo_4k_downgraded_and_no_audio(self):
        _, _, client = _run_generate_video(
            _FakeResponse(_OK, 200), model="kling-v3-turbo", duration=5,
            quality="4k", sound="on",
        )
        body = client.last_json
        assert body["quality"] == "1080p"        # turbo has no 4K
        assert body["sound"] == "off"            # audio ignored off the base route

    def test_kling_v3_image_to_video_uses_media_url(self):
        refs = [{"type": "image", "url": "https://cdn/x.jpg"}]
        _, _, client = _run_generate_video(
            _FakeResponse(_OK, 200), model="kling-v3", duration=5, reference_files=refs
        )
        assert client.last_json["media_url"] == "https://cdn/x.jpg"

    def test_kling_v3_guide_video_triggers_motion(self):
        refs = [{"type": "video", "url": "https://cdn/guide.mp4"}]
        _, _, client = _run_generate_video(
            _FakeResponse(_OK, 200), model="kling-v3", duration=5, reference_files=refs
        )
        body = client.last_json
        assert body["motion_video"] == "https://cdn/guide.mp4"
        assert "media_url" not in body           # guide video must NOT go in media_url for v3

    def test_kling_o3_video_triggers_edit(self):
        refs = [{"type": "video", "url": "https://cdn/source.mp4"}]
        _, _, client = _run_generate_video(
            _FakeResponse(_OK, 200), model="kling-o3", duration=6,
            resolution="1080p", reference_files=refs,
        )
        body = client.last_json
        assert body["edit_video"] == "https://cdn/source.mp4"
        assert body["quality"] == "1080p"

    def test_kling_o3_reference_images_trigger_reference(self):
        _, _, client = _run_generate_video(
            _FakeResponse(_OK, 200), model="kling-o3", duration=5,
            reference_images=["https://cdn/a.jpg", "https://cdn/b.jpg"],
        )
        assert client.last_json["reference_images"] == ["https://cdn/a.jpg", "https://cdn/b.jpg"]

    def test_invalid_provider_is_rejected(self):
        result, _, client = _run_generate_video(
            _FakeResponse(_OK, 200), model="kling-v3-omni-std", duration=5
        )
        assert result.lower().startswith("error")
        assert client.last_json is None          # never posted
