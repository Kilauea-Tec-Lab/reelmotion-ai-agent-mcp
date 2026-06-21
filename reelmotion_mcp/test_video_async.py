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

    def test_is_processing_rejects_errors_and_success(self):
        assert is_generation_processing("GENERATION_ERROR | type=video | ...") is False
        assert is_generation_processing("Video generated successfully with runway-4.5.") is False
        assert is_generation_processing("") is False
        assert is_generation_processing(None) is False

    def test_processing_message_is_localized(self):
        es = processing_message("es")
        en = processing_message("en")
        assert "notificación" in es
        assert "notification" in en
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

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        return self._response


def _run_generate_video(response, model="runway-4.5", duration=5):
    sm = AsyncMock()
    sm.get_reference_files = AsyncMock(return_value=[])
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
        result = asyncio.run(tools.generate_video("a sunset over the sea", model, duration))
    return result, sm


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
        result, sm = _run_generate_video(_FakeResponse(body, 200))
        assert result == "Video generated successfully with runway-4.5."
        sm.save_generated_file.assert_awaited_once_with("conv-1", "https://gcs/v.mp4", "video")
        sm.clear_reference_files.assert_awaited_once_with("conv-1")

    def test_202_returns_processing_marker_without_saving(self):
        body = {
            "success": True,
            "status": "processing",
            "generation_id": "uuid",
            "poll_url": "/api/ai/generation-status/uuid",
            "estimated_tokens": 65,
        }
        result, sm = _run_generate_video(_FakeResponse(body, 202))
        assert is_generation_processing(result)
        # 202 is NOT a finished result — nothing is saved to the library.
        sm.save_generated_file.assert_not_awaited()
        sm.clear_reference_files.assert_not_awaited()

    def test_422_failed_surfaces_error_and_refund(self):
        body = {
            "success": False,
            "status": "failed",
            "generation_id": "uuid",
            "error": "content blocked by provider",
            "refunded": True,
        }
        result, sm = _run_generate_video(_FakeResponse(body, 422))
        parsed = parse_generation_error(result)
        assert parsed["type"] == "video"
        assert parsed["status"] == 422
        assert "content blocked by provider" in parsed["detail"]
        assert "refunded" in parsed["detail"].lower()
        sm.save_generated_file.assert_not_awaited()

    def test_422_failed_without_refund_flag_omits_refund_note(self):
        body = {"success": False, "status": "failed", "error": "boom", "refunded": False}
        result, sm = _run_generate_video(_FakeResponse(body, 422))
        parsed = parse_generation_error(result)
        assert "boom" in parsed["detail"]
        assert "refunded" not in parsed["detail"].lower()

    def test_402_insufficient_tokens_is_classified(self):
        body = {"message": "Insufficient token balance"}
        result, sm = _run_generate_video(_FakeResponse(body, 402))
        parsed = parse_generation_error(result)
        assert parsed["category"] == "insufficient_tokens"
        assert parsed["status"] == 402
        sm.save_generated_file.assert_not_awaited()

    def test_200_without_url_does_not_claim_success(self):
        result, sm = _run_generate_video(_FakeResponse({"status": "completed"}, 200))
        assert "not immediately available" in result.lower()
        sm.save_generated_file.assert_not_awaited()
