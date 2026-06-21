"""
Unit tests for the async (202 + poll) video generation path in tools.py:
classify_poll_result (pure), _build_status_url (pure), and the
_poll_generation_status loop. No network or Redis — httpx, asyncio.sleep, and
the session manager are faked, so the loop runs instantly and deterministically.
"""
import asyncio
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import tools
from tools import classify_poll_result, _build_status_url, _poll_generation_status
from generation_errors import GENERATION_ERROR_PREFIX, parse_generation_error


# ---------------------------------------------------------------------------
# classify_poll_result (pure)
# ---------------------------------------------------------------------------
class TestClassifyPollResult:
    def test_completed_returns_result_url(self):
        out = classify_poll_result({"status": "completed", "result_url": "https://x/v.mp4"})
        assert out == {"state": "completed", "result_url": "https://x/v.mp4"}

    def test_completed_without_url(self):
        out = classify_poll_result({"status": "completed"})
        assert out["state"] == "completed"
        assert out["result_url"] is None

    def test_failed_carries_error_and_refund(self):
        out = classify_poll_result({"status": "failed", "error": "boom", "refunded": True})
        assert out == {"state": "failed", "error": "boom", "refunded": True}

    def test_failed_defaults_when_fields_absent(self):
        out = classify_poll_result({"status": "failed"})
        assert out["state"] == "failed"
        assert out["refunded"] is False
        assert out["error"]  # non-empty default message

    @pytest.mark.parametrize("status", ["queued", "processing", "", "weird"])
    def test_non_terminal_is_pending(self, status):
        assert classify_poll_result({"status": status})["state"] == "pending"

    @pytest.mark.parametrize("value", [None, "not a dict", 42, []])
    def test_non_dict_is_pending(self, value):
        assert classify_poll_result(value)["state"] == "pending"


# ---------------------------------------------------------------------------
# _build_status_url (pure)
# ---------------------------------------------------------------------------
class TestBuildStatusUrl:
    def test_prefers_relative_poll_url(self):
        url = _build_status_url(
            "https://api.test",
            {"poll_url": "/api/ai/generation-status/abc", "generation_id": "abc"},
        )
        assert url == "https://api.test/api/ai/generation-status/abc"

    def test_absolute_poll_url_used_as_is(self):
        url = _build_status_url("https://api.test", {"poll_url": "https://cdn.test/status/abc"})
        assert url == "https://cdn.test/status/abc"

    def test_falls_back_to_generation_id(self):
        url = _build_status_url("https://api.test", {"generation_id": "xyz"})
        assert url == "https://api.test/api/ai/generation-status/xyz"

    def test_missing_both_returns_none(self):
        assert _build_status_url("https://api.test", {}) is None


# ---------------------------------------------------------------------------
# Fakes for the polling loop
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error",
                request=httpx.Request("GET", "http://test/status"),
                response=httpx.Response(self.status_code, text=self.text),
            )

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Returns scripted responses, repeating the LAST one once exhausted.

    The "repeat last" rule guarantees the polling loop always terminates in a
    test: the final scripted response must be a terminal (completed/failed) or
    error state, never an endless 'pending'.
    """

    def __init__(self, responses):
        self._responses = list(responses)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, headers=None):
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


def _run_poll(responses, monotonic=None):
    sm = AsyncMock()
    fake = _FakeAsyncClient(responses)
    patches = [
        patch.object(tools.httpx, "AsyncClient", lambda *a, **k: fake),
        patch.object(tools.asyncio, "sleep", AsyncMock()),
    ]
    if monotonic is not None:
        patches.append(patch.object(tools.time, "monotonic", monotonic))

    for p in patches:
        p.start()
    try:
        result = asyncio.run(
            _poll_generation_status(
                "https://api.test/api/ai/generation-status/abc",
                {"Authorization": "Bearer t"},
                "video",
                "runway-4.5",
                "conv-1",
                sm,
            )
        )
    finally:
        for p in patches:
            p.stop()
    return result, sm


# ---------------------------------------------------------------------------
# _poll_generation_status (loop)
# ---------------------------------------------------------------------------
class TestPollGenerationStatus:
    def test_completes_and_saves_result(self):
        responses = [
            _FakeResponse({"status": "processing"}),
            _FakeResponse({"status": "completed", "result_url": "https://gcs/v.mp4"}),
        ]
        result, sm = _run_poll(responses)
        assert result == "Video generated successfully with runway-4.5."
        sm.save_generated_file.assert_awaited_once_with("conv-1", "https://gcs/v.mp4", "video")
        sm.clear_reference_files.assert_awaited_once_with("conv-1")

    def test_failed_with_refund_is_reported(self):
        responses = [_FakeResponse({"status": "failed", "error": "content blocked", "refunded": True})]
        result, sm = _run_poll(responses)
        assert result.startswith(GENERATION_ERROR_PREFIX)
        parsed = parse_generation_error(result)
        assert parsed["type"] == "video"
        assert "content blocked" in parsed["detail"]
        assert "refunded" in parsed["detail"].lower()
        sm.save_generated_file.assert_not_awaited()

    def test_completed_without_url_is_error(self):
        responses = [_FakeResponse({"status": "completed"})]
        result, sm = _run_poll(responses)
        parsed = parse_generation_error(result)
        assert parsed["category"] == "unknown"
        sm.save_generated_file.assert_not_awaited()

    def test_auth_error_fails_fast(self):
        responses = [_FakeResponse({"message": "Unauthenticated"}, status_code=401)]
        result, sm = _run_poll(responses)
        parsed = parse_generation_error(result)
        assert parsed["status"] == 401
        assert parsed["category"] == "auth"
        sm.save_generated_file.assert_not_awaited()

    def test_timeout_when_never_completes(self):
        # monotonic: 1st call sets the deadline (=0+900); the next call jumps
        # past it so the loop exits via the timeout branch after one poll.
        times = iter([0.0, 1000.0])
        fake_monotonic = lambda: next(times, 9999.0)
        responses = [_FakeResponse({"status": "processing"})]
        result, sm = _run_poll(responses, monotonic=fake_monotonic)
        parsed = parse_generation_error(result)
        assert parsed["category"] == "timeout"
        sm.save_generated_file.assert_not_awaited()
