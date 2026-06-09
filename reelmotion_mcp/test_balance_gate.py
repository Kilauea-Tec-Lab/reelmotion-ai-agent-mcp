"""
Unit tests for the token-balance gates and the atomic pending-action claim
in chatbot.GeminiChatbot (session manager and tools are mocked — no Redis,
no network).
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import chatbot as chatbot_module
from chatbot import GeminiChatbot
from request_context import (
    clear_insufficient_block,
    get_insufficient_block,
    set_token_balance,
)


VIDEO_ACTION = {
    "function": "generate_video",
    "args": {"prompt": "a sunset", "model": "veo-3.1", "duration": 8},
    "cost_message": "Costo: 352 tokens (44 tokens/sec × 8 sec). ¿Confirmas?",
    "estimated_cost": 352,
}


def run_with_block(coro):
    """
    Await `coro` and read the insufficient-block flag INSIDE the same task.

    asyncio.run() executes in a COPY of the caller's context, so ContextVars
    set inside the coroutine aren't visible after it returns (in production
    the request handler and send_message share one task, so this isn't an
    issue there).
    """
    async def wrapper():
        result = await coro
        return result, get_insufficient_block()

    return asyncio.run(wrapper())


@pytest.fixture
def bot():
    instance = GeminiChatbot(conversation_uuid="test-balance")
    instance.session_manager = MagicMock()
    instance.session_manager.get_pending_action = AsyncMock(return_value=None)
    instance.session_manager.claim_pending_action = AsyncMock(return_value=None)
    instance.session_manager.save_pending_action = AsyncMock()
    instance.session_manager.get_reference_files = AsyncMock(return_value=[])
    yield instance
    set_token_balance(None)
    clear_insufficient_block()


class TestExecutePendingActionBalanceGate:
    def test_blocks_when_balance_below_cost_and_keeps_action(self, bot):
        bot.session_manager.get_pending_action = AsyncMock(return_value=dict(VIDEO_ACTION))
        set_token_balance(10)

        with patch.object(chatbot_module, "generate_video", new=AsyncMock()) as tool:
            (tool_result, response), block = run_with_block(bot.execute_pending_action())

        assert tool_result is None
        assert "No tienes tokens suficientes" in response  # Spanish cost_message -> es
        assert "352" in response and "10" in response
        tool.assert_not_awaited()
        # Action must NOT be claimed/deleted: user can adjust or top up
        bot.session_manager.claim_pending_action.assert_not_awaited()
        assert block == {"required": 352, "available": 10}

    def test_executes_when_balance_unknown(self, bot):
        bot.session_manager.get_pending_action = AsyncMock(return_value=dict(VIDEO_ACTION))
        bot.session_manager.claim_pending_action = AsyncMock(return_value=dict(VIDEO_ACTION))
        set_token_balance(None)

        success = "Video generated successfully with veo-3.1."
        with patch.object(chatbot_module, "generate_video", new=AsyncMock(return_value=success)) as tool:
            tool_result, response = asyncio.run(bot.execute_pending_action())

        tool.assert_awaited_once()
        assert tool_result == success
        assert response

    def test_executes_when_balance_sufficient(self, bot):
        bot.session_manager.get_pending_action = AsyncMock(return_value=dict(VIDEO_ACTION))
        bot.session_manager.claim_pending_action = AsyncMock(return_value=dict(VIDEO_ACTION))
        set_token_balance(500)

        success = "Video generated successfully with veo-3.1."
        with patch.object(chatbot_module, "generate_video", new=AsyncMock(return_value=success)) as tool:
            tool_result, response = asyncio.run(bot.execute_pending_action())

        tool.assert_awaited_once()
        assert tool_result == success

    def test_concurrent_loser_gets_none(self, bot):
        # Peek sees the action, but another request claims it first
        bot.session_manager.get_pending_action = AsyncMock(return_value=dict(VIDEO_ACTION))
        bot.session_manager.claim_pending_action = AsyncMock(return_value=None)
        set_token_balance(500)

        with patch.object(chatbot_module, "generate_video", new=AsyncMock()) as tool:
            tool_result, response = asyncio.run(bot.execute_pending_action())

        assert (tool_result, response) == (None, None)
        tool.assert_not_awaited()

    def test_generator_error_returns_friendly_explanation(self, bot):
        bot.session_manager.get_pending_action = AsyncMock(return_value=dict(VIDEO_ACTION))
        bot.session_manager.claim_pending_action = AsyncMock(return_value=dict(VIDEO_ACTION))
        set_token_balance(500)

        error = (
            "GENERATION_ERROR | type=video | category=provider_validation "
            "| status=422 | detail=Prompt rejected"
        )
        with patch.object(chatbot_module, "generate_video", new=AsyncMock(return_value=error)), \
             patch.object(bot, "_explain_generation_error", new=AsyncMock(return_value="⚠️ friendly")) as explain:
            tool_result, response = asyncio.run(bot.execute_pending_action())

        explain.assert_awaited_once()
        assert tool_result is None  # never sets just_generated for failures
        assert response == "⚠️ friendly"


class TestSavePendingOrBlock:
    CONFIRMATION_EN = "Cost: 352 tokens (44 tokens/sec × 8 sec). Do you confirm?"

    def test_blocks_and_does_not_save_when_insufficient(self, bot):
        set_token_balance(10)
        blocked, block = run_with_block(
            bot._save_pending_or_block(
                "generate_video",
                {"prompt": "a sunset", "model": "veo-3.1", "duration": 8},
                self.CONFIRMATION_EN,
            )
        )
        assert blocked is not None
        assert "don't have enough tokens" in blocked  # English confirmation -> en
        bot.session_manager.save_pending_action.assert_not_awaited()
        assert block == {"required": 352, "available": 10}

    def test_saves_when_sufficient(self, bot):
        set_token_balance(500)
        blocked = asyncio.run(
            bot._save_pending_or_block(
                "generate_video",
                {"prompt": "a sunset", "model": "veo-3.1", "duration": 8},
                self.CONFIRMATION_EN,
            )
        )
        assert blocked is None
        bot.session_manager.save_pending_action.assert_awaited_once()

    def test_saves_when_balance_unknown(self, bot):
        set_token_balance(None)
        blocked = asyncio.run(
            bot._save_pending_or_block(
                "generate_video",
                {"prompt": "a sunset", "model": "veo-3.1", "duration": 8},
                self.CONFIRMATION_EN,
            )
        )
        assert blocked is None
        bot.session_manager.save_pending_action.assert_awaited_once()

    def test_legacy_model_uses_quoted_cost_fallback(self, bot):
        # luma-labs has no rate in pricing.py -> falls back to "Cost: N tokens"
        set_token_balance(10)
        blocked, block = run_with_block(
            bot._save_pending_or_block(
                "generate_video",
                {"prompt": "a sunset", "model": "luma-labs", "duration": 5},
                "Cost: 85 tokens. Do you confirm?",
            )
        )
        assert blocked is not None
        assert block == {"required": 85, "available": 10}

    @pytest.mark.parametrize(
        "confirmation",
        [
            "Cost: 85 tokens. Do you confirm?",
            "Cost: **85** tokens. Do you confirm?",
            "Costo total: 85 tokens. ¿Confirmas?",
            "El costo es 85 tokens. ¿Confirmas?",
        ],
    )
    def test_quoted_cost_regex_variants(self, bot, confirmation):
        set_token_balance(10)
        blocked, block = run_with_block(
            bot._save_pending_or_block(
                "generate_video",
                {"prompt": "a sunset", "model": "luma-labs", "duration": 5},
                confirmation,
            )
        )
        assert blocked is not None
        assert block == {"required": 85, "available": 10}

    def test_unknown_pending_function_returns_friendly_message(self, bot):
        bad_action = dict(VIDEO_ACTION, function="generate_music")
        bot.session_manager.get_pending_action = AsyncMock(return_value=bad_action)
        bot.session_manager.clear_pending_action = AsyncMock()
        set_token_balance(500)

        tool_result, response = asyncio.run(bot.execute_pending_action())

        assert tool_result is None
        assert "generate_music" not in response  # no raw internals leaked
        assert response.startswith("⚠️")
        bot.session_manager.clear_pending_action.assert_awaited_once()
        bot.session_manager.claim_pending_action.assert_not_awaited()

    def test_unknown_cost_never_blocks(self, bot):
        set_token_balance(10)
        blocked = asyncio.run(
            bot._save_pending_or_block(
                "generate_video",
                {"prompt": "a sunset", "model": "luma-labs", "duration": 5},
                "Shall we proceed?",  # no quoted cost anywhere
            )
        )
        assert blocked is None
        bot.session_manager.save_pending_action.assert_awaited_once()
