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
from generation_errors import format_generation_processing
from pricing import affordable_options, min_video_cost
from workflow_state import (
    WORKFLOW_IMAGE,
    WORKFLOW_VIDEO_GEN,
    apply_user_message,
    new_state,
)
from request_context import (
    clear_insufficient_block,
    get_insufficient_block,
    set_token_balance,
)


VIDEO_ACTION = {
    "function": "generate_video",
    "args": {"prompt": "a sunset", "model": "veo-3.1", "duration": 8},
    "cost_message": "Costo: 336 tokens (42 tokens/sec × 8 sec). ¿Confirmas?",
    "estimated_cost": 336,
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
        assert "336" in response and "10" in response
        tool.assert_not_awaited()
        # Action must NOT be claimed/deleted: user can adjust or top up
        bot.session_manager.claim_pending_action.assert_not_awaited()
        assert block == {"required": 336, "available": 10}

    def test_executes_when_balance_unknown(self, bot):
        bot.session_manager.get_pending_action = AsyncMock(return_value=dict(VIDEO_ACTION))
        bot.session_manager.claim_pending_action = AsyncMock(return_value=dict(VIDEO_ACTION))
        set_token_balance(None)

        success = "Video generated successfully with veo-3.1."
        with patch.object(chatbot_module, "generate_video", new=AsyncMock(return_value=success)) as tool, \
             patch.object(bot, "_friendly_success_message", new=AsyncMock(return_value="✅ done")):
            tool_result, response = asyncio.run(bot.execute_pending_action())

        tool.assert_awaited_once()
        assert tool_result == success
        assert response

    def test_executes_when_balance_sufficient(self, bot):
        bot.session_manager.get_pending_action = AsyncMock(return_value=dict(VIDEO_ACTION))
        bot.session_manager.claim_pending_action = AsyncMock(return_value=dict(VIDEO_ACTION))
        set_token_balance(500)

        success = "Video generated successfully with veo-3.1."
        with patch.object(chatbot_module, "generate_video", new=AsyncMock(return_value=success)) as tool, \
             patch.object(bot, "_friendly_success_message", new=AsyncMock(return_value="✅ done")):
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

    def test_processing_202_returns_localized_message(self, bot):
        # The hybrid backend answered 202: the generation was accepted but isn't
        # finished. The user gets a friendly "on the way" message (Spanish here,
        # since VIDEO_ACTION's cost_message is Spanish) — not an error.
        bot.session_manager.get_pending_action = AsyncMock(return_value=dict(VIDEO_ACTION))
        bot.session_manager.claim_pending_action = AsyncMock(return_value=dict(VIDEO_ACTION))
        set_token_balance(500)

        marker = format_generation_processing("video")
        with patch.object(chatbot_module, "generate_video", new=AsyncMock(return_value=marker)) as tool:
            tool_result, response = asyncio.run(bot.execute_pending_action())

        tool.assert_awaited_once()
        assert tool_result == marker
        assert "aparecerá" in response  # localized to es
        assert "⚠️" not in response  # not framed as an error


class TestContextualMessageRendering:
    @pytest.fixture
    def bot(self):
        return GeminiChatbot(conversation_uuid="test-render")

    def test_processing_marker_renders_spanish(self, bot):
        msg = bot._generate_contextual_success_message(
            format_generation_processing("video"), tool_was_called=True, lang="es"
        )
        assert "aparecerá" in msg

    def test_processing_marker_renders_english(self, bot):
        msg = bot._generate_contextual_success_message(
            format_generation_processing("video"), tool_was_called=True, lang="en"
        )
        assert "appear" in msg

    def test_success_renders_in_conversation_language_not_raw_signal(self, bot):
        # tools.py returns an English internal signal; the user-facing message
        # must follow the conversation language, not leak that signal verbatim.
        raw = "Video generated successfully with veo-3.1."
        es = bot._generate_contextual_success_message(raw, tool_was_called=True, lang="es")
        en = bot._generate_contextual_success_message(raw, tool_was_called=True, lang="en")
        assert es == "✅ ¡Tu video se generó correctamente!"
        assert en == "✅ Your video was generated successfully!"
        assert "veo-3.1" not in es  # raw signal never leaked

    def test_success_detects_image_and_audio_types(self, bot):
        img = bot._generate_contextual_success_message(
            "Images generated successfully with Seedream.", tool_was_called=True, lang="es"
        )
        audio = bot._generate_contextual_success_message(
            "Audio generated successfully (1234 bytes). Link generated automatically.",
            tool_was_called=True, lang="es",
        )
        assert "imagen" in img.lower()
        assert "audio" in audio.lower()


class TestSavePendingOrBlock:
    CONFIRMATION_EN = "Cost: 336 tokens (42 tokens/sec × 8 sec). Do you confirm?"

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
        assert block == {"required": 336, "available": 10}

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


class TestConversationLanguageOverridesHeuristic:
    """
    The block must render in the CONVERSATION language, not in whatever
    language the internal cost/confirmation string happens to be written in.
    """

    def test_lang_for_prefers_conversation_language(self, bot):
        bot._conv_lang = "en"
        # Spanish fallback text, but the resolved conversation is English
        assert bot._lang_for("El costo será 336 tokens. ¿Confirmas?") == "en"

    def test_lang_for_falls_back_to_heuristic_when_unresolved(self, bot):
        bot._conv_lang = None
        assert bot._lang_for("El costo será 336 tokens. ¿Confirmas?") == "es"
        assert bot._lang_for("Cost: 336 tokens. Do you confirm?") == "en"

    def test_execute_pending_block_uses_english_when_conversation_is_english(self, bot):
        # Reproduces the reported bug: Spanish cost_message but English chat.
        bot.session_manager.get_pending_action = AsyncMock(return_value=dict(VIDEO_ACTION))
        bot._conv_lang = "en"
        set_token_balance(10)

        with patch.object(chatbot_module, "generate_video", new=AsyncMock()):
            (tool_result, response), block = run_with_block(bot.execute_pending_action())

        assert tool_result is None
        assert "don't have enough tokens" in response  # English, not Spanish
        assert "No tienes tokens suficientes" not in response

    def test_save_pending_block_uses_spanish_when_conversation_is_spanish(self, bot):
        # English confirmation text, but the conversation is Spanish.
        bot._conv_lang = "es"
        set_token_balance(10)

        blocked, _ = run_with_block(
            bot._save_pending_or_block(
                "generate_video",
                {"prompt": "a sunset", "model": "veo-3.1", "duration": 8},
                "Cost: 336 tokens. Do you confirm?",
            )
        )
        assert blocked is not None
        assert "No tienes tokens suficientes" in blocked


class TestResolveConversationLanguage:
    def _history(self, *pairs):
        return [{"role": role, "content": content} for role, content in pairs]

    def test_clear_current_message_wins(self, bot):
        bot.session_manager.get_session = AsyncMock(return_value={"messages": []})
        lang = asyncio.run(
            bot._resolve_conversation_language("genera un video de un perro", [])
        )
        assert lang == "es"

    def test_ambiguous_current_falls_back_to_history(self, bot):
        history = self._history(
            ("user", "generate a realistic image of a pug"),
            ("assistant", "Which model would you like to use?"),
        )
        lang = asyncio.run(
            bot._resolve_conversation_language("veo 3.1 flash 8 sec", history)
        )
        assert lang == "en"

    def test_history_uses_user_messages_only(self, bot):
        # Assistant spoke English, but the last clear USER message was Spanish.
        history = self._history(
            ("user", "quiero un video de una rana saltando"),
            ("assistant", "Which model would you like to use?"),
        )
        lang = asyncio.run(
            bot._resolve_conversation_language("8", history)
        )
        assert lang == "es"

    def test_defaults_to_english_when_no_signal(self, bot):
        lang = asyncio.run(bot._resolve_conversation_language("ok", []))
        assert lang == "en"


class TestBuildLanguageSample:
    def test_joins_recent_user_messages_only(self, bot):
        history = [
            {"role": "user", "content": "erstelle ein Video von einem Hund"},
            {"role": "assistant", "content": "Which model?"},
        ]
        sample = bot._build_language_sample("veo 3.1", history)
        assert "veo 3.1" in sample
        assert "erstelle ein Video" in sample
        assert "Which model?" not in sample  # assistant turns excluded

    def test_skips_empty_messages(self, bot):
        sample = bot._build_language_sample("hola", [{"role": "user", "content": "  "}])
        assert sample == "hola"


class TestClarificationLocalization:
    """
    Ambiguity clarification prompts bypass Gemini, so they must render in the
    resolved conversation language — not a hardcoded one. Regression for the bot
    replying in Spanish during an English conversation.
    """

    def test_generic_clarification_follows_english_conversation(self, bot):
        bot._conv_lang = "en"
        assert bot._clarification_text("generic") == (
            "What exactly would you like to create? An image or a video?"
        )

    def test_generic_clarification_follows_spanish_conversation(self, bot):
        bot._conv_lang = "es"
        assert bot._clarification_text("generic").startswith("¿Qué quieres crear")

    def test_ref_file_clarification_follows_conversation_language(self, bot):
        bot._conv_lang = "en"
        assert bot._clarification_text("ref_file").startswith("What would you like to do")

    def test_unknown_key_falls_back_to_generic_english(self, bot):
        bot._conv_lang = None  # outside an active turn → heuristic default = en
        assert bot._clarification_text("nope") == (
            "What exactly would you like to create? An image or a video?"
        )

    def test_needs_clarification_returns_language_neutral_key(self):
        # The module function must NOT bake in a language — it returns a key.
        assert chatbot_module.needs_clarification("make it", False) == (True, "generic")
        assert chatbot_module.needs_clarification("crea", False) == (True, "generic")
        assert chatbot_module.needs_clarification("this", True) == (True, "ref_file")
        assert chatbot_module.needs_clarification("make a video", False) == (False, "")


class TestLocalizeBalanceBlock:
    """
    DATA is always code-computed; only the wording is localized. Spanish/English
    use the deterministic template (no model call); other languages are rendered
    by a one-shot model with the template as the offline fallback.
    """

    def test_english_sample_returns_template_without_model_call(self, bot):
        bot._conv_lang = "en"
        bot._lang_sample = "generate a video of a dog running in the park"
        with patch.object(chatbot_module.genai, "GenerativeModel") as model_cls:
            block = asyncio.run(
                bot._localize_balance_block(352, 10, affordable_options(10))
            )
        assert "don't have enough tokens" in block
        model_cls.assert_not_called()  # es/en never hit the LLM

    def test_spanish_sample_returns_template_without_model_call(self, bot):
        bot._conv_lang = "es"
        bot._lang_sample = "genera un video de un perro corriendo en el parque"
        with patch.object(chatbot_module.genai, "GenerativeModel") as model_cls:
            block = asyncio.run(
                bot._localize_balance_block(352, 10, affordable_options(10))
            )
        assert "No tienes tokens suficientes" in block
        model_cls.assert_not_called()

    def test_other_language_sample_invokes_model(self, bot):
        bot._conv_lang = "en"  # default for an undetected language
        bot._lang_sample = "erstelle ein Video von einem springenden Frosch bitte"

        fake_model = MagicMock()
        fake_model.generate_content_async = AsyncMock(
            return_value=MagicMock(text="⚠️ Sie haben nicht genügend Tokens.")
        )
        with patch.object(chatbot_module.genai, "GenerativeModel", return_value=fake_model) as model_cls:
            block = asyncio.run(
                bot._localize_balance_block(352, 10, affordable_options(10))
            )
        model_cls.assert_called_once()
        assert block == "⚠️ Sie haben nicht genügend Tokens."

    def test_other_language_falls_back_to_template_on_model_failure(self, bot):
        bot._conv_lang = "en"
        bot._lang_sample = "erstelle ein Video von einem springenden Frosch bitte"

        fake_model = MagicMock()
        fake_model.generate_content_async = AsyncMock(side_effect=RuntimeError("boom"))
        with patch.object(chatbot_module.genai, "GenerativeModel", return_value=fake_model):
            block = asyncio.run(
                bot._localize_balance_block(352, 10, affordable_options(10))
            )
        # Falls back to the English template — a block is never lost.
        assert "don't have enough tokens" in block

    def test_no_sample_uses_fallback_text_and_skips_model(self, bot):
        # Outside an active turn (e.g. helper called in isolation): no sample,
        # so language comes from the fallback text and the model is never called.
        bot._conv_lang = None
        bot._lang_sample = ""
        with patch.object(chatbot_module.genai, "GenerativeModel") as model_cls:
            block = asyncio.run(
                bot._localize_balance_block(
                    352, 10, affordable_options(10),
                    fallback_text="El costo es 336 tokens. ¿Confirmas?",
                )
            )
        assert "No tienes tokens suficientes" in block
        model_cls.assert_not_called()


class TestReplyConfirmsCost:
    """Intent-based confirmation: obvious yeses are instant, free-form replies
    at the confirmation step defer to the LLM, and non-confirmations never do."""

    def _confirms(self, bot, message, pending=None, state=None, llm=False):
        with patch.object(bot, "_llm_confirms_intent", new=AsyncMock(return_value=llm)) as llm_mock:
            result = asyncio.run(
                bot._reply_confirms_cost(message, pending, state, has_ref_video=False)
            )
        return result, llm_mock

    def test_obvious_yes_confirms_without_llm(self, bot):
        for msg in ("yes", "dale", "ok", "yes confirm", "si dale"):
            confirmed, llm = self._confirms(bot, msg, pending=dict(VIDEO_ACTION))
            assert confirmed is True
            llm.assert_not_awaited()  # instant regex, no model call

    def test_not_at_confirmation_returns_false_without_llm(self, bot):
        # No pending action and no ready state → not a confirmation point.
        confirmed, llm = self._confirms(bot, "sounds interesting", pending=None, state=None)
        assert confirmed is False
        llm.assert_not_awaited()

    def test_decline_at_confirmation_returns_false_without_llm(self, bot):
        confirmed, llm = self._confirms(bot, "no, wait", pending=dict(VIDEO_ACTION))
        assert confirmed is False
        llm.assert_not_awaited()

    def test_new_request_at_confirmation_returns_false_without_llm(self, bot):
        # A fresh generation request is a new intent, not a bare confirmation.
        confirmed, llm = self._confirms(
            bot, "hazme un video de un perro", pending=dict(VIDEO_ACTION)
        )
        assert confirmed is False
        llm.assert_not_awaited()

    def test_freeform_reply_defers_to_llm_yes(self, bot):
        confirmed, llm = self._confirms(
            bot, "me encanta, hazlo ya", pending=dict(VIDEO_ACTION), llm=True
        )
        assert confirmed is True
        llm.assert_awaited_once()

    def test_freeform_reply_defers_to_llm_no(self, bot):
        confirmed, llm = self._confirms(
            bot, "how much is that in dollars?", pending=dict(VIDEO_ACTION), llm=False
        )
        assert confirmed is False
        llm.assert_awaited_once()


# ---------------------------------------------------------------------------
# Up-front video budget gate (fires BEFORE the prompt/model/duration interview)
# ---------------------------------------------------------------------------
class TestUpFrontVideoBudgetGate:
    """`_video_budget_block` must warn at the START of a video workflow, not
    after the full interview, and must stay silent once a prompt is captured."""

    def _block(self, bot, state, balance, message="I want to create a video"):
        set_token_balance(balance)
        bot.session_manager.add_message = AsyncMock()
        bot.session_manager.save_workflow_state = AsyncMock()
        return asyncio.run(bot._video_budget_block(state, message))

    def test_blocks_video_intent_when_no_video_is_affordable(self, bot):
        state = new_state(WORKFLOW_VIDEO_GEN)
        assert state["step"] == "awaiting_prompt"

        # Must sit below min_video_cost, which Seedance Mini pushed down to
        # 8 tokens (2 tokens/s x its 4s minimum).
        text = self._block(bot, state, balance=5)

        assert text is not None
        assert "5" in text
        assert str(min_video_cost()) in text
        # Steers to what the balance CAN buy instead of dead-ending.
        assert "Seedream" in text
        bot.session_manager.save_workflow_state.assert_awaited_once()

    def test_silent_once_a_prompt_is_captured(self, bot):
        """No nagging mid-flow: past awaiting_prompt the gate stops firing."""
        state = new_state(WORKFLOW_VIDEO_GEN)
        state = apply_user_message(state, "a frog jumping across lily pads in the rain")
        assert state["step"] != "awaiting_prompt"

        assert self._block(bot, state, balance=20) is None

    def test_silent_when_balance_covers_a_video(self, bot):
        state = new_state(WORKFLOW_VIDEO_GEN)
        assert self._block(bot, state, balance=min_video_cost()) is None

    def test_silent_for_image_workflows(self, bot):
        """Images cost 4 tokens — a 20-token balance is fine, never warn."""
        state = new_state(WORKFLOW_IMAGE)
        assert self._block(bot, state, balance=20) is None

    def test_silent_when_balance_is_unknown(self, bot):
        state = new_state(WORKFLOW_VIDEO_GEN)
        assert self._block(bot, state, balance=None) is None
