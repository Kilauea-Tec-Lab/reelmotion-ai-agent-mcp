"""
Integration-style tests for the state-driven confirmation rebuild in
chatbot.GeminiChatbot.send_message (session manager, tools, moderation and the
LLM extractor are mocked — no Redis, no network).
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import chatbot as chatbot_module
from chatbot import GeminiChatbot
from request_context import clear_insufficient_block, set_token_balance
from workflow_state import WORKFLOW_VIDEO_GEN, compute_step, new_state

VEO_JSON = '{"scene": "neon Tokyo alley", "camera": {"movement": "dolly"}, "duration": "8s"}'

COST_MESSAGE = (
    'Prompt: "a frog jumping in the jungle"\n'
    "Model: Veo 3.1\nDuration: 8 seconds\n"
    "Cost: 352 tokens (44 tokens/sec × 8 sec). Do you confirm?"
)


def ready_video_state(prompt="a frog jumping in the jungle", prompt_format="text"):
    state = new_state(WORKFLOW_VIDEO_GEN)
    state["params"].update({
        "prompt": prompt,
        "prompt_format": prompt_format,
        "refine_resolved": True,
        "model": "veo-3.1",
        "duration": 8,
    })
    state["step"] = compute_step(state)
    return state


class FakeSessionManager:
    """In-memory stand-in tracking pending actions and workflow state."""

    def __init__(self, workflow_state=None, history=None):
        self.pending = None
        self.workflow_state = workflow_state
        self.history = history or []
        self.cleared_workflow = False

        self.add_message = AsyncMock()
        self.set_just_generated = AsyncMock()
        self.get_just_generated = AsyncMock(return_value=False)
        self.clear_just_generated = AsyncMock()
        self.get_reference_files = AsyncMock(return_value=[])

    async def get_workflow_state(self, uuid):
        return self.workflow_state

    async def save_workflow_state(self, uuid, state):
        self.workflow_state = state

    async def clear_workflow_state(self, uuid):
        self.workflow_state = None
        self.cleared_workflow = True

    async def save_pending_action(self, uuid, action):
        self.pending = action

    async def get_pending_action(self, uuid):
        return self.pending

    async def clear_pending_action(self, uuid):
        self.pending = None

    async def claim_pending_action(self, uuid):
        action, self.pending = self.pending, None
        return action

    async def get_session(self, uuid):
        return {"messages": self.history}


@pytest.fixture
def make_bot():
    def _make(workflow_state=None, history=None):
        bot = GeminiChatbot(conversation_uuid="test-rebuild")
        bot.session_manager = FakeSessionManager(workflow_state, history)
        bot.chat_session = MagicMock()  # skip start_chat
        return bot

    yield _make
    set_token_balance(None)
    clear_insufficient_block()


def run(coro):
    return asyncio.run(coro)


MODERATION_OK = patch.object(
    chatbot_module, "is_disallowed_content_full", new=AsyncMock(return_value=False)
)


class TestRebuildFromState:
    def test_confirmation_rebuilds_from_complete_state(self, make_bot):
        bot = make_bot(workflow_state=ready_video_state())
        set_token_balance(None)
        success = "Video generated successfully with veo-3.1."

        with MODERATION_OK, patch.object(
            chatbot_module, "generate_video", new=AsyncMock(return_value=success)
        ) as tool, patch.object(
            chatbot_module, "extract_generation_params", new=AsyncMock(return_value={})
        ) as extractor:
            response = run(bot.send_message("yes"))

        tool.assert_awaited_once()
        args = tool.await_args.kwargs
        assert args["prompt"] == "a frog jumping in the jungle"
        assert args["model"] == "veo-3.1"
        assert args["duration"] == 8
        extractor.assert_not_awaited()  # state was complete — no LLM fallback
        assert response  # user got a success message
        bot.session_manager.set_just_generated.assert_awaited_once()
        assert bot.session_manager.workflow_state is None  # cleared after success

    def test_json_prompt_reaches_tool_byte_identical(self, make_bot):
        bot = make_bot(workflow_state=ready_video_state(VEO_JSON, "json"))
        set_token_balance(None)

        with MODERATION_OK, patch.object(
            chatbot_module, "generate_video", new=AsyncMock(return_value="ok video")
        ) as tool:
            run(bot.send_message("dale"))

        assert tool.await_args.kwargs["prompt"] == VEO_JSON

    def test_balance_gate_blocks_rebuilt_action(self, make_bot):
        bot = make_bot(workflow_state=ready_video_state())
        set_token_balance(10)  # veo-3.1 8s = 352 tokens

        with MODERATION_OK, patch.object(
            chatbot_module, "generate_video", new=AsyncMock()
        ) as tool:
            response = run(bot.send_message("yes"))

        tool.assert_not_awaited()
        assert "352" in response and "10" in response
        # State survives a block so the user can adjust or top up
        assert bot.session_manager.workflow_state is not None

    def test_no_state_falls_back_to_extractor(self, make_bot):
        history = [
            {"role": "user", "content": "a frog jumping in the jungle, veo 3.1"},
            {"role": "assistant", "content": COST_MESSAGE},
        ]
        bot = make_bot(workflow_state=None, history=history)
        set_token_balance(None)
        extracted = {
            "intent": "video_gen",
            "prompt": "a frog jumping in the jungle",
            "model": "veo 3.1",
            "duration": 8,
        }

        with MODERATION_OK, patch.object(
            chatbot_module, "generate_video", new=AsyncMock(return_value="ok video")
        ) as tool, patch.object(
            chatbot_module, "extract_generation_params", new=AsyncMock(return_value=extracted)
        ) as extractor:
            response = run(bot.send_message("yes"))

        extractor.assert_awaited_once()
        tool.assert_awaited_once()
        assert tool.await_args.kwargs["model"] == "veo-3.1"  # normalized
        assert response

    def test_extractor_failure_executes_nothing(self, make_bot):
        history = [{"role": "assistant", "content": COST_MESSAGE}]
        bot = make_bot(workflow_state=None, history=history)
        set_token_balance(None)
        # Gemini also unavailable → outer handler returns an error message,
        # but crucially NO tool ran and NO pending action was saved.
        bot.chat_session.send_message_async = AsyncMock(side_effect=RuntimeError("no gemini"))

        with MODERATION_OK, patch.object(
            chatbot_module, "generate_video", new=AsyncMock()
        ) as tool, patch.object(
            chatbot_module, "extract_generation_params", new=AsyncMock(return_value={})
        ):
            run(bot.send_message("yes"))

        tool.assert_not_awaited()
        assert bot.session_manager.pending is None


class TestFastPathWithPendingAction:
    def test_pending_action_still_executes_and_clears_state(self, make_bot):
        bot = make_bot(workflow_state=ready_video_state())
        bot.session_manager.pending = {
            "function": "generate_video",
            "args": {"prompt": "a frog", "model": "veo-3.1", "duration": 8},
            "cost_message": COST_MESSAGE,
            "estimated_cost": 352,
        }
        set_token_balance(500)

        with MODERATION_OK, patch.object(
            chatbot_module, "generate_video", new=AsyncMock(return_value="ok video")
        ) as tool:
            response = run(bot.send_message("yes"))

        tool.assert_awaited_once()
        assert response
        assert bot.session_manager.workflow_state is None


class TestRecoveryPathBalanceGate:
    def test_recovery_tool_call_is_blocked_when_balance_insufficient(self, make_bot):
        """The recovery function-call path must also respect the balance gate."""
        from types import SimpleNamespace

        # First Gemini response: no function call, no text → triggers recovery.
        empty_part = SimpleNamespace(text=None, function_call=None)
        empty_response = SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[empty_part]))],
            parts=[],
            text=None,
        )
        # Recovery response: Gemini insists on calling generate_video.
        fc = SimpleNamespace(
            name="generate_video",
            args={"prompt": "a frog", "model": "veo-3.1", "duration": 8},
        )
        fc_part = SimpleNamespace(text=None, function_call=fc)
        recovery_response = SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[fc_part]))],
            parts=[],
            text=None,
        )

        bot = make_bot()  # no state, no pending action
        bot.chat_session.send_message_async = AsyncMock(
            side_effect=[empty_response, recovery_response]
        )
        set_token_balance(10)  # veo-3.1 8s = 352 tokens

        with MODERATION_OK, patch.object(
            chatbot_module, "generate_video", new=AsyncMock()
        ) as tool, patch.object(
            chatbot_module, "extract_generation_params", new=AsyncMock(return_value={})
        ):
            response = run(bot.send_message("a frog jumping in the jungle with veo 3.1"))

        tool.assert_not_awaited()
        assert "352" in response and "10" in response


class TestMergeStateIntoArgs:
    def test_json_prompt_overrides_gemini_mangled_arg(self, make_bot):
        bot = make_bot()
        state = ready_video_state(VEO_JSON, "json")
        mangled = '{"camera": {"movement": "dolly"}, "scene": "neon Tokyo alley"}'

        merged = bot._merge_state_into_args(
            "generate_video", {"prompt": mangled, "model": "veo-3.1", "duration": 8}, state
        )
        assert merged["prompt"] == VEO_JSON

    def test_fills_missing_params_from_state(self, make_bot):
        bot = make_bot()
        state = ready_video_state()

        merged = bot._merge_state_into_args("generate_video", {"prompt": "a frog"}, state)
        assert merged["model"] == "veo-3.1"
        assert merged["duration"] == 8

    def test_state_wins_on_model_conflict(self, make_bot):
        bot = make_bot()
        state = ready_video_state()

        merged = bot._merge_state_into_args(
            "generate_video",
            {"prompt": "a frog", "model": "kling-v3-omni-std", "duration": 8},
            state,
        )
        assert merged["model"] == "veo-3.1"

    def test_none_state_is_passthrough(self, make_bot):
        bot = make_bot()
        args = {"prompt": "a frog", "model": "veo-3.1", "duration": 8}
        assert bot._merge_state_into_args("generate_video", args, None) == args
