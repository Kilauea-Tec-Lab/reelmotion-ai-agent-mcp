"""
Unit tests for workflow_state.py (pure module — no Redis/genai/network).

Run: python -m pytest test_workflow_state.py -v
"""
import json

import pytest

import pricing
from workflow_state import (
    WORKFLOW_IMAGE,
    WORKFLOW_SPEECH,
    WORKFLOW_UNKNOWN,
    WORKFLOW_VIDEO_EDIT,
    WORKFLOW_VIDEO_GEN,
    RACHEL_VOICE_ID,
    apply_user_message,
    build_action_args,
    capture_refined_prompt,
    compute_step,
    detect_json_prompt,
    detect_workflow_intent,
    is_ready_for_confirmation,
    json_prompt_summary,
    merge_extracted,
    missing_fields,
    new_state,
    normalize_model_name,
    parse_duration,
    parse_resolution,
    parse_voice,
    state_context_note,
)

VEO_JSON = (
    '{\n  "scene": "a neon-lit Tokyo alley at night",\n'
    '  "camera": {"movement": "slow dolly", "angle": "low"},\n'
    '  "lighting": "rain reflections, cyan and magenta",\n'
    '  "duration": "8s"\n}'
)


# ---------------------------------------------------------------------------
# detect_json_prompt
# ---------------------------------------------------------------------------
class TestDetectJsonPrompt:
    def test_raw_json_object(self):
        assert detect_json_prompt(VEO_JSON) == VEO_JSON

    def test_verbatim_byte_identical_substring(self):
        message = "please use this:\n" + VEO_JSON + "\nthanks!"
        result = detect_json_prompt(message)
        assert result == VEO_JSON
        assert result in message  # exact substring, never re-serialized

    def test_fenced_json_block(self):
        message = "here is my prompt\n```json\n" + VEO_JSON + "\n```\nmake it a video"
        assert detect_json_prompt(message) == VEO_JSON

    def test_fenced_block_without_language_tag(self):
        message = "```\n" + VEO_JSON + "\n```"
        assert detect_json_prompt(message) == VEO_JSON

    def test_prose_before_and_after(self):
        message = "I made this prompt: " + VEO_JSON + " can you improve it?"
        assert detect_json_prompt(message) == VEO_JSON

    def test_malformed_json_returns_none(self):
        assert detect_json_prompt('{"scene": "x", "camera": }') is None

    def test_trailing_comma_returns_none(self):
        assert detect_json_prompt('{"scene": "x",}') is None

    def test_array_rejected(self):
        assert detect_json_prompt('["a", "b"]') is None

    def test_scalar_rejected(self):
        assert detect_json_prompt('"just a string"') is None

    def test_empty_object_rejected(self):
        assert detect_json_prompt("{}") is None

    def test_no_braces(self):
        assert detect_json_prompt("a frog jumping in the jungle") is None

    def test_braces_inside_string_values(self):
        blob = '{"scene": "a sign that says {hello}", "mood": "fun"}'
        assert detect_json_prompt(blob) == blob

    def test_escaped_quotes_inside_values(self):
        blob = '{"scene": "she said \\"go\\" loudly"}'
        assert detect_json_prompt(blob) == blob

    def test_two_json_blocks_first_wins(self):
        first = '{"scene": "first"}'
        second = '{"scene": "second"}'
        message = first + " or maybe " + second
        assert detect_json_prompt(message) == first

    def test_nested_objects(self):
        blob = '{"a": {"b": {"c": 1}}, "d": [1, 2]}'
        assert detect_json_prompt(blob) == blob

    def test_json_prompt_summary_video_keys(self):
        count, looks_video = json_prompt_summary(VEO_JSON)
        assert count == 4
        assert looks_video is True

    def test_json_prompt_summary_non_video(self):
        count, looks_video = json_prompt_summary('{"subject": "a cat", "style": "anime"}')
        assert count == 2
        assert looks_video is False


# ---------------------------------------------------------------------------
# normalize_model_name
# ---------------------------------------------------------------------------
class TestNormalizeModelName:
    @pytest.mark.parametrize("canonical", list(pricing.IMAGE_COSTS) + list(pricing.VIDEO_TOKEN_RATES) + list(pricing.SEEDANCE2_MODELS))
    def test_every_pricing_key_resolves_to_itself(self, canonical):
        assert normalize_model_name(f"use {canonical} please") == canonical

    @pytest.mark.parametrize("alias,expected", [
        ("lets go with veo 3.1", "veo-3.1"),
        ("veo 3.1 ultra", "veo-3.1-ultra"),
        ("Veo 3.1 Flash", "veo-3.1-flash"),
        ("seedance 2.0 fast", "seedance-2.0-fast"),
        ("seedance 2", "seedance-2.0"),
        ("runway aleph", "runway-aleph"),
        ("runway 4.5", "runway-4.5"),
        ("kling pro", "kling-v3-omni-pro"),
        ("kling std", "kling-v3-omni-std"),
        ("nano banana", "Nano Banana 2"),
        ("use GPT", "GPT"),
        ("freepik", "Freepik"),
    ])
    def test_aliases(self, alias, expected):
        assert normalize_model_name(alias) == expected

    def test_longest_alias_wins(self):
        assert normalize_model_name("veo 3.1 ultra please") == "veo-3.1-ultra"

    def test_no_match(self):
        assert normalize_model_name("a frog jumping") is None

    def test_no_partial_word_match(self):
        # "gpt" inside a word must not match
        assert normalize_model_name("egpt2 something") is None


# ---------------------------------------------------------------------------
# parsers
# ---------------------------------------------------------------------------
class TestParsers:
    def test_parse_duration_bare_int_allowed(self):
        assert parse_duration("8", allow_bare=True) == 8

    def test_parse_duration_bare_int_disallowed(self):
        assert parse_duration("8", allow_bare=False) is None

    def test_parse_duration_suffixed(self):
        assert parse_duration("hazlo de 10 segundos", allow_bare=False) == 10
        assert parse_duration("make it 5s", allow_bare=False) == 5

    def test_parse_duration_validates_model_rules(self):
        assert parse_duration("7 seconds", model="runway-aleph") is None  # only 5/10
        assert parse_duration("10 seconds", model="runway-aleph") == 10

    def test_parse_resolution(self):
        assert parse_resolution("720p please") == "720p"
        assert parse_resolution("nothing here") is None

    def test_parse_voice(self):
        name, voice_id = parse_voice("lets go with Rachel")
        assert name == "Rachel"
        assert voice_id == RACHEL_VOICE_ID
        assert parse_voice("no voice here") is None


# ---------------------------------------------------------------------------
# intent detection
# ---------------------------------------------------------------------------
class TestDetectWorkflowIntent:
    def test_video(self):
        assert detect_workflow_intent("quiero crear un video") == WORKFLOW_VIDEO_GEN

    def test_image(self):
        assert detect_workflow_intent("make an image of a cat") == WORKFLOW_IMAGE

    def test_speech(self):
        assert detect_workflow_intent("create a voice narration") == WORKFLOW_SPEECH

    def test_video_model_implies_video(self):
        assert detect_workflow_intent("use veo 3.1") == WORKFLOW_VIDEO_GEN

    def test_image_model_implies_image(self):
        assert detect_workflow_intent("con nano banana") == WORKFLOW_IMAGE

    def test_edit_with_reference_video(self):
        assert detect_workflow_intent("edit this video", has_reference_video=True) == WORKFLOW_VIDEO_EDIT

    def test_edit_without_reference_is_video_gen(self):
        assert detect_workflow_intent("edit this video", has_reference_video=False) == WORKFLOW_VIDEO_GEN

    def test_no_intent(self):
        assert detect_workflow_intent("hola, ¿cómo estás?") is None


# ---------------------------------------------------------------------------
# transitions: image workflow
# ---------------------------------------------------------------------------
class TestImageWorkflow:
    def test_full_flow(self):
        state = new_state(WORKFLOW_IMAGE)
        assert state["step"] == "awaiting_prompt"

        state = apply_user_message(state, "a red dragon flying over snowy mountains")
        assert state["params"]["prompt"] == "a red dragon flying over snowy mountains"
        assert state["step"] == "offer_refine"

        state = apply_user_message(state, "no, use mine")
        assert state["params"]["refine_resolved"] is True
        assert state["step"] == "awaiting_model"

        state = apply_user_message(state, "GPT")
        assert state["params"]["model"] == "GPT"
        assert state["step"] == "awaiting_confirmation"
        assert is_ready_for_confirmation(state)

    def test_refine_acceptance_uses_candidate(self):
        state = new_state(WORKFLOW_IMAGE)
        state = apply_user_message(state, "a dragon over mountains at dawn")
        state = apply_user_message(state, "yes")  # wants help refining
        assert state["step"] == "offer_refine"  # candidate not yet produced

        refined = 'Here you go!\n✨ **Refined Prompt:** "A majestic crimson dragon soaring over snow-capped peaks at golden dawn"'
        state = capture_refined_prompt(state, refined)
        assert state["step"] == "refining"

        state = apply_user_message(state, "perfecto")
        assert state["params"]["prompt"].startswith("A majestic crimson dragon")
        assert state["step"] == "awaiting_model"

    def test_intent_only_message_is_not_a_prompt(self):
        state = new_state(WORKFLOW_IMAGE)
        state = apply_user_message(state, "quiero crear una imagen")
        assert state["params"]["prompt"] is None
        assert state["step"] == "awaiting_prompt"


# ---------------------------------------------------------------------------
# transitions: video workflow
# ---------------------------------------------------------------------------
class TestVideoWorkflow:
    def _state_with_prompt(self):
        state = new_state(WORKFLOW_VIDEO_GEN)
        state = apply_user_message(state, "a frog jumping across lily pads in the rain")
        state = apply_user_message(state, "no")
        return state

    def test_veo_auto_duration_skips_awaiting_duration(self):
        state = self._state_with_prompt()
        state = apply_user_message(state, "veo 3.1")
        assert state["params"]["model"] == "veo-3.1"
        assert state["params"]["duration"] == 8
        assert state["step"] == "awaiting_confirmation"

    def test_seedance_requires_resolution_then_duration(self):
        state = self._state_with_prompt()
        state = apply_user_message(state, "seedance 2.0")
        assert state["step"] == "awaiting_resolution"
        state = apply_user_message(state, "720p")
        assert state["step"] == "awaiting_duration"
        state = apply_user_message(state, "10")
        assert state["params"]["duration"] == 10
        assert state["step"] == "awaiting_confirmation"

    def test_seedance_fast_1080p_downgraded(self):
        state = self._state_with_prompt()
        state = apply_user_message(state, "seedance 2.0 fast")
        state = apply_user_message(state, "1080p")
        assert state["params"]["resolution"] == "720p"  # fast tier has no 1080p

    def test_bare_int_only_consumed_in_awaiting_duration(self):
        state = self._state_with_prompt()  # step: awaiting_model
        state = apply_user_message(state, "8")
        assert state["params"]["duration"] is None

    def test_model_change_invalidates_incompatible_duration(self):
        state = self._state_with_prompt()
        state = apply_user_message(state, "kling std")
        state = apply_user_message(state, "7")  # valid for kling (3-15)
        assert state["params"]["duration"] == 7
        state = apply_user_message(state, "mejor usa runway aleph")
        assert state["params"]["model"] == "runway-aleph"
        assert state["params"]["duration"] is None  # 7 invalid for runway-aleph
        assert state["step"] == "awaiting_duration"

    def test_jump_ahead_in_one_message(self):
        state = new_state(WORKFLOW_VIDEO_GEN)
        state = apply_user_message(
            state, "a frog jumping across lily pads, use veo 3.1, just generate it"
        )
        # prompt is substantive, model found, veo auto-8s — but refine not resolved
        assert state["params"]["model"] == "veo-3.1"
        assert state["params"]["duration"] == 8
        assert state["params"]["prompt"] is not None

    def test_type_switch_resets_workflow(self):
        state = self._state_with_prompt()
        state = apply_user_message(state, "olvida el video, mejor crea una imagen")
        assert state["workflow_type"] == WORKFLOW_IMAGE
        assert state["params"]["prompt"] is None

    def test_video_edit_rejects_non_edit_models(self):
        state = new_state(WORKFLOW_VIDEO_EDIT)
        state = apply_user_message(state, "remove the background people")
        state = apply_user_message(state, "no")
        state = apply_user_message(state, "veo 3.1")  # not a video-edit model
        assert state["params"]["model"] is None
        state = apply_user_message(state, "runway aleph")
        assert state["params"]["model"] == "runway-aleph"


# ---------------------------------------------------------------------------
# transitions: JSON prompts
# ---------------------------------------------------------------------------
class TestJsonPromptFlow:
    def test_json_prompt_skips_refine(self):
        state = new_state(WORKFLOW_VIDEO_GEN)
        state = apply_user_message(state, VEO_JSON)
        assert state["params"]["prompt"] == VEO_JSON
        assert state["params"]["prompt_format"] == "json"
        assert state["params"]["refine_resolved"] is True
        assert state["step"] == "awaiting_model"

    def test_json_with_model_in_same_message(self):
        state = new_state(WORKFLOW_VIDEO_GEN)
        state = apply_user_message(state, VEO_JSON + "\nuse veo 3.1 ultra")
        assert state["params"]["prompt"] == VEO_JSON
        assert state["params"]["model"] == "veo-3.1-ultra"
        assert state["params"]["duration"] == 8
        assert state["step"] == "awaiting_confirmation"

    def test_json_paste_without_intent_creates_unknown(self):
        state = new_state(WORKFLOW_UNKNOWN)
        state = apply_user_message(state, VEO_JSON)
        assert state["params"]["prompt"] == VEO_JSON
        assert state["step"] == "awaiting_type"

    def test_unknown_resolves_via_intent_keeping_prompt(self):
        state = new_state(WORKFLOW_UNKNOWN)
        state = apply_user_message(state, VEO_JSON)
        state = apply_user_message(state, "video")
        assert state["workflow_type"] == WORKFLOW_VIDEO_GEN
        assert state["params"]["prompt"] == VEO_JSON

    def test_model_inside_json_value_not_parsed(self):
        blob = '{"scene": "a billboard advertising veo 3.1", "mood": "retro"}'
        state = new_state(WORKFLOW_VIDEO_GEN)
        state = apply_user_message(state, blob)
        assert state["params"]["prompt"] == blob
        assert state["params"]["model"] is None  # mention was inside the JSON

    def test_capture_refined_json_candidate(self):
        state = new_state(WORKFLOW_VIDEO_GEN)
        state = apply_user_message(state, VEO_JSON)
        improved = '{"scene": "a neon-lit Tokyo alley at night", "camera": {"movement": "slow dolly"}, "style": "cinematic"}'
        # User asks for improvements; Gemini answers with a ✨ + json block.
        state_with_candidate = capture_refined_prompt(
            {**state, "params": {**state["params"], "refine_resolved": False}},
            "✨ Here is an improved version:\n```json\n" + improved + "\n```",
        )
        assert state_with_candidate["params"]["candidate_prompt"] == improved


# ---------------------------------------------------------------------------
# transitions: speech workflow
# ---------------------------------------------------------------------------
class TestSpeechWorkflow:
    def test_full_flow(self):
        state = new_state(WORKFLOW_SPEECH)
        assert state["step"] == "awaiting_text"

        state = apply_user_message(state, "Welcome to reelmotion, where ideas become videos!")
        assert state["params"]["speech_text"].startswith("Welcome")
        assert state["step"] == "confirm_text"

        state = apply_user_message(state, "yes")
        assert state["step"] == "awaiting_voice"

        state = apply_user_message(state, "Adam")
        assert state["params"]["voice_name"] == "Adam"
        assert state["step"] == "awaiting_confirmation"

    def test_default_voice_on_confirmation(self):
        state = new_state(WORKFLOW_SPEECH)
        state = apply_user_message(state, "Hello world, this is a longer test text")
        state = apply_user_message(state, "yes")
        state = apply_user_message(state, "ok")  # accept default
        assert state["params"]["voice_id"] == RACHEL_VOICE_ID
        assert state["step"] == "awaiting_confirmation"


# ---------------------------------------------------------------------------
# merge_extracted
# ---------------------------------------------------------------------------
class TestMergeExtracted:
    def test_fills_gaps_only(self):
        state = new_state(WORKFLOW_VIDEO_GEN)
        state["params"]["prompt"] = "my original prompt text"
        state["params"]["refine_resolved"] = True
        merged = merge_extracted(state, {
            "intent": "video_gen",
            "prompt": "SOMETHING ELSE",
            "model": "veo 3.1",
            "duration": 8,
        })
        assert merged["params"]["prompt"] == "my original prompt text"  # not overwritten
        assert merged["params"]["model"] == "veo-3.1"
        assert merged["params"]["duration"] == 8
        assert merged["step"] == "awaiting_confirmation"

    def test_from_none_state(self):
        merged = merge_extracted(None, {
            "intent": "image",
            "prompt": "a cat wearing a hat",
            "model": "GPT",
        })
        assert merged["workflow_type"] == WORKFLOW_IMAGE
        assert merged["params"]["model"] == "GPT"
        assert is_ready_for_confirmation(merged)

    def test_speech_extraction(self):
        merged = merge_extracted(None, {
            "intent": "speech",
            "prompt": "Welcome to the show everyone",
            "voice_name": "rachel",
        })
        assert merged["workflow_type"] == WORKFLOW_SPEECH
        assert merged["params"]["speech_text"] == "Welcome to the show everyone"
        assert merged["params"]["voice_id"] == RACHEL_VOICE_ID
        assert is_ready_for_confirmation(merged)

    def test_empty_extraction_is_noop(self):
        merged = merge_extracted(None, {})
        assert not is_ready_for_confirmation(merged)


# ---------------------------------------------------------------------------
# build_action_args
# ---------------------------------------------------------------------------
class TestBuildActionArgs:
    def _ready_video_state(self):
        state = new_state(WORKFLOW_VIDEO_GEN)
        state["params"].update({
            "prompt": VEO_JSON, "prompt_format": "json", "refine_resolved": True,
            "model": "veo-3.1", "duration": 8,
        })
        state["step"] = compute_step(state)
        return state

    def test_video_args_json_verbatim(self):
        result = build_action_args(self._ready_video_state())
        assert result is not None
        function_name, args = result
        assert function_name == "generate_video"
        assert args["prompt"] == VEO_JSON  # byte-identical
        assert args["model"] == "veo-3.1"
        assert args["duration"] == 8

    def test_seedance_resolution_included(self):
        state = self._ready_video_state()
        state["params"].update({"model": "seedance-2.0-fast", "duration": 5, "resolution": "1080p"})
        _, args = build_action_args(state)
        assert args["resolution"] == "720p"  # normalized for the fast tier

    def test_image_args_reference_types(self):
        state = new_state(WORKFLOW_IMAGE)
        state["params"].update({"prompt": "a cat", "refine_resolved": True, "model": "GPT"})

        _, args = build_action_args(state)
        assert args["image_type"] == 1

        _, args = build_action_args(state, ["http://x/1.png"])
        assert args["image_type"] == 2
        assert args["reference_image"] == "http://x/1.png"

        _, args = build_action_args(state, ["http://x/1.png", "http://x/2.png"])
        assert args["image_type"] == 3
        assert args["reference_images"] == ["http://x/1.png", "http://x/2.png"]

    def test_speech_args(self):
        state = new_state(WORKFLOW_SPEECH)
        state["params"].update({"speech_text": "Hello there", "text_confirmed": True})
        function_name, args = build_action_args(state)
        assert function_name == "generate_speech"
        assert args == {"text": "Hello there", "voice_id": RACHEL_VOICE_ID}

    def test_missing_fields_returns_none(self):
        state = new_state(WORKFLOW_VIDEO_GEN)
        state["params"]["prompt"] = "something"
        assert build_action_args(state) is None
        assert "model" in missing_fields(state)

    def test_unknown_type_returns_none(self):
        state = new_state(WORKFLOW_UNKNOWN)
        state["params"]["prompt"] = VEO_JSON
        assert build_action_args(state) is None


# ---------------------------------------------------------------------------
# state_context_note
# ---------------------------------------------------------------------------
class TestStateContextNote:
    def test_text_prompt_truncated(self):
        state = new_state(WORKFLOW_VIDEO_GEN)
        state["params"].update({"prompt": "x" * 300, "refine_resolved": True})
        note = state_context_note(state)
        assert "[SYSTEM CONTEXT — workflow state" in note
        assert "x" * 300 not in note  # truncated
        assert "model" in note  # listed as missing

    def test_json_prompt_flagged_verbatim(self):
        state = new_state(WORKFLOW_VIDEO_GEN)
        state["params"].update({
            "prompt": VEO_JSON, "prompt_format": "json", "refine_resolved": True,
            "model": "veo-3.1", "duration": 8,
        })
        note = state_context_note(state)
        assert "VERBATIM" in note
        assert "ready for cost confirmation" in note
        assert "VIDEO prompt" in note  # video-key hint
