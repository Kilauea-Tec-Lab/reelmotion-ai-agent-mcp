"""Unit tests for pricing.py and generation_errors.py (no network/Redis needed)."""
import json

import pytest

from pricing import (
    IMAGE_COSTS,
    VIDEO_TOKEN_RATES,
    affordable_options,
    build_insufficient_balance_message,
    compute_kling_cost,
    compute_seedance2_cost,
    detect_language,
    estimate_generation_cost,
    is_insufficient_balance_message,
    is_spanish,
    normalize_kling_quality,
    speech_cost,
)
from generation_errors import (
    GENERATION_ERROR_PREFIX,
    SUPPORT_EMAIL,
    fallback_error_message,
    format_generation_error,
    is_generation_success,
    parse_backend_error,
    parse_generation_error,
    success_gen_type,
)


# ---------------------------------------------------------------------------
# speech_cost
# ---------------------------------------------------------------------------
class TestSpeechCost:
    def test_returns_zero_for_empty_text(self):
        assert speech_cost(0) == 0

    def test_bills_whole_1000_char_blocks(self):
        assert speech_cost(1) == 12
        assert speech_cost(1000) == 12
        assert speech_cost(1001) == 24
        assert speech_cost(2500) == 36

    def test_flash_is_half_price(self):
        assert speech_cost(1000, "eleven_flash_v2_5") == 6
        assert speech_cost(2500, "eleven_flash_v2_5") == 18

    def test_quality_models_share_the_full_rate(self):
        assert speech_cost(1000, "eleven_v3") == 12
        assert speech_cost(1000, "eleven_multilingual_v2") == 12

    def test_short_text_is_never_free(self):
        # The old flat tier charged 1 token for up to 500 chars, under cost.
        assert speech_cost(400) == 12


# ---------------------------------------------------------------------------
# estimate_generation_cost
# ---------------------------------------------------------------------------
class TestEstimateGenerationCost:
    def test_image_gpt_costs_seven(self):
        assert estimate_generation_cost("generate_image", {"model": "GPT"}) == 7

    def test_image_seedream_costs_four(self):
        assert estimate_generation_cost("generate_image", {"model": "Seedream"}) == 4

    def test_image_seedream_pro_costs_four(self):
        assert estimate_generation_cost("generate_image", {"model": "Seedream Pro"}) == 4

    def test_image_midjourney_costs_nine(self):
        assert estimate_generation_cost("generate_image", {"model": "Midjourney"}) == 10

    def test_image_quantity_multiplies_cost_for_supported_models(self):
        cost = estimate_generation_cost(
            "generate_image", {"model": "GPT", "quantity": 3}
        )
        assert cost == 21

    def test_image_quantity_ignored_for_seedream(self):
        # Seedream always bills for one image per call regardless of quantity.
        cost = estimate_generation_cost(
            "generate_image", {"model": "Seedream", "quantity": 5}
        )
        assert cost == 4

    def test_image_quantity_ignored_for_midjourney(self):
        cost = estimate_generation_cost(
            "generate_image", {"model": "Midjourney", "quantity": 4}
        )
        assert cost == 10

    def test_image_loose_model_name_is_normalized(self):
        assert estimate_generation_cost("generate_image", {"model": "nano-banana"}) == 8
        assert estimate_generation_cost("generate_image", {"model": "seedream 4.0"}) == 4
        assert estimate_generation_cost("generate_image", {"model": "seedream pro"}) == 4

    def test_image_freepik_is_not_a_valid_model(self):
        # Freepik was removed from the catalog — unknown model → no estimate.
        assert estimate_generation_cost("generate_image", {"model": "Freepik"}) is None

    def test_image_unknown_model_returns_none(self):
        assert estimate_generation_cost("generate_image", {"model": "dall-e"}) is None

    def test_video_veo_31_eight_seconds(self):
        cost = estimate_generation_cost(
            "generate_video", {"model": "veo-3.1", "duration": 8}
        )
        assert cost == 46 * 8

    def test_video_kling_v3_1080p_five_seconds(self):
        cost = estimate_generation_cost(
            "generate_video",
            {"model": "kling-v3", "duration": 5, "resolution": "1080p"},
        )
        assert cost == 14 * 5  # base route, 1080p

    def test_video_kling_turbo_720p_five_seconds(self):
        cost = estimate_generation_cost(
            "generate_video",
            {"model": "kling-v3-turbo", "duration": 5, "resolution": "720p"},
        )
        assert cost == 14 * 5  # turbo route, 720p

    def test_video_kling_v3_4k_only_on_base_route(self):
        cost = estimate_generation_cost(
            "generate_video",
            {"model": "kling-v3", "duration": 3, "resolution": "4k"},
        )
        assert cost == 46 * 3

    def test_video_kling_o3_edit_mode_uses_edit_rate(self):
        cost = estimate_generation_cost(
            "generate_video",
            {"model": "kling-o3", "duration": 6, "resolution": "1080p", "mode": "edit"},
        )
        assert cost == 19 * 6

    def test_video_kling_base_audio_surcharge(self):
        cost = estimate_generation_cost(
            "generate_video",
            {"model": "kling-v3", "duration": 5, "resolution": "720p", "sound": "on"},
        )
        assert cost == 14 * 5  # 720p + audio = 14/sec

    def test_video_kling_missing_duration_returns_none(self):
        assert estimate_generation_cost(
            "generate_video", {"model": "kling-v3", "resolution": "720p"}
        ) is None

    def test_video_seedance_720p_five_seconds(self):
        cost = estimate_generation_cost(
            "generate_video",
            {"model": "seedance-2.5", "duration": 5, "resolution": "720p"},
        )
        assert cost == 175

    def test_video_seedance_mini_1080p_downgrades_to_720p(self):
        cost = estimate_generation_cost(
            "generate_video",
            {"model": "seedance-2.0-mini", "duration": 5, "resolution": "1080p"},
        )
        assert cost == 12 * 5  # clamped to 720p

    def test_legacy_seedance_keys_resolve_to_new_models(self):
        # Deployed clients still send the retired keys.
        assert estimate_generation_cost(
            "generate_video",
            {"model": "seedance-2.0", "duration": 5, "resolution": "1080p"},
        ) == 85 * 5
        assert estimate_generation_cost(
            "generate_video",
            {"model": "seedance-2.0-fast", "duration": 5, "resolution": "720p"},
        ) == 12 * 5

    def test_video_kling_o1_is_flat_and_snaps_duration(self):
        assert estimate_generation_cost(
            "generate_video", {"model": "kling-o1", "duration": 5, "resolution": "1080p"}
        ) == 12 * 5
        # 7s is not offered; it snaps to 5s rather than billing 7.
        assert estimate_generation_cost(
            "generate_video", {"model": "kling-o1", "duration": 7}
        ) == 12 * 5

    def test_video_veo_lite_is_the_cheapest_veo(self):
        assert estimate_generation_cost(
            "generate_video", {"model": "veo-3.1-lite", "duration": 8}
        ) == 6 * 8

    def test_video_aleph2_rate(self):
        assert estimate_generation_cost(
            "generate_video", {"model": "runway-aleph", "duration": 5}
        ) == 33 * 5

    def test_video_seedance_reference_videos_use_discount_table(self):
        cost = estimate_generation_cost(
            "generate_video",
            {
                "model": "seedance-2.5",
                "duration": 5,
                "resolution": "720p",
                "reference_videos": ["https://example.com/v.mp4"],
            },
        )
        assert cost == 21 * 5

    def test_video_legacy_model_returns_none(self):
        assert (
            estimate_generation_cost("generate_video", {"model": "luma-labs", "duration": 5})
            is None
        )

    def test_video_missing_duration_returns_none(self):
        assert estimate_generation_cost("generate_video", {"model": "veo-3.1"}) is None

    def test_speech_uses_text_length(self):
        assert estimate_generation_cost("generate_speech", {"text": "x" * 600}) == 12

    def test_speech_missing_text_returns_none(self):
        assert estimate_generation_cost("generate_speech", {}) is None

    def test_unknown_function_returns_none(self):
        assert estimate_generation_cost("generate_music", {}) is None


# ---------------------------------------------------------------------------
# compute_seedance2_cost (moved from tools.py — keep behavior pinned)
# ---------------------------------------------------------------------------
class TestComputeSeedance2Cost:
    def test_normal_rate(self):
        assert compute_seedance2_cost("seedance-2.5", "1080p", 4) == 85 * 4

    def test_default_duration_is_five(self):
        assert compute_seedance2_cost("seedance-2.5", "480p", None) == 16 * 5

    def test_duration_clamped_to_thirty(self):
        assert compute_seedance2_cost("seedance-2.5", "480p", 99) == 16 * 30

    def test_mini_duration_still_stops_at_fifteen(self):
        assert compute_seedance2_cost("seedance-2.0-mini", "480p", 99) == 6 * 15


# ---------------------------------------------------------------------------
# Kling pricing + quality clamping
# ---------------------------------------------------------------------------
class TestKlingPricing:
    def test_base_route_matches_spec_example(self):
        # spec: kling-v3 1080p, 5s, no audio = 5 × 12 = 60
        assert compute_kling_cost("kling-v3", "base", "1080p", 5, False) == 70

    def test_4k_clamped_off_non_base_routes(self):
        # 4K requested for the edit route is clamped to 1080p (17/sec).
        assert normalize_kling_quality("kling-o3", "edit", "4k") == "1080p"
        assert compute_kling_cost("kling-o3", "edit", "4k", 4, False) == 19 * 4

    def test_turbo_never_4k(self):
        assert normalize_kling_quality("kling-v3-turbo", "turbo", "4k") == "1080p"

    def test_invalid_resolution_defaults_720p(self):
        assert normalize_kling_quality("kling-v3", "base", "480p") == "720p"

    def test_reference_duration_capped_at_ten(self):
        # 15s requested on the reference route is clamped to 10s (13/sec @720p).
        assert compute_kling_cost("kling-o3", "reference", "720p", 15, False) == 15 * 10

    def test_audio_ignored_off_base_route(self):
        # sound has no effect (and no surcharge) on the edit route.
        assert compute_kling_cost("kling-o3", "edit", "720p", 5, True) == 15 * 5


# ---------------------------------------------------------------------------
# affordable_options
# ---------------------------------------------------------------------------
class TestAffordableOptions:
    def test_balance_120_includes_expected_options(self):
        options = affordable_options(120)
        image_models = {i["model"] for i in options["images"]}
        assert image_models == set(IMAGE_COSTS)

        videos = {(v["model"], v["resolution"]): v for v in options["videos"]}
        assert videos[("runway-4.5", None)]["max_duration"] == 8
        assert videos[("runway-4.5", None)]["cost"] == 112
        # Mini is cheap enough to reach its longest clip on this balance.
        assert videos[("seedance-2.0-mini", "480p")]["max_duration"] == 15
        assert videos[("seedance-2.0-mini", "480p")]["cost"] == 90
        # veo-3.1 (46 x 8 = 368) is unaffordable
        assert ("veo-3.1", None) not in videos

    def test_balance_zero_affords_nothing(self):
        options = affordable_options(0)
        assert options["images"] == []
        assert options["videos"] == []
        assert options["speech_max_chars"] == 0

    def test_speech_max_chars_quotes_the_cheapest_voice(self):
        # Quoted at the flash rate (6 tokens / 1000 chars), the cheapest voice.
        assert affordable_options(1)["speech_max_chars"] == 0
        assert affordable_options(6)["speech_max_chars"] == 1000
        assert affordable_options(12)["speech_max_chars"] == 2000
        assert affordable_options(40)["speech_max_chars"] == 6000


# ---------------------------------------------------------------------------
# is_spanish + build_insufficient_balance_message
# ---------------------------------------------------------------------------
class TestLanguageAndMessage:
    def test_detects_spanish_confirmation(self):
        assert is_spanish("El costo será de 352 tokens. ¿Confirmas?") is True

    def test_defaults_to_english(self):
        assert is_spanish("Cost: 352 tokens. Do you confirm?") is False

    def test_spanish_message_contains_cost_balance_and_alternative(self):
        msg = build_insufficient_balance_message(352, 120, "es", affordable_options(120))
        assert "352" in msg
        assert "120" in msg
        assert "No tienes tokens suficientes" in msg
        assert "Seedream" in msg

    def test_english_message_contains_cost_balance_and_alternative(self):
        msg = build_insufficient_balance_message(352, 120, "en", affordable_options(120))
        assert "352" in msg
        assert "120" in msg
        assert "don't have enough tokens" in msg
        assert "Seedream" in msg

    def test_zero_balance_message_suggests_topup_only(self):
        msg = build_insufficient_balance_message(160, 0, "en", affordable_options(0))
        assert "top up" in msg.lower()
        assert "Seedream" not in msg


# ---------------------------------------------------------------------------
# detect_language (tri-state, used to resolve the conversation language)
# ---------------------------------------------------------------------------
class TestDetectLanguage:
    @pytest.mark.parametrize(
        "text",
        [
            "genera una imagen realista de un pug en el mundial",
            "quiero comprar más tokens",
            "hazme un video de una rana saltando",
            "¿cuántos tokens cuesta esto?",
            "necesito que confirmes antes de generar",
        ],
    )
    def test_clear_spanish_returns_es(self, text):
        assert detect_language(text) == "es"

    @pytest.mark.parametrize(
        "text",
        [
            "generate a realistic image of a pug at the world cup",
            "i want to buy more tokens",
            "make me a video of a frog jumping",
            "how much does this cost?",
            "where do i edit this video?",
        ],
    )
    def test_clear_english_returns_en(self, text):
        assert detect_language(text) == "en"

    @pytest.mark.parametrize(
        "text",
        ["veo 3.1 flash 8 sec", "gpt", "8", "5s", "", "   ", "nano banana 2"],
    )
    def test_ambiguous_or_empty_returns_none(self, text):
        # Bare model names / numbers / short tokens carry no clear signal — the
        # caller must fall back to the running conversation language.
        assert detect_language(text) is None


# ---------------------------------------------------------------------------
# is_insufficient_balance_message (recognises the code-generated block)
# ---------------------------------------------------------------------------
class TestIsInsufficientBalanceMessage:
    def test_recognises_spanish_block(self):
        msg = build_insufficient_balance_message(352, 10, "es", affordable_options(10))
        assert is_insufficient_balance_message(msg) is True

    def test_recognises_english_block(self):
        msg = build_insufficient_balance_message(352, 10, "en", affordable_options(10))
        assert is_insufficient_balance_message(msg) is True

    def test_ignores_unrelated_text(self):
        assert is_insufficient_balance_message("Which model would you like to use?") is False
        assert is_insufficient_balance_message("") is False


# ---------------------------------------------------------------------------
# parse_backend_error
# ---------------------------------------------------------------------------
class TestParseBackendError:
    def test_laravel_json_message(self):
        result = parse_backend_error(422, json.dumps({"message": "The prompt is invalid."}))
        assert result["category"] == "provider_validation"
        assert result["status"] == 422
        assert result["detail"] == "The prompt is invalid."

    def test_laravel_validation_errors_dict(self):
        body = json.dumps(
            {"message": "Validation failed", "errors": {"prompt": ["The prompt field is required."]}}
        )
        result = parse_backend_error(422, body)
        assert "Validation failed" in result["detail"]
        assert "The prompt field is required." in result["detail"]

    def test_html_body_is_stripped_and_truncated(self):
        body = "<html><body><h1>Server Error</h1>" + "<p>x</p>" * 500 + "</body></html>"
        result = parse_backend_error(500, body)
        assert result["category"] == "backend_unavailable"
        assert "<" not in result["detail"]
        assert len(result["detail"]) <= 500

    def test_status_classification(self):
        assert parse_backend_error(401, "")["category"] == "auth"
        assert parse_backend_error(402, "")["category"] == "insufficient_tokens"
        assert parse_backend_error(429, "")["category"] == "rate_limit"
        assert parse_backend_error(503, "")["category"] == "backend_unavailable"
        assert parse_backend_error(404, "")["category"] == "unknown"

    def test_empty_body_gets_default_detail(self):
        assert parse_backend_error(500, None)["detail"]


# ---------------------------------------------------------------------------
# format / parse generation error round-trip + fallbacks
# ---------------------------------------------------------------------------
class TestGenerationErrorFormat:
    def test_round_trip(self):
        text = format_generation_error("video", "provider_validation", 422, "Prompt rejected")
        assert text.startswith(GENERATION_ERROR_PREFIX)
        parsed = parse_generation_error(text)
        assert parsed == {
            "type": "video",
            "category": "provider_validation",
            "status": 422,
            "detail": "Prompt rejected",
        }

    def test_parse_returns_none_for_plain_errors(self):
        assert parse_generation_error("Error generating image: boom") is None
        assert parse_generation_error("") is None

    def test_fallback_messages_exist_in_both_languages(self):
        for category in (
            "auth", "insufficient_tokens", "provider_validation",
            "rate_limit", "backend_unavailable", "timeout", "unknown",
        ):
            for lang in ("es", "en"):
                msg = fallback_error_message(category, lang)
                assert msg.startswith("⚠️")
                assert len(msg) > 20

    def test_unknown_category_falls_back_to_unknown(self):
        assert fallback_error_message("weird", "en") == fallback_error_message("unknown", "en")

    def test_support_email_present_in_self_service_failures(self):
        # auth/unknown may need a human — surface the real support channel.
        for category in ("auth", "unknown"):
            for lang in ("es", "en"):
                assert SUPPORT_EMAIL in fallback_error_message(category, lang)

    def test_support_email_absent_from_self_service_categories(self):
        # A low balance or a bad prompt is self-service — don't send them to support.
        for category in ("insufficient_tokens", "provider_validation"):
            for lang in ("es", "en"):
                assert SUPPORT_EMAIL not in fallback_error_message(category, lang)


# ---------------------------------------------------------------------------
# is_generation_success / success_gen_type — classify the fast-path tool result
# ---------------------------------------------------------------------------
class TestSuccessClassification:
    @pytest.mark.parametrize("text", [
        "Video generated successfully with kling-v3.",
        "Images generated successfully with Seedream.",
        "Audio generated successfully (1234 bytes). Link generated automatically.",
        "Imagen generada exitosamente.",
    ])
    def test_success_signals(self, text):
        assert is_generation_success(text) is True

    @pytest.mark.parametrize("text", [
        "",
        "GENERATION_ERROR | category=provider_validation | status=422 | detail=bad",
        "Error executing generate_video: boom",
        "GENERATION_PROCESSING | type=video",
        "Video generation initiated but URL not immediately available. Check status later.",
    ])
    def test_non_success_signals(self, text):
        assert is_generation_success(text) is False

    def test_gen_type_classification(self):
        assert success_gen_type("Video generated successfully with kling-v3.") == "video"
        assert success_gen_type("Audio generated successfully (12 bytes).") == "audio"
        assert success_gen_type("Images generated successfully with GPT.") == "image"
        assert success_gen_type("something else") == "image"
