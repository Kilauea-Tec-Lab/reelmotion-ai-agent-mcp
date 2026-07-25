"""
Unit tests for the frontend "actions" hints: the <<ACTION:...>> marker parsing
and the action-array computation in server.py (pure functions, no network).
"""
import pytest

import server
from server import (
    ACTION_EDITOR,
    ACTION_HOW_TO_USE,
    ACTION_SUPPORT,
    ACTION_TOKENS_SALE,
    LOW_BALANCE_THRESHOLD,
    compute_response_actions,
    detect_user_requested_actions,
    extract_action_markers,
)


# --- extract_action_markers ------------------------------------------------

def test_no_marker_returns_text_unchanged_and_empty_actions():
    text = "Just a normal reply with no buttons."
    clean, actions = extract_action_markers(text)
    assert clean == text
    assert actions == []


def test_editor_marker_is_extracted_and_stripped():
    clean, actions = extract_action_markers(
        "Your video is ready! 🎬\n<<ACTION:editor>>"
    )
    assert actions == [ACTION_EDITOR]
    assert "<<" not in clean
    assert "ACTION" not in clean
    assert clean == "Your video is ready! 🎬"


def test_tokens_sale_marker_is_extracted_and_stripped():
    clean, actions = extract_action_markers(
        "Te quedan muy pocos tokens, te conviene recargar.\n<<ACTION:tokens_sale>>"
    )
    assert actions == [ACTION_TOKENS_SALE]
    assert "tokens_sale" not in clean
    assert clean == "Te quedan muy pocos tokens, te conviene recargar."


def test_both_markers_extracted_in_order():
    clean, actions = extract_action_markers(
        "¡Listo! Te quedan pocos tokens.\n<<ACTION:editor>>\n<<ACTION:tokens_sale>>"
    )
    assert actions == [ACTION_EDITOR, ACTION_TOKENS_SALE]
    assert "<<" not in clean


def test_marker_matching_is_case_insensitive_and_tolerates_spaces():
    clean, actions = extract_action_markers("Done. <<  action : EDITOR  >>")
    assert actions == [ACTION_EDITOR]
    assert "action" not in clean.lower()


def test_duplicate_markers_are_deduped():
    clean, actions = extract_action_markers(
        "<<ACTION:editor>> text <<ACTION:editor>>"
    )
    assert actions == [ACTION_EDITOR]


def test_how_to_use_marker_is_extracted_and_stripped():
    clean, actions = extract_action_markers(
        "No worries! Here's a quick overview.\n<<ACTION:how_to_use>>"
    )
    assert actions == [ACTION_HOW_TO_USE]
    assert "how_to_use" not in clean
    assert clean == "No worries! Here's a quick overview."


def test_empty_or_none_text_is_safe():
    assert extract_action_markers("") == ("", [])
    assert extract_action_markers(None) == (None, [])


def test_inline_marker_does_not_leave_double_space():
    clean, actions = extract_action_markers("see <<ACTION:editor>> for more")
    assert actions == [ACTION_EDITOR]
    assert clean == "see for more"  # single space, no marker leakage


def test_horizontal_whitespace_in_reply_is_preserved():
    # Indented/code-like content must survive marker stripping untouched.
    text = "def f():\n    return 1\n<<ACTION:editor>>"
    clean, actions = extract_action_markers(text)
    assert actions == [ACTION_EDITOR]
    assert "    return 1" in clean  # 4-space indent intact


# --- compute_response_actions ----------------------------------------------

FILES = [{"url": "https://example.com/x.jpg", "type": "image"}]
HEALTHY_BALANCE = 500
LOW_BALANCE = max(0, LOW_BALANCE_THRESHOLD - 1)


def test_files_present_yields_editor():
    actions = compute_response_actions(FILES, None, HEALTHY_BALANCE)
    assert actions == [ACTION_EDITOR]


def test_insufficient_block_yields_tokens_sale():
    block = {"required": 352, "available": 10}
    actions = compute_response_actions([], block, 10)
    assert actions == [ACTION_TOKENS_SALE]


def test_low_balance_yields_tokens_sale():
    actions = compute_response_actions([], None, LOW_BALANCE)
    assert actions == [ACTION_TOKENS_SALE]


def test_healthy_balance_no_files_no_marker_yields_nothing():
    actions = compute_response_actions([], None, HEALTHY_BALANCE)
    assert actions == []


def test_unknown_balance_never_triggers_tokens_sale():
    actions = compute_response_actions([], None, None)
    assert actions == []


def test_editor_marker_adds_editor_without_files():
    actions = compute_response_actions(
        [], None, HEALTHY_BALANCE, marker_actions=[ACTION_EDITOR]
    )
    assert actions == [ACTION_EDITOR]


def test_tokens_sale_marker_adds_it_even_with_healthy_balance():
    actions = compute_response_actions(
        [], None, HEALTHY_BALANCE, marker_actions=[ACTION_TOKENS_SALE]
    )
    assert actions == [ACTION_TOKENS_SALE]


def test_how_to_use_marker_passes_through():
    actions = compute_response_actions(
        [], None, HEALTHY_BALANCE, marker_actions=[ACTION_HOW_TO_USE]
    )
    assert actions == [ACTION_HOW_TO_USE]


def test_both_sources_combine_and_keep_order():
    # files -> editor (mechanical), marker -> tokens_sale (conversational)
    actions = compute_response_actions(
        FILES, None, HEALTHY_BALANCE, marker_actions=[ACTION_TOKENS_SALE]
    )
    assert actions == [ACTION_EDITOR, ACTION_TOKENS_SALE]


def test_editor_not_duplicated_when_files_and_marker_both_present():
    actions = compute_response_actions(
        FILES, None, HEALTHY_BALANCE, marker_actions=[ACTION_EDITOR]
    )
    assert actions == [ACTION_EDITOR]


def test_low_balance_threshold_is_positive():
    # Sanity: the threshold mirrors the cheapest video and must be > 0,
    # otherwise tokens_sale on low balance could never fire.
    assert server.LOW_BALANCE_THRESHOLD > 0


# --- detect_user_requested_actions -----------------------------------------

@pytest.mark.parametrize("message", [
    "¿dónde edito el video?",
    "donde edito esto",
    "cómo edito mi video?",
    "quiero editar el video",
    "llévame al editor",
    "ir al editor por favor",
    "where do I edit this?",
    "how do I edit the video",
    "take me to the editor",
    "I want to edit my video",
    "open the editor",
])
def test_editor_request_phrases_detected(message):
    assert detect_user_requested_actions(message) == [ACTION_EDITOR]


@pytest.mark.parametrize("message", [
    "quiero comprar más tokens",
    "donde compro tokens",
    "cómo compro tokens",
    "necesito más tokens",
    "quiero recargar tokens",
    "recargar saldo",
    "I want to buy tokens",
    "buy more tokens",
    "where can I get more tokens",
    "I need to top up",
    "how do I recharge?",
])
def test_tokens_sale_request_phrases_detected(message):
    assert detect_user_requested_actions(message) == [ACTION_TOKENS_SALE]


def test_request_detection_handles_both_in_one_message():
    msg = "ya casi no tengo tokens, quiero comprar más y luego ir al editor"
    assert detect_user_requested_actions(msg) == [ACTION_EDITOR, ACTION_TOKENS_SALE]


@pytest.mark.parametrize("message", [
    "how do I subscribe to a plan?",
    "I want to subscribe",
    "what's included in the subscription?",
    "which plan should I get?",
    "can I upgrade my plan?",
    "quiero suscribirme",
    "¿cómo funciona la suscripción?",
    "¿qué planes tienen?",
    "quiero cambiar de plan",
])
def test_subscription_phrases_yield_tokens_sale(message):
    assert detect_user_requested_actions(message) == [ACTION_TOKENS_SALE]


@pytest.mark.parametrize("message", [
    "i am confused",
    "I'm confused, what do I do now?",
    "im lost",
    "I don't understand this",
    "how does this work?",
    "how to use this platform?",
    "what can you do?",
    "estoy confundido",
    "no entiendo qué hacer",
    "¿cómo funciona esto?",
    "¿cómo se usa la plataforma?",
    "estoy perdida",
])
def test_confused_phrases_yield_how_to_use(message):
    assert detect_user_requested_actions(message) == [ACTION_HOW_TO_USE]


@pytest.mark.parametrize("message", [
    "quiero hablar con una persona",
    "necesito hablar con alguien de soporte",
    "quiero hablar con un humano",
    "atención al cliente por favor",
    "quiero reportar un problema con mi cuenta",
    "quiero un reembolso",
    "i want to talk to a human",
    "can i speak to someone?",
    "how do i contact support?",
    "i need support with my account",
    "i want a refund",
    "i want to report a bug",
])
def test_support_phrases_yield_support(message):
    assert detect_user_requested_actions(message) == [ACTION_SUPPORT]


def test_support_marker_is_extracted_and_stripped():
    clean, actions = extract_action_markers(
        "Claro, te paso con una persona: +1 555-748-1227.\n<<ACTION:support>>"
    )
    assert actions == [ACTION_SUPPORT]
    assert "ACTION" not in clean
    assert clean == "Claro, te paso con una persona: +1 555-748-1227."


def test_support_marker_passes_through():
    actions = compute_response_actions(
        [], None, HEALTHY_BALANCE, marker_actions=[ACTION_SUPPORT]
    )
    assert actions == [ACTION_SUPPORT]


@pytest.mark.parametrize("message", [
    "genera una imagen realista de un pug en el mundial",
    "usa veo 3.1",
    "ok",
    "describe esta imagen",
    "quiero generar con mis tokens",  # USE tokens, not buy — must NOT fire
    "¿cuántos tokens cuesta?",        # asking cost, not buying
    "mi plan es hacer un video de mi perro",   # "plan" as intention, not subscription
    "a confused cat looking at a mirror",      # generation prompt, not user confusion
    "una persona real caminando por la playa",  # generation prompt, not escalation
    "a real person walking on the beach",
    "",
])
def test_unrelated_messages_request_no_actions(message):
    assert detect_user_requested_actions(message) == []


def test_request_detection_is_accent_insensitive():
    # Same phrase with and without accents must both match.
    assert detect_user_requested_actions("donde edito") == [ACTION_EDITOR]
    assert detect_user_requested_actions("DÓNDE EDITO") == [ACTION_EDITOR]


def test_full_pipeline_user_asks_to_edit_existing_asset():
    # No fresh files (asset was generated earlier), bot reply has no marker,
    # but the user asked where to edit -> editor action still fires.
    _clean, marker_actions = extract_action_markers(
        "Para editar, dime qué cambios quieres."
    )
    for action in detect_user_requested_actions("¿dónde edito esto?"):
        if action not in marker_actions:
            marker_actions.append(action)
    actions = compute_response_actions([], None, 500, marker_actions)
    assert actions == [ACTION_EDITOR]


def test_full_pipeline_user_asks_to_buy_tokens_with_healthy_balance():
    _clean, marker_actions = extract_action_markers(
        "¡Claro! Puedes recargar tokens en la sección de planes."
    )
    for action in detect_user_requested_actions("quiero comprar más tokens"):
        if action not in marker_actions:
            marker_actions.append(action)
    actions = compute_response_actions([], None, 999, marker_actions)
    assert actions == [ACTION_TOKENS_SALE]
