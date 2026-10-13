"""Pure-logic pieces of tts.py: sentence splitting and the punctuation-driven
pause/prosody heuristics documented in tts.py's module docstring as "better
rhythm, not learned emotion" -- these tests just pin that arithmetic down."""
from potato_agent.tts import (
    _BASE_NOISE_SCALE,
    _BASE_NOISE_W_SCALE,
    _pause_seconds,
    _prosody_for,
    split_sentences,
)


def test_split_sentences_basic():
    assert split_sentences("Hi there. How are you?") == ["Hi there.", "How are you?"]


def test_split_sentences_strips_and_drops_empty():
    assert split_sentences("  One sentence only.  ") == ["One sentence only."]
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_sentences_handles_exclamation_and_ellipsis():
    result = split_sentences("Wait... what?! Really.")
    assert result == ["Wait...", "what?!", "Really."]


def test_pause_seconds_trailing_off():
    assert _pause_seconds("I was thinking...") == 0.55


def test_pause_seconds_excited_or_question():
    assert _pause_seconds("Really!") == 0.30
    assert _pause_seconds("Are you sure?") == 0.30


def test_pause_seconds_plain_statement():
    assert _pause_seconds("That's the answer.") == 0.25


def test_pause_seconds_fragment_no_terminal_punctuation():
    assert _pause_seconds("just a fragment") == 0.12


def test_pause_seconds_ignores_trailing_whitespace():
    assert _pause_seconds("Done.   ") == 0.25


def test_prosody_for_exclamation_speeds_up_and_adds_energy():
    cfg = _prosody_for("This is great!")
    assert cfg.length_scale == 0.92
    assert cfg.noise_scale == _BASE_NOISE_SCALE + 0.08


def test_prosody_for_question_adds_pitch_variance():
    cfg = _prosody_for("Is this right?")
    assert cfg.noise_w_scale == _BASE_NOISE_W_SCALE + 0.15


def test_prosody_for_plain_statement_is_default():
    default = _prosody_for("")
    cfg = _prosody_for("Just a statement.")
    assert cfg.length_scale == default.length_scale
    assert cfg.noise_scale == default.noise_scale
    assert cfg.noise_w_scale == default.noise_w_scale
