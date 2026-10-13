"""Voice mode's spoken equivalent of text mode's exact "/reset": a fixed,
exact-match set of natural-language phrases, not a substring/fuzzy match --
a false positive silently wipes conversation history."""
from potato_agent.agent import is_reset_command


def test_bare_reset_matches():
    assert is_reset_command("reset")


def test_phrase_variants_match():
    assert is_reset_command("clear history")
    assert is_reset_command("forget everything")
    assert is_reset_command("start over")
    assert is_reset_command("new conversation")


def test_case_and_punctuation_are_ignored():
    assert is_reset_command("Reset!")
    assert is_reset_command("Clear History.")
    assert is_reset_command("FORGET EVERYTHING")


def test_leading_politeness_words_are_stripped():
    assert is_reset_command("please reset")
    assert is_reset_command("can you reset the conversation")
    assert is_reset_command("hey, could you clear history")
    assert is_reset_command("okay please reset")


def test_whitespace_is_normalized():
    assert is_reset_command("  reset   the   conversation  ")


def test_unrelated_questions_do_not_match():
    assert not is_reset_command("what's the capital of France?")
    assert not is_reset_command("can you reset my password")
    assert not is_reset_command("tell me about the reset button on my router")


def test_reset_as_part_of_a_longer_request_does_not_match():
    assert not is_reset_command("reset and then tell me a joke")
    assert not is_reset_command("")
