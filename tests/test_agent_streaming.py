"""agent.py's streaming JSON decoder is what lets TTS start speaking sentence
1 before the LLM finishes generating the rest of the reply (see tts.py /
voice_loop.py) -- these tests pin down both the raw string decoder and the
end-to-end on_sentence behavior, without needing a real model.
"""
from potato_agent.agent import (
    _FINAL_ANSWER_KEY_RE,
    _SENTENCE_BOUNDARY_RE,
    Agent,
    _decode_partial_json_string,
)


class FakeLLM:
    """Replays pre-built full JSON responses one character at a time, the
    same granularity llama.cpp's real token stream would arrive in (just
    finer -- char-by-char is a stricter test than token-by-token)."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self.calls = 0

    def chat_json_stream(self, messages, json_schema):
        raw = self._responses[self.calls]
        self.calls += 1
        for ch in raw:
            yield ch


# -- _decode_partial_json_string -------------------------------------------


def test_decode_plain_text():
    raw = '"hello world"'
    assert _decode_partial_json_string(raw, 1) == "hello world"


def test_decode_stops_at_unescaped_quote():
    raw = '"hello" extra'
    assert _decode_partial_json_string(raw, 1) == "hello"


def test_decode_handles_common_escapes():
    raw = r'"line1\nline2\ttabbed\rcr"'
    assert _decode_partial_json_string(raw, 1) == "line1\nline2\ttabbed\rcr"


def test_decode_handles_escaped_quote_and_backslash():
    raw = r'"she said \"hi\" and used a \\ backslash"'
    assert _decode_partial_json_string(raw, 1) == 'she said "hi" and used a \\ backslash'


def test_decode_handles_unicode_escape():
    raw = r'"AB"'
    assert _decode_partial_json_string(raw, 1) == "AB"


def test_decode_holds_back_incomplete_trailing_escape():
    # A lone trailing backslash hasn't got its escaped character yet --
    # must not guess, must decode everything before it and stop there.
    raw = '"partial text\\'
    assert _decode_partial_json_string(raw, 1) == "partial text"


def test_decode_holds_back_incomplete_unicode_escape():
    raw = '"abc\\u00'
    assert _decode_partial_json_string(raw, 1) == "abc"


def test_decode_grows_monotonically_as_more_arrives():
    full = '"The answer is forty-two."'
    seen = ""
    for end in range(1, len(full) + 1):
        decoded = _decode_partial_json_string(full[:end], 1)
        assert decoded.startswith(seen)
        seen = decoded
    assert seen == "The answer is forty-two."


# -- regexes ----------------------------------------------------------------


def test_final_answer_key_re_matches_with_varied_spacing():
    assert _FINAL_ANSWER_KEY_RE.search('{"thought": "x", "final_answer":   "hi') is not None
    assert _FINAL_ANSWER_KEY_RE.search('{"tool": "calculator"') is None


def test_sentence_boundary_re_splits_after_terminal_punctuation():
    assert _SENTENCE_BOUNDARY_RE.split("Hi there. How are you?") == ["Hi there.", "How are you?"]


# -- end-to-end: Agent.run() streaming on_sentence --------------------------


def test_run_streams_final_answer_sentences_in_order():
    llm = FakeLLM(['{"thought": "ok", "final_answer": "Hi there. How are you?"}'])
    agent = Agent(llm=llm)
    sentences = []

    answer = agent.run("hello", verbose=False, on_sentence=sentences.append)

    assert answer == "Hi there. How are you?"
    assert sentences == ["Hi there.", "How are you?"]


def test_run_does_not_stream_tool_call_steps():
    llm = FakeLLM(
        [
            '{"thought": "need calculator", "tool": "calculator", "args": {"expression": "2 + 2"}}',
            '{"thought": "done", "final_answer": "The answer is 4."}',
        ]
    )
    agent = Agent(llm=llm)
    sentences = []

    answer = agent.run("what's 2 + 2?", verbose=False, on_sentence=sentences.append)

    assert answer == "The answer is 4."
    assert sentences == ["The answer is 4."]
    assert llm.calls == 2


def test_run_without_on_sentence_still_returns_final_answer():
    llm = FakeLLM(['{"final_answer": "fine without a callback"}'])
    agent = Agent(llm=llm)
    assert agent.run("hi", verbose=False) == "fine without a callback"
