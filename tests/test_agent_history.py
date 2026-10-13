"""Multi-turn conversation memory: Agent keeps the last history_turns
(user, assistant) pairs from *completed* turns and feeds them back into
every subsequent prompt, so a follow-up question has context -- but a
turn's internal tool-call scratch never ends up in that history."""
from potato_agent.agent import SYSTEM_PROMPT, Agent


class FakeLLM:
    """Replays one full JSON response per call and records the exact
    `messages` list it was given, so tests can assert on what the agent
    actually sent -- the real point of these tests."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self.calls = 0
        self.seen_messages: list[list[dict]] = []

    def chat_json_stream(self, messages, json_schema):
        self.seen_messages.append(messages)
        raw = self._responses[self.calls]
        self.calls += 1
        for ch in raw:
            yield ch


def test_first_turn_has_no_history():
    llm = FakeLLM(['{"final_answer": "hi there"}'])
    agent = Agent(llm=llm)
    agent.run("hello", verbose=False)
    assert llm.seen_messages[0] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "hello"},
    ]


def test_second_turn_includes_prior_exchange():
    llm = FakeLLM(['{"final_answer": "Paris."}', '{"final_answer": "About 2.1 million."}'])
    agent = Agent(llm=llm)

    agent.run("what's the capital of France?", verbose=False)
    agent.run("what's its population?", verbose=False)

    second_call_messages = llm.seen_messages[1]
    assert second_call_messages == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "what's the capital of France?"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "what's its population?"},
    ]


def test_tool_call_scratch_is_not_remembered():
    llm = FakeLLM(
        [
            '{"tool": "calculator", "args": {"expression": "2 + 2"}}',
            '{"final_answer": "4."}',
        ]
    )
    agent = Agent(llm=llm)

    agent.run("what's 2 + 2?", verbose=False)

    # Only the clean (user, final_answer) pair should have been kept --
    # no tool-call JSON or "Tool result: ..." scratch messages.
    assert agent.history == [
        {"role": "user", "content": "what's 2 + 2?"},
        {"role": "assistant", "content": "4."},
    ]


def test_history_is_capped_at_history_turns():
    responses = [f'{{"final_answer": "answer {i}"}}' for i in range(5)]
    llm = FakeLLM(responses)
    agent = Agent(llm=llm, history_turns=2)

    for i in range(5):
        agent.run(f"question {i}", verbose=False)

    # Capped to the last 2 pairs (4 messages), oldest dropped first.
    assert agent.history == [
        {"role": "user", "content": "question 3"},
        {"role": "assistant", "content": "answer 3"},
        {"role": "user", "content": "question 4"},
        {"role": "assistant", "content": "answer 4"},
    ]


def test_clear_history_resets_to_a_fresh_conversation():
    llm = FakeLLM(['{"final_answer": "one"}', '{"final_answer": "two"}'])
    agent = Agent(llm=llm)

    agent.run("first", verbose=False)
    agent.clear_history()
    agent.run("second", verbose=False)

    assert llm.seen_messages[1] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "second"},
    ]


def test_step_limit_fallback_is_still_remembered():
    # Every step returns a tool call, never a final_answer, so the agent
    # exhausts max_steps -- the fallback message should still end up in
    # history, since it's what the user actually heard/read for that turn.
    llm = FakeLLM(['{"tool": "calculator", "args": {"expression": "1"}}'] * 3)
    agent = Agent(llm=llm, max_steps=3)

    answer = agent.run("loop forever", verbose=False)

    assert "couldn't finish" in answer
    assert agent.history == [
        {"role": "user", "content": "loop forever"},
        {"role": "assistant", "content": answer},
    ]
