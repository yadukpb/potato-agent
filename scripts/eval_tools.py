#!/usr/bin/env python3
"""Eval: does the small on-CPU model pick the right tool (or correctly pick
no tool) on the first step, across a fixed set of prompts?

This is a quality gate on the model/prompt pair, not on agent.py's logic --
the ReAct loop itself is covered by tests/test_agent_history.py and
tests/test_agent_streaming.py against a FakeLLM. This script instead runs
the real model (loads models/qwen2.5-3b-instruct-q4_k_m.gguf, same as
production) against real prompts and checks the *first* step's tool choice
via Agent.run()'s on_step callback -- not just whether the final answer
looks right, since a model that calls calculator on a general-knowledge
question and still stumbles into a correct-looking final answer is a
reliability problem this is meant to catch, not credit.

A fresh Agent is used per case (no shared conversation history) so one
case's turn can't bias the next one's tool choice.

Run: python scripts/eval_tools.py
"""
import time

from potato_agent.agent import Agent

# Each case checks only the *first* step's tool choice (or that it skipped
# tools entirely for "none"), not the final answer text -- judging answer
# correctness would need a second judge model and isn't what this is for.
CASES = [
    # (prompt, expected tool name, or None for "should answer directly")
    ("What's 847 times 23?", "calculator"),
    ("If I have 15 apples, give away 6, then buy 9 more, how many do I have?", "calculator"),
    ("What's the square root of 169?", "calculator"),
    ("Divide 100 by 4.", "calculator"),
    ("What's the latest version of Python?", "web_search"),
    ("Who won the most recent Super Bowl?", "web_search"),
    ("What's today's weather in Tokyo?", "web_search"),
    ("What is the current price of Bitcoin?", "web_search"),
    ("Write 'hello world' to a file called test.txt", "file_op"),
    ("List the files in the workspace.", "file_op"),
    ("Read the contents of greeting.txt", "file_op"),
    ("What's the capital of France?", None),
    ("What color do you get mixing blue and yellow?", None),
    ("Say hello in Spanish.", None),
    ("Tell me a fun fact about octopuses.", None),
    ("What does the word 'ephemeral' mean?", None),
]


def _first_step_tool(parsed: dict) -> str | None:
    return parsed.get("tool") if "tool" in parsed else None


def main():
    print("Loading model...\n")
    results = []

    for prompt, expected_tool in CASES:
        agent = Agent(max_steps=1)  # only need the first step's choice
        steps: list[dict] = []
        start = time.perf_counter()
        try:
            agent.run(prompt, verbose=False, on_step=steps.append)
        except Exception as e:
            results.append((prompt, expected_tool, "ERROR", False, time.perf_counter() - start))
            print(f"  ERROR on {prompt!r}: {e}")
            continue
        elapsed = time.perf_counter() - start

        actual_tool = _first_step_tool(steps[0]) if steps else None
        passed = actual_tool == expected_tool
        results.append((prompt, expected_tool, actual_tool, passed, elapsed))

        mark = "PASS" if passed else "FAIL"
        want = expected_tool or "(no tool)"
        got = actual_tool or "(no tool)"
        print(f"[{mark}] {elapsed:5.1f}s  want={want:<12} got={got:<12} {prompt}")

    total = len(results)
    correct = sum(1 for *_, passed, _ in results if passed)

    print("\n" + "=" * 60)
    print(f"Tool-choice accuracy: {correct}/{total} ({100 * correct / total:.0f}%)")
    print("=" * 60)

    failures = [r for r in results if not r[3]]
    if failures:
        print("\nFailed cases:")
        for prompt, expected_tool, actual_tool, _, _ in failures:
            print(f"  - {prompt!r}: wanted {expected_tool or '(no tool)'}, got {actual_tool or '(no tool)'}")


if __name__ == "__main__":
    main()
