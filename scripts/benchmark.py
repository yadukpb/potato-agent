#!/usr/bin/env python3
"""Quick CPU inference benchmark: time-to-first-token + tokens/sec.

Run after setup_models.sh. Prints a markdown row you can paste into BENCHMARKS.md.
"""
import platform
import time

from potato_agent.llm import LocalLLM

PROMPT = "Explain in 3 sentences why the sky is blue."


def main():
    llm = LocalLLM()
    messages = [{"role": "user", "content": PROMPT}]

    start = time.perf_counter()
    first_token_time = None
    token_count = 0
    text = ""
    for chunk in llm.chat_stream(messages, max_tokens=200):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        token_count += 1
        text += chunk
    end = time.perf_counter()

    ttft = (first_token_time - start) if first_token_time else 0
    total = end - start
    tps = token_count / total if total > 0 else 0

    cpu = platform.processor() or platform.machine()
    print(f"\nCPU: {cpu}")
    print(f"Time to first token: {ttft:.2f}s")
    print(f"Total time: {total:.2f}s for ~{token_count} chunks")
    print(f"Approx tokens/sec: {tps:.1f}")
    print(f"\nOutput:\n{text}\n")
    print(f"| {cpu} | {ttft:.2f}s | {tps:.1f} tok/s |")


if __name__ == "__main__":
    main()
