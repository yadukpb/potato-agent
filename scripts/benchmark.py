#!/usr/bin/env python3
"""Benchmark + potato-tier certification.

This is the actual point of the repo, not a side script: run it, get your
tier, paste the printed row into LEADERBOARD.md as a PR. See LEADERBOARD.md
for the challenge — how weak can your hardware be and still run this?
"""
import platform
import time

import cpuinfo
import psutil

from potato_agent.llm import LocalLLM
from potato_agent.tiers import get_tier

PROMPT = "Explain in 3 sentences why the sky is blue."


def _cpu_name() -> str:
    try:
        name = cpuinfo.get_cpu_info().get("brand_raw")
        if name:
            return name
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown CPU"


def _ram_gb() -> float:
    try:
        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        return float("nan")


def main():
    cpu_name = _cpu_name()
    cores = psutil.cpu_count(logical=False) or "?"
    threads = psutil.cpu_count(logical=True) or "?"
    ram_gb = _ram_gb()

    print(f"CPU: {cpu_name} ({cores}c/{threads}t)")
    print(f"RAM: {ram_gb:.0f}GB")
    print("Loading model...")

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

    ttft = (first_token_time - start) if first_token_time else 0.0
    total = end - start
    tps = token_count / total if total > 0 else 0.0
    emoji, tier_name, blurb = get_tier(tps)

    print(f"\nTime to first token: {ttft:.2f}s")
    print(f"Tokens/sec: {tps:.1f}")
    print(f"\nOutput:\n{text}\n")

    badge_width = 44
    print("=" * badge_width)
    print(f"  {emoji} CERTIFIED {tier_name.upper()} {emoji}".center(badge_width))
    print("=" * badge_width)
    print(f"  CPU:   {cpu_name}")
    print(f"  Cores: {cores}c/{threads}t")
    print(f"  RAM:   {ram_gb:.0f}GB")
    print(f"  TTFT:  {ttft:.2f}s")
    print(f"  Speed: {tps:.1f} tok/s")
    print(f"  Tier:  {emoji} {tier_name} — {blurb}")
    print("=" * badge_width)

    print("\nPaste this row into LEADERBOARD.md (PRs welcome):\n")
    print(f"| {cpu_name} | {cores}c/{threads}t | {ram_gb:.0f}GB | {ttft:.2f}s | {tps:.1f} | {emoji} {tier_name} |")

    print("\nShare-sized brag:\n")
    print(
        f'"{emoji} My {cpu_name} just ran an agentic voice LLM at {tps:.1f} tok/s '
        f'— zero GPU. Certified {tier_name} tier. #potato-agent"'
    )


if __name__ == "__main__":
    main()
