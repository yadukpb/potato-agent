# 🥔 Potato Leaderboard

The challenge: **how weak can your hardware be and still run a real agentic voice LLM?**

Not "how fast can the fastest machine go" — the opposite. Slow entries are the
point. A Raspberry Pi Zero, a 2009 netbook, a Chromebook, a 1-vCPU free-tier
VM — that's exactly what belongs here.

## How to submit a row

```bash
python scripts/setup_models.sh   # if you haven't already
python scripts/benchmark.py
```

It prints a markdown row with your real hardware info (auto-detected — no
need to type anything by hand). Copy it, add it below sorted by **tokens/sec
ascending** (slowest at the top), and open a PR.

## Tiers

| Tier | Tokens/sec | Verdict |
|---|---|---|
| 🥔 Raw Potato | < 2 | It technically works |
| 🍠 Baked Potato | 2–6 | Usable, grab a coffee while it thinks |
| 🧈 Buttered Potato | 6–12 | Comfortable conversation pace |
| 🍟 Golden Fries | 12–25 | Snappy |
| 🚀 Potato Supreme | 25+ | Overkill, you absolute baller |

## Leaderboard

| CPU | Cores | RAM | Time to First Token | Tokens/sec | Tier |
|---|---|---|---|---|---|
| AMD Ryzen 5 5500U with Radeon Graphics | 6c/12t | 15GB | 1.48s | 8.9 | 🧈 Buttered Potato |

*Got something weaker? Open a PR — that's the whole point of this repo.*
