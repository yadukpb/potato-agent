# potato-agent 🥔

**Can your hardware run a real agentic voice LLM? Probably. Find out your tier.**

potato-agent is a fully agentic, voice-driven LLM — it searches the web, does math, reads/writes files, and talks back — running entirely on CPU. No GPU, no VRAM, no cloud API, no API key, no $20/month subscription.

This is **not** trying to be as smooth as ElevenLabs or OpenAI's voice products. It isn't, and chasing that is a losing game — those are polished, funded, cloud-optimized products, and a 3B model on a laptop CPU will always lose that comparison. So this repo plays a different game: the [**🥔 Potato Leaderboard**](./LEADERBOARD.md) — how weak can your hardware be and still run this?

```
$ python scripts/benchmark.py
============================================
   🧈 CERTIFIED BUTTERED POTATO 🧈
============================================
  CPU:   AMD Ryzen 5 5500U with Radeon Graphics
  Cores: 6c/12t
  RAM:   15GB
  TTFT:  1.48s
  Speed: 8.9 tok/s
  Tier:  🧈 Buttered Potato — comfortable conversation pace
============================================
```

Run the benchmark, get your tier, [open a PR with your row](./LEADERBOARD.md). A Raspberry Pi Zero, a 2009 netbook, a Chromebook, a free-tier 1-vCPU VM — that's exactly what belongs on the leaderboard. Slow is the point.

## Why

Most "local AI" projects quietly assume you have a GPU. This one doesn't, and it turns that constraint into the actual hook instead of apologizing for it. The whole pipeline — speech-to-text, the LLM, text-to-speech, and the agent loop — is chosen and tuned specifically for CPU-only, memory-constrained hardware. No framework overhead: the agent loop is ~80 lines of plain Python, not a LangChain graph.

## Architecture

```
  mic ──► webrtcvad (auto turn detection) ──► faster-whisper (STT) ──► ReAct agent loop ──► Piper (TTS, sentence-pipelined) ──► speaker
                                                                             │     │
                                                                             │     └─► llama.cpp (Qwen2.5-3B, Q4_K_M)
                                                                             ▼
                                                                  tools: web_search · calculator
                                                                          file_op · run_shell_command
```

- **LLM**: [llama.cpp](https://github.com/ggerganov/llama.cpp) running Qwen2.5-3B-Instruct, Q4_K_M quantization (~2GB)
- **VAD**: [webrtcvad](https://github.com/wiseman/py-webrtcvad) for continuous listening — no push-to-talk by default, no torch dependency
- **STT**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper), `base.en`, int8 on CPU
- **TTS**: [Piper](https://github.com/rhasspy/piper), real-time on CPU, sentence-pipelined (next sentence synthesizes while the current one plays)
- **Agent loop**: hand-rolled ReAct over JSON, grammar-constrained — no LangChain/LangGraph, so it stays fast and easy to hack on

### Conversational mode — what it is and isn't

By default (`potato-agent`) the mic is always listening; you don't press anything to talk, and the agent auto-detects when you start and stop speaking (via VAD). The mic closes while the agent is replying, so it never hears itself — that's what makes automatic turn-taking work without echo cancellation.

What it does **not** do yet: barge-in. You can't interrupt the agent mid-sentence by talking over it, the way ElevenLabs' conversational AI does — that needs real echo cancellation (to tell "you talking" apart from "mic picking up our own TTS output") or headphones to be reliable, and is a reasonable v2, not something bundled into this loop. If `POTATO_PUSH_TO_TALK=1` doesn't suit a noisy room, tune `POTATO_VAD_AGGRESSIVENESS` (0-3) first before falling back to it.

Heads up if you're extending the VAD: webrtcvad is tuned for real speech/silence, not synthetic test signals — pure white noise can trip it at any aggressiveness setting, so don't use a synthetic sine/noise generator to validate changes here. Test with your actual mic and room.

### Why not llama.cpp's built-in OpenAI-style tool calling?

We tried it first (`chat_format="chatml-function-calling"` in llama-cpp-python). It has two real bugs that make it unusable for a multi-turn agent loop: its chat template has no branch for tool-result messages (they get silently dropped from the prompt), and when the model responds with free text instead of a tool call, the library constrains that generation with the wrong grammar — one meant for picking a function name or stopping — which corrupts every plain answer into garbage.

Instead, `agent.py` builds its own small JSON Schema (`{"tool": ..., "args": ...}` or `{"final_answer": ...}`) and compiles it into a GBNF grammar via `LlamaGrammar.from_json_schema`, so llama.cpp masks logits during decoding and the model literally cannot generate invalid JSON. Same reliability win, without inheriting an upstream bug. See the comment at the top of `agent.py` for details.

## Quickstart

```bash
git clone <this-repo>
cd potato-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

./scripts/setup_models.sh    # downloads ~2.5GB of models

# text mode (fastest way to try it, no mic needed)
potato-agent-text

# voice mode — just start talking, no button to press
potato-agent

# or, if VAD misbehaves on your mic/room: push-to-talk fallback
POTATO_PUSH_TO_TALK=1 potato-agent
```

## Tools the agent can use

| Tool | What it does | Default |
|---|---|---|
| `calculator` | Safe AST-based arithmetic (no `eval`) | on |
| `web_search` | DuckDuckGo search, no API key needed | on |
| `file_op` | Read/write/list files, sandboxed to `workspace/` | on |
| `run_shell_command` | Runs a shell command | **off** — set `POTATO_ALLOW_SHELL=1`, and every command still needs a y/n confirmation |

## Benchmarks & the leaderboard

See [LEADERBOARD.md](./LEADERBOARD.md) — this is the actual point of the repo, not a side note. Run `python scripts/benchmark.py`, get your potato tier, open a PR with your row.

## Configuration

All via environment variables (see `src/potato_agent/config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `POTATO_LLM_PATH` | `models/qwen2.5-3b-instruct-q4_k_m.gguf` | swap in a different GGUF model |
| `POTATO_LLM_CTX` | `4096` | context window |
| `POTATO_WHISPER_MODEL` | `base.en` | faster-whisper model size |
| `POTATO_AGENT_MAX_STEPS` | `6` | tool-call steps before giving up |
| `POTATO_ALLOW_SHELL` | `0` | enable the shell tool |
| `POTATO_VAD_AGGRESSIVENESS` | `2` | 0-3, higher = more strict about filtering non-speech |
| `POTATO_PUSH_TO_TALK` | `0` | fall back to manual press-Enter instead of continuous listening |

Swap the LLM for anything else llama.cpp supports (bigger model if you have more RAM, smaller/faster if you don't) — it's one GGUF file and one path.

## Project status

Early. Phase 1 (voice loop), phase 2 (agentic tools), and continuous VAD-based conversation are all in. Verified so far: text-mode agent loop (plain Q&A, calculator, file ops) tested end-to-end on real hardware; VAD's silence-rejection and the sentence-splitting/pipelined TTS logic are unit-tested. **Not yet verified**: the live VAD loop against a real mic and real speech — that needs an actual person talking into an actual microphone, which hasn't happened yet. If you hit false starts or missed utterances, tune `POTATO_VAD_AGGRESSIVENESS` first and open an issue with what didn't work.

Next up: barge-in (interrupting the agent mid-sentence), speculative decoding for faster generation, and a proper eval set for small-model tool-calling reliability. Issues and PRs welcome — the whole point of this repo is that people with modest hardware can actually run and improve it, not just read about it.

## License

MIT
