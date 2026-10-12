# potato-agent

A fully agentic voice LLM that runs smoothly on a laptop with **no GPU, no VRAM, no cloud API**.

Talk to it. It can search the web, do math, read/write files, and (opt-in) run shell commands — all while staying entirely on your CPU and in your RAM. Built and daily-driven on a $400 laptop: AMD Ryzen 5 5500U, 14GB RAM, integrated graphics only.

If it can run here, it'll run on basically anything.

## Why

Most "local AI" projects quietly assume you have a GPU. This one doesn't. The whole pipeline — speech-to-text, the LLM, text-to-speech, and the agent loop — is chosen and tuned specifically for CPU-only, memory-constrained hardware. No framework overhead: the agent loop is ~80 lines of plain Python, not a LangChain graph.

## Architecture

```
  mic ──► faster-whisper (STT) ──► ReAct agent loop ──► Piper (TTS) ──► speaker
                                         │     │
                                         │     └─► llama.cpp (Qwen2.5-3B, Q4_K_M)
                                         ▼
                              tools: web_search · calculator
                                      file_op · run_shell_command
```

- **LLM**: [llama.cpp](https://github.com/ggerganov/llama.cpp) running Qwen2.5-3B-Instruct, Q4_K_M quantization (~2GB)
- **STT**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper), `base.en`, int8 on CPU
- **TTS**: [Piper](https://github.com/rhasspy/piper), real-time on CPU
- **Agent loop**: hand-rolled ReAct over JSON, grammar-constrained — no LangChain/LangGraph, so it stays fast and easy to hack on

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

# voice mode (push-to-talk: press Enter to record, Enter again to stop)
potato-agent
```

## Tools the agent can use

| Tool | What it does | Default |
|---|---|---|
| `calculator` | Safe AST-based arithmetic (no `eval`) | on |
| `web_search` | DuckDuckGo search, no API key needed | on |
| `file_op` | Read/write/list files, sandboxed to `workspace/` | on |
| `run_shell_command` | Runs a shell command | **off** — set `POTATO_ALLOW_SHELL=1`, and every command still needs a y/n confirmation |

## Benchmarks

Run `python scripts/benchmark.py` after setup and paste your row here.

| Hardware | RAM | Time to first token | Tokens/sec |
|---|---|---|---|
| AMD Ryzen 5 5500U (6c/12t) | 14GB | _run it_ | _run it_ |

PRs with your own hardware's numbers welcome — that table is the point.

## Configuration

All via environment variables (see `src/potato_agent/config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `POTATO_LLM_PATH` | `models/qwen2.5-3b-instruct-q4_k_m.gguf` | swap in a different GGUF model |
| `POTATO_LLM_CTX` | `4096` | context window |
| `POTATO_WHISPER_MODEL` | `base.en` | faster-whisper model size |
| `POTATO_AGENT_MAX_STEPS` | `6` | tool-call steps before giving up |
| `POTATO_ALLOW_SHELL` | `0` | enable the shell tool |

Swap the LLM for anything else llama.cpp supports (bigger model if you have more RAM, smaller/faster if you don't) — it's one GGUF file and one path.

## Project status

Early. Phase 1 (voice loop) and phase 2 (agentic tools) are both in; next up is wake-word/VAD instead of push-to-talk, and a proper eval set for the tool-calling reliability of small models. Issues and PRs welcome — the whole point of this repo is that people with modest hardware can actually run and improve it, not just read about it.

## License

MIT
