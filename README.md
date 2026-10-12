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

- **LLM**: [llama.cpp](https://github.com/ggerganov/llama.cpp) running Qwen2.5-3B-Instruct, Q4_K_M quantization (~2GB), with flash attention and prompt-lookup speculative decoding on by default
- **VAD**: [webrtcvad](https://github.com/wiseman/py-webrtcvad) for continuous listening — no push-to-talk by default, no torch dependency
- **STT**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper), `base.en`, int8 on CPU
- **TTS**: [Piper](https://github.com/rhasspy/piper), real-time on CPU (measured ~18x), sentence-pipelined and streamed sentence-by-sentence as the LLM generates, with punctuation-driven pauses and pacing
- **Agent loop**: hand-rolled ReAct over JSON, grammar-constrained — no LangChain/LangGraph, so it stays fast and easy to hack on

### Conversational mode — what it is and isn't

By default (`potato-agent`) the mic is always listening; you don't press anything to talk, and the agent auto-detects when you start and stop speaking (via VAD).

**Barge-in**: you can interrupt the agent mid-reply by just talking over it. There's no real echo cancellation here (that's what ElevenLabs' conversational AI uses to tell "you talking" apart from "mic picking up our own TTS output") — instead, barge-in uses a stricter, slower-to-trigger speech-onset threshold while the agent is actually speaking (`POTATO_VAD_BARGE_IN_FRAMES`, ~300ms vs. the normal ~90ms) to resist false triggers from speaker bleed. It's reliable with headphones; without them it can occasionally misfire on a loud speaker setup. Set `POTATO_BARGE_IN=0` to disable it and fall back to strict turn-taking. If `POTATO_PUSH_TO_TALK=1` doesn't suit a noisy room, tune `POTATO_VAD_AGGRESSIVENESS` (0-3) first before falling back to it (push-to-talk also disables barge-in, since there's no continuous mic stream to listen on while speaking).

**Streaming speech**: the agent doesn't wait for its whole reply before speaking. `agent.run()`'s `on_sentence` callback fires per sentence as the LLM finishes generating it — by incrementally decoding the JSON string value straight out of the grammar-constrained token stream — so TTS can start on sentence 1 while the model is still generating the rest.

**Startup warmup**: the first LLM call in a process pays a one-time ~7-9s cost (llama.cpp backend/thread-pool init), and separately, llama-cpp-python caches the longest matching prompt *prefix* across calls and skips re-evaluating it. Since every turn's prompt starts with the same system prompt, `Agent.warm_up()` pre-runs a throwaway generation with that exact prompt during "Loading models...", so the KV cache is already warm by the time you can talk — dropped the first real turn from ~9-11s to ~2-3s.

Heads up if you're extending the VAD: webrtcvad is tuned for real speech/silence, not synthetic test signals — pure white noise can trip it at any aggressiveness setting, so don't use a synthetic sine/noise generator to validate changes here. Test with your actual mic and room.

### Why not llama.cpp's built-in OpenAI-style tool calling?

We tried it first (`chat_format="chatml-function-calling"` in llama-cpp-python). It has two real bugs that make it unusable for a multi-turn agent loop: its chat template has no branch for tool-result messages (they get silently dropped from the prompt), and when the model responds with free text instead of a tool call, the library constrains that generation with the wrong grammar — one meant for picking a function name or stopping — which corrupts every plain answer into garbage.

Instead, `agent.py` builds its own small JSON Schema (`{"tool": ..., "args": ...}` or `{"final_answer": ...}`) and compiles it into a GBNF grammar via `LlamaGrammar.from_json_schema`, so llama.cpp masks logits during decoding and the model literally cannot generate invalid JSON. Same reliability win, without inheriting an upstream bug. See the comment at the top of `agent.py` for details.

### Another upstream bug: speculative decoding's logits buffer

Enabling `llama-cpp-python`'s speculative decoding (`draft_model=...`) silently forces its internal `logits_all` flag to `True` for decoding — but the `scores` buffer it writes into is still sized off the separate `logits_all` *constructor parameter*, which defaults to `False`. Pass a `draft_model` without also passing `logits_all=True` yourself, and that buffer comes out `n_batch`-sized (512) instead of `n_ctx`-sized (4096); past the first ~512 tokens, `eval()` crashes with `could not broadcast input array ... into shape (0,)`. Worked around in `llm.py` by keeping the two in sync explicitly (`logits_all=config.LLM_SPECULATIVE`) — see its comment for the full trace.

### Evaluating a TTS swap: why Kokoro didn't make the cut

[Kokoro](https://github.com/thewh1teagle/kokoro-onnx) (StyleTTS2-based, ~82M params) is a genuinely more expressive, more natural-sounding open TTS model than Piper, and it's CPU-runnable via ONNX — looked like a real candidate for better delivery. Benchmarked the int8 build in this repo before committing to anything: **0.67x real-time** even with `intra_op_num_threads` maxed out, vs. Piper's **18x** on the identical sentence. A model that takes 13.5s to synthesize 9s of audio falls further behind its own playback with every sentence — not viable for a live agent on this hardware, regardless of how much better it sounds. Piper stays; see `tts.py`'s docstring for the numbers.

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
| `POTATO_LLM_FLASH_ATTN` | `1` | flash attention in llama.cpp |
| `POTATO_LLM_SPECULATIVE` | `1` | prompt-lookup speculative decoding |
| `POTATO_LLM_NGRAM_SIZE` / `POTATO_LLM_NUM_PRED_TOKENS` | `2` / `10` | speculative decoding tuning (max n-gram size, predicted tokens) |
| `POTATO_WHISPER_MODEL` | `base.en` | faster-whisper model size |
| `POTATO_AGENT_MAX_STEPS` | `6` | tool-call steps before giving up |
| `POTATO_ALLOW_SHELL` | `0` | enable the shell tool |
| `POTATO_VAD_AGGRESSIVENESS` | `2` | 0-3, higher = more strict about filtering non-speech |
| `POTATO_PUSH_TO_TALK` | `0` | fall back to manual press-Enter instead of continuous listening |
| `POTATO_BARGE_IN` | `1` | interrupt the agent's own TTS by talking over it |
| `POTATO_VAD_BARGE_IN_FRAMES` | `10` (~300ms) | stricter speech-onset threshold used only while the agent is speaking |
| `POTATO_FORCE_IPV4` | `0` | force IPv4-only DNS resolution, for networks where IPv6 routes to model-download hosts hang instead of failing fast |

Swap the LLM for anything else llama.cpp supports (bigger model if you have more RAM, smaller/faster if you don't) — it's one GGUF file and one path.

## Project status

Early, but the live voice loop has now actually been run against a real mic and real speech, not just unit-tested — plain Q&A, calculator/file-op tool calls, barge-in, and streamed speech all verified end-to-end on real hardware. If you hit false starts or missed utterances on your own mic/room, tune `POTATO_VAD_AGGRESSIVENESS` first and open an issue with what didn't work.

Also in: barge-in, speculative decoding + flash attention, sentence-streamed TTS with punctuation-driven pauses, and a startup warmup that cuts first-turn latency from ~9-11s to ~2-3s.

Next up: multi-turn conversation memory (each turn currently only shares the system prompt, not prior turns — a real follow-up question won't have context from what you asked before), and a proper eval set for small-model tool-calling reliability. Issues and PRs welcome — the whole point of this repo is that people with modest hardware can actually run and improve it, not just read about it.

## License

MIT
