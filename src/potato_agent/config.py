import os
from pathlib import Path


def _env_int(name: str, default: str, *, min_value: int | None = None, max_value: int | None = None) -> int:
    """Reads an integer env var, failing fast and clearly at import time
    instead of letting a bad value (a typo, an out-of-range number) surface
    later as a confusing error deep inside webrtcvad/llama.cpp/etc."""
    raw = os.environ.get(name, default)
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a valid integer.") from None
    if min_value is not None and value < min_value:
        raise ValueError(f"{name}={value} must be >= {min_value}.")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name}={value} must be <= {max_value}.")
    return value


def _env_bool(name: str, default: str) -> bool:
    raw = os.environ.get(name, default)
    if raw not in ("0", "1"):
        raise ValueError(f"{name}={raw!r} must be '0' or '1'.")
    return raw == "1"


ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("POTATO_MODELS_DIR", ROOT / "models"))

LLM_GGUF_PATH = Path(os.environ.get("POTATO_LLM_PATH", MODELS_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf"))
LLM_CTX_SIZE = _env_int("POTATO_LLM_CTX", "4096", min_value=1)
LLM_THREADS = _env_int("POTATO_LLM_THREADS", str(os.cpu_count() or 4), min_value=1)
LLM_MAX_TOKENS = _env_int("POTATO_LLM_MAX_TOKENS", "512", min_value=1)
LLM_FLASH_ATTN = _env_bool("POTATO_LLM_FLASH_ATTN", "1")
LLM_SPECULATIVE = _env_bool("POTATO_LLM_SPECULATIVE", "1")
LLM_NGRAM_SIZE = _env_int("POTATO_LLM_NGRAM_SIZE", "2", min_value=1)
LLM_NUM_PRED_TOKENS = _env_int("POTATO_LLM_NUM_PRED_TOKENS", "10", min_value=1)

WHISPER_MODEL_SIZE = os.environ.get("POTATO_WHISPER_MODEL", "base.en")
WHISPER_COMPUTE_TYPE = os.environ.get("POTATO_WHISPER_COMPUTE", "int8")

PIPER_VOICE_PATH = Path(os.environ.get("POTATO_PIPER_VOICE", MODELS_DIR / "en_US-lessac-medium.onnx"))

AGENT_MAX_STEPS = _env_int("POTATO_AGENT_MAX_STEPS", "6", min_value=1)
# How many (user, assistant) turn-pairs of conversation history to keep.
# Bounded because the context window (LLM_CTX_SIZE) is finite and small on
# CPU-friendly models/configs -- unbounded history would eventually crowd
# out the system prompt and tool docs. Tool-call scratch steps within a
# single turn are never stored, only the final exchange (see agent.py).
AGENT_HISTORY_TURNS = _env_int("POTATO_HISTORY_TURNS", "6", min_value=0)
AGENT_WORKSPACE = Path(os.environ.get("POTATO_WORKSPACE", ROOT / "workspace"))
SHELL_TOOL_ENABLED = _env_bool("POTATO_ALLOW_SHELL", "0")

# 0-3, higher = more aggressive about filtering out non-speech (webrtcvad).
# Raise this if background noise is triggering false starts.
VAD_AGGRESSIVENESS = _env_int("POTATO_VAD_AGGRESSIVENESS", "2", min_value=0, max_value=3)
# Fall back to manual push-to-talk (old behavior) if VAD doesn't suit your mic/room.
PUSH_TO_TALK = _env_bool("POTATO_PUSH_TO_TALK", "0")

# Let the user interrupt the agent's own TTS playback by talking over it.
# Without echo cancellation, the mic can hear the agent's own voice bleeding
# in from the speakers, so barge-in uses a stricter (slower-to-trigger)
# speech-onset threshold than normal turn-taking to resist false triggers.
# Most reliable with headphones. Disable if it misfires on your setup.
BARGE_IN_ENABLED = _env_bool("POTATO_BARGE_IN", "1")
VAD_BARGE_IN_FRAMES = _env_int("POTATO_VAD_BARGE_IN_FRAMES", "10", min_value=1)  # ~300ms

AGENT_WORKSPACE.mkdir(parents=True, exist_ok=True)

# Only affects internal diagnostics routed through logging (see
# logging_setup.py) -- the conversational output itself (what the agent
# heard/said) is always printed, since that's the UI, not a log.
_VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LOG_LEVEL = os.environ.get("POTATO_LOG_LEVEL", "INFO").upper()
if LOG_LEVEL not in _VALID_LOG_LEVELS:
    raise ValueError(f"POTATO_LOG_LEVEL={LOG_LEVEL!r} must be one of {', '.join(_VALID_LOG_LEVELS)}.")
