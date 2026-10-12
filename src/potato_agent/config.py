import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("POTATO_MODELS_DIR", ROOT / "models"))

LLM_GGUF_PATH = Path(os.environ.get("POTATO_LLM_PATH", MODELS_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf"))
LLM_CTX_SIZE = int(os.environ.get("POTATO_LLM_CTX", "4096"))
LLM_THREADS = int(os.environ.get("POTATO_LLM_THREADS", str(os.cpu_count() or 4)))
LLM_MAX_TOKENS = int(os.environ.get("POTATO_LLM_MAX_TOKENS", "512"))
LLM_FLASH_ATTN = os.environ.get("POTATO_LLM_FLASH_ATTN", "1") == "1"
LLM_SPECULATIVE = os.environ.get("POTATO_LLM_SPECULATIVE", "1") == "1"
LLM_NGRAM_SIZE = int(os.environ.get("POTATO_LLM_NGRAM_SIZE", "2"))
LLM_NUM_PRED_TOKENS = int(os.environ.get("POTATO_LLM_NUM_PRED_TOKENS", "10"))

WHISPER_MODEL_SIZE = os.environ.get("POTATO_WHISPER_MODEL", "base.en")
WHISPER_COMPUTE_TYPE = os.environ.get("POTATO_WHISPER_COMPUTE", "int8")

PIPER_VOICE_PATH = Path(os.environ.get("POTATO_PIPER_VOICE", MODELS_DIR / "en_US-lessac-medium.onnx"))

AGENT_MAX_STEPS = int(os.environ.get("POTATO_AGENT_MAX_STEPS", "6"))
AGENT_WORKSPACE = Path(os.environ.get("POTATO_WORKSPACE", ROOT / "workspace"))
SHELL_TOOL_ENABLED = os.environ.get("POTATO_ALLOW_SHELL", "0") == "1"

# 0-3, higher = more aggressive about filtering out non-speech (webrtcvad).
# Raise this if background noise is triggering false starts.
VAD_AGGRESSIVENESS = int(os.environ.get("POTATO_VAD_AGGRESSIVENESS", "2"))
# Fall back to manual push-to-talk (old behavior) if VAD doesn't suit your mic/room.
PUSH_TO_TALK = os.environ.get("POTATO_PUSH_TO_TALK", "0") == "1"

# Let the user interrupt the agent's own TTS playback by talking over it.
# Without echo cancellation, the mic can hear the agent's own voice bleeding
# in from the speakers, so barge-in uses a stricter (slower-to-trigger)
# speech-onset threshold than normal turn-taking to resist false triggers.
# Most reliable with headphones. Disable if it misfires on your setup.
BARGE_IN_ENABLED = os.environ.get("POTATO_BARGE_IN", "1") == "1"
VAD_BARGE_IN_FRAMES = int(os.environ.get("POTATO_VAD_BARGE_IN_FRAMES", "10"))  # ~300ms

AGENT_WORKSPACE.mkdir(parents=True, exist_ok=True)
