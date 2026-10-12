#!/usr/bin/env bash
# Downloads the models potato-agent needs. ~2.5GB total.
# faster-whisper's model is NOT downloaded here — it fetches itself on first run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${POTATO_MODELS_DIR:-$ROOT/models}"
mkdir -p "$MODELS_DIR"

LLM_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
LLM_PATH="$MODELS_DIR/qwen2.5-3b-instruct-q4_k_m.gguf"

PIPER_ONNX_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
PIPER_JSON_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
PIPER_ONNX_PATH="$MODELS_DIR/en_US-lessac-medium.onnx"
PIPER_JSON_PATH="$MODELS_DIR/en_US-lessac-medium.onnx.json"

download() {
  local url="$1" dest="$2"
  if [ -f "$dest" ]; then
    echo "already have $(basename "$dest"), skipping"
    return
  fi
  echo "downloading $(basename "$dest")..."
  curl -L --fail -o "$dest" "$url"
}

echo "== LLM (Qwen2.5-3B-Instruct, Q4_K_M, ~2GB) =="
download "$LLM_URL" "$LLM_PATH"

echo "== Piper TTS voice (en_US-lessac-medium, ~60MB) =="
download "$PIPER_ONNX_URL" "$PIPER_ONNX_PATH"
download "$PIPER_JSON_URL" "$PIPER_JSON_PATH"

echo ""
echo "Done. faster-whisper will download its STT model (base.en, ~150MB) automatically"
echo "the first time you run the agent."
