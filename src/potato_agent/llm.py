"""Thin wrapper around llama.cpp for CPU-only inference."""
from __future__ import annotations

from llama_cpp import Llama

from . import config


class LocalLLM:
    def __init__(
        self,
        model_path=None,
        n_ctx: int | None = None,
        n_threads: int | None = None,
    ):
        model_path = str(model_path or config.LLM_GGUF_PATH)
        if not config.LLM_GGUF_PATH.exists() and model_path == str(config.LLM_GGUF_PATH):
            raise FileNotFoundError(
                f"LLM model not found at {model_path}. Run scripts/setup_models.sh first."
            )
        self._llama = Llama(
            model_path=model_path,
            n_ctx=n_ctx or config.LLM_CTX_SIZE,
            n_threads=n_threads or config.LLM_THREADS,
            verbose=False,
        )

    def chat(self, messages: list[dict], max_tokens: int | None = None, stop: list[str] | None = None) -> str:
        """One-shot chat completion. messages is the usual [{role, content}, ...] list."""
        result = self._llama.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens or config.LLM_MAX_TOKENS,
            temperature=0.4,
            stop=stop,
        )
        return result["choices"][0]["message"]["content"]

    def chat_stream(self, messages: list[dict], max_tokens: int | None = None, stop: list[str] | None = None):
        """Yields text chunks as they're generated."""
        stream = self._llama.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens or config.LLM_MAX_TOKENS,
            temperature=0.4,
            stop=stop,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta and delta["content"]:
                yield delta["content"]
