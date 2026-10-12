"""Thin wrapper around llama.cpp for CPU-only inference."""
from __future__ import annotations

import json

from llama_cpp import Llama, LlamaGrammar
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

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

        # Prompt-lookup speculative decoding: speculates upcoming tokens by
        # matching n-grams already seen in the context, so it needs no
        # second draft model/download. Works especially well for us because
        # the agent's grammar-constrained JSON output and growing
        # conversation history both have a lot of repeated structure for it
        # to match against. Falls back to normal decoding with no quality
        # loss when nothing matches.
        draft_model = (
            LlamaPromptLookupDecoding(
                max_ngram_size=config.LLM_NGRAM_SIZE,
                num_pred_tokens=config.LLM_NUM_PRED_TOKENS,
            )
            if config.LLM_SPECULATIVE
            else None
        )

        self._llama = Llama(
            model_path=model_path,
            n_ctx=n_ctx or config.LLM_CTX_SIZE,
            n_threads=n_threads or config.LLM_THREADS,
            flash_attn=config.LLM_FLASH_ATTN,
            draft_model=draft_model,
            # llama-cpp-python sizes its internal scores buffer off this
            # parameter directly, but sizes the *decode-time* writes off a
            # separate internal flag that it force-sets to True whenever a
            # draft_model is present. If we don't also pass True here, that
            # buffer comes out n_batch-sized instead of n_ctx-sized and
            # eval() overflows it past the first ~512 tokens. Upstream bug,
            # worked around by keeping the two in sync ourselves.
            logits_all=config.LLM_SPECULATIVE,
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

    def chat_json(self, messages: list[dict], json_schema: dict, max_tokens: int | None = None) -> str:
        """Chat completion grammar-constrained to match the given JSON Schema.

        Unlike regex-extracting JSON from free-form text, this guarantees the
        returned string parses as valid JSON matching the schema's shape —
        llama.cpp compiles the schema into a GBNF grammar and masks logits
        during decoding so invalid tokens can't be generated at all.
        """
        grammar = LlamaGrammar.from_json_schema(json.dumps(json_schema))
        result = self._llama.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens or config.LLM_MAX_TOKENS,
            temperature=0.2,
            grammar=grammar,
        )
        return result["choices"][0]["message"]["content"]

    def chat_json_stream(self, messages: list[dict], json_schema: dict, max_tokens: int | None = None):
        """Streaming variant of chat_json: yields text deltas of the
        grammar-constrained JSON as they're generated, instead of waiting
        for the whole response. agent.py uses this to start speaking a
        final_answer's text before the model finishes generating the rest
        of the JSON object.
        """
        grammar = LlamaGrammar.from_json_schema(json.dumps(json_schema))
        stream = self._llama.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens or config.LLM_MAX_TOKENS,
            temperature=0.2,
            grammar=grammar,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta and delta["content"]:
                yield delta["content"]

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
