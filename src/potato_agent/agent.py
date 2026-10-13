"""A small, framework-free ReAct loop: LLM <-> tools, via grammar-constrained JSON.

Why not llama-cpp-python's built-in "chatml-function-calling" chat format?
It has a real bug: when the model responds with free text (no tool call), the
library constrains that generation with the WRONG grammar — one meant for
picking between function names or stopping — which corrupts every plain
answer into garbage like "functions.calculator:" or an empty string. It also
has no template branch for tool-result messages, so those get silently
dropped from the prompt. Both are upstream issues, not something fixable
from a caller.

Instead, this loop builds its own small JSON Schema (tool call vs. final
answer) and asks llama.cpp to compile it into a GBNF grammar via
LlamaGrammar.from_json_schema — see llm.chat_json(). That guarantees valid,
on-schema JSON every turn without depending on any chat-format's tool-calling
scaffolding.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from . import config
from .llm import LocalLLM
from .tools import TOOLS_SCHEMA, call_tool

_TOOL_NAMES = [t["function"]["name"] for t in TOOLS_SCHEMA]

_FINAL_ANSWER_KEY_RE = re.compile(r'"final_answer"\s*:\s*"')
# Deliberately duplicated from tts.split_sentences's boundary rule rather
# than imported: agent.py stays framework-free (no TTS/audio dependency,
# so cli.py's text-only REPL doesn't need to import sounddevice etc.), and
# both sides just need to agree on what counts as "a complete sentence".
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def _decode_partial_json_string(raw: str, start: int) -> str:
    """Decode as much as possible of a JSON string value starting right
    after its opening quote, stopping at the first unescaped closing quote
    (or at the end of `raw`, if the value isn't finished yet).

    If `raw` ends mid-escape-sequence (a lone trailing backslash, or a
    \\uXXXX cut short), that trailing partial escape is held back rather
    than guessed at -- it decodes correctly once more characters arrive.
    """
    out = []
    i, n = start, len(raw)
    while i < n:
        c = raw[i]
        if c == '"':
            break
        if c == "\\":
            if i + 1 >= n:
                break
            nxt = raw[i + 1]
            if nxt == "u":
                if i + 6 > n:
                    break
                out.append(chr(int(raw[i + 2 : i + 6], 16)))
                i += 6
                continue
            out.append({"n": "\n", "t": "\t", "r": "\r"}.get(nxt, nxt))
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


_RESET_PREFIXES = ("hey ", "okay ", "ok ", "please ", "can you ", "could you ", "would you ")
_RESET_PHRASES = frozenset(
    {
        "reset",
        "reset conversation",
        "reset the conversation",
        "clear history",
        "clear your history",
        "clear the history",
        "clear your memory",
        "clear memory",
        "forget everything",
        "forget our conversation",
        "forget what we talked about",
        "start over",
        "start a new conversation",
        "new conversation",
    }
)


def is_reset_command(text: str) -> bool:
    """True if `text` is (just) a request to clear conversation history.

    Text mode has an exact "/reset" command, but voice mode has no typed
    input to match against -- this is its equivalent, matched against a
    fixed set of natural-language phrases after stripping punctuation and
    a few common leading politeness words ("please reset", "can you reset
    the conversation", ...).

    Deliberately an exact match against the whole (stripped) utterance, not
    a substring/fuzzy one: a false positive here silently wipes context,
    which is worse than making a user rephrase once. "reset" on its own
    matches, but "can you reset my password" does not.
    """
    normalized = re.sub(r"[^a-z0-9\s]", "", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    stripped = True
    while stripped:
        stripped = False
        for prefix in _RESET_PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                stripped = True
    return normalized in _RESET_PHRASES


def _tool_docs() -> str:
    lines = []
    for t in TOOLS_SCHEMA:
        fn = t["function"]
        lines.append(f"- {fn['name']}: {fn['description']} Parameters: {json.dumps(fn['parameters'])}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are Potato Agent, a helpful assistant running entirely on CPU.

Available tools:
{_tool_docs()}

Respond with ONE JSON object per turn. Only use a tool when it's genuinely
needed — for simple questions, answer directly. As soon as you have enough
information to answer, respond immediately with final_answer — do not take
any further actions beyond what the user actually asked for. Keep
final_answer short and conversational, since it may be spoken aloud.

Any request to read, write, or list a file is a file_op call, every single
time -- never respond with final_answer claiming a file was read, written,
or listed unless you actually called file_op for it first in this
conversation.

Examples:
User: "What's 12 times 4?"
{{"thought": "Need the calculator.", "tool": "calculator", "args": {{"expression": "12 * 4"}}}}

User: "Save the word 'hello' to notes.txt"
{{"thought": "Writing the requested file.", "tool": "file_op", "args": {{"action": "write", "path": "notes.txt", "content": "hello"}}}}

User: "What files are in the workspace?"
{{"thought": "Need to actually list them, not guess.", "tool": "file_op", "args": {{"action": "list", "path": ""}}}}

User: "What's the tallest mountain?"
{{"thought": "This is general knowledge, no tool needed.", "final_answer": "Mount Everest, at about 8,849 meters."}}"""

RESPONSE_SCHEMA = {
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "tool": {"type": "string", "enum": _TOOL_NAMES},
                "args": {"type": "object"},
            },
            "required": ["tool", "args"],
        },
        {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "final_answer": {"type": "string"},
            },
            "required": ["final_answer"],
        },
    ]
}


class Agent:
    def __init__(
        self,
        llm: LocalLLM | None = None,
        max_steps: int | None = None,
        history_turns: int | None = None,
    ):
        self.llm = llm or LocalLLM()
        self.max_steps = max_steps or config.AGENT_MAX_STEPS
        self.history_turns = config.AGENT_HISTORY_TURNS if history_turns is None else history_turns
        # (user, assistant) messages from completed turns only -- a turn's
        # internal tool-call/tool-result scratch never ends up here, just
        # like it's never spoken (see run()'s on_sentence docstring).
        self.history: list[dict] = []

    def clear_history(self) -> None:
        """Forget all prior turns. The next run() starts a fresh conversation
        (still sharing SYSTEM_PROMPT, so warm_up()'s KV-cache benefit holds)."""
        self.history = []

    def _remember(self, user_input: str, answer: str) -> None:
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": answer})
        max_messages = self.history_turns * 2
        if len(self.history) > max_messages:
            # Drop the oldest pair(s). Note this means the *next* call's
            # prompt no longer shares a full prefix with this one -- the
            # llama-cpp-python KV-cache prefix match (see warm_up()) falls
            # back to just SYSTEM_PROMPT for that one turn, not the full
            # history. A real cost, but bounded context beats an ever-
            # growing prompt that eventually crowds out the tool docs.
            self.history = self.history[-max_messages:]

    def warm_up(self) -> None:
        """Pre-fills the KV cache with SYSTEM_PROMPT and pays llama.cpp's
        one-time backend/thread-pool init cost, both up front rather than
        on the user's first real turn.

        This only actually helps because llama-cpp-python caches the
        longest matching prefix between consecutive prompts and skips
        re-evaluating it (see Llama.generate's prefix-match handling). Every
        call to run() starts its `messages` with this same SYSTEM_PROMPT, so
        warming up with anything else -- or with no system prompt at all --
        doesn't share a prefix with the real first turn and buys nothing;
        measured at ~8-11s either way. Warming up with the real prefix in
        place drops the real first turn from ~9-11s to ~3s.
        """
        self.llm.chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "hi"}],
            max_tokens=1,
        )

    def run(
        self,
        user_input: str,
        verbose: bool = True,
        on_sentence: Callable[[str], None] | None = None,
        on_step: Callable[[dict], None] | None = None,
    ) -> str:
        """Runs the ReAct loop to completion and returns the final answer text.

        on_sentence, if given, is called with each sentence of the final
        answer as soon as it's fully generated -- before the model has
        finished generating the rest of the response -- so a caller doing
        TTS can start speaking immediately instead of waiting for the whole
        reply. It only ever fires for the turn that produces final_answer;
        tool-call steps are internal and never spoken.

        on_step, if given, is called once per loop iteration with that
        step's parsed JSON response ({"tool": ..., "args": ...} or
        {"final_answer": ...}) right after it's decoded, before the tool
        actually runs. Unlike on_sentence it fires for every step, not just
        the final one -- scripts/eval_tools.py uses it to check which tool
        (if any) the model picked on the *first* step, not just whether it
        eventually landed on the right final answer.

        Prior turns are included via self.history, so a follow-up question
        ("what about in Celsius?") has context from what was actually asked
        and answered before -- not from this turn's own tool-call scratch,
        which is discarded once a final_answer is reached. See clear_history()
        to start a fresh conversation.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.history,
            {"role": "user", "content": user_input},
        ]

        for step in range(self.max_steps):
            raw_parts: list[str] = []
            is_final_answer: bool | None = None
            value_start: int | None = None
            decoded_len = 0
            pending = ""

            for delta in self.llm.chat_json_stream(messages, RESPONSE_SCHEMA):
                raw_parts.append(delta)

                if on_sentence is None:
                    continue
                raw = "".join(raw_parts)

                if is_final_answer is None:
                    if '"final_answer"' in raw:
                        is_final_answer = True
                    elif '"tool"' in raw:
                        is_final_answer = False
                if not is_final_answer:
                    continue

                if value_start is None:
                    match = _FINAL_ANSWER_KEY_RE.search(raw)
                    if match is None:
                        continue
                    value_start = match.end()

                decoded = _decode_partial_json_string(raw, value_start)
                pending += decoded[decoded_len:]
                decoded_len = len(decoded)

                *complete, pending = _SENTENCE_BOUNDARY_RE.split(pending)
                for sentence in complete:
                    sentence = sentence.strip()
                    if sentence:
                        on_sentence(sentence)

            raw = "".join(raw_parts)
            if is_final_answer and on_sentence is not None and pending.strip():
                on_sentence(pending.strip())

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                # Shouldn't happen — the grammar guarantees valid JSON — but
                # don't crash the loop if it somehow does.
                answer = raw.strip()
                self._remember(user_input, answer)
                return answer

            if on_step is not None:
                on_step(parsed)

            if "final_answer" in parsed:
                answer = str(parsed["final_answer"]).strip()
                self._remember(user_input, answer)
                return answer

            tool_name = parsed.get("tool")
            tool_args = parsed.get("args", {})

            if verbose:
                print(f"[step {step + 1}] tool={tool_name} args={tool_args}")

            observation = call_tool(tool_name, tool_args)

            if verbose:
                print(f"[step {step + 1}] observation: {observation[:300]}")

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Tool result: {observation}"})

        answer = "I couldn't finish within the step limit — try rephrasing or simplifying the request."
        self._remember(user_input, answer)
        return answer
