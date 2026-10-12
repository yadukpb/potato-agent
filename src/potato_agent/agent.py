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

from . import config
from .llm import LocalLLM
from .tools import TOOLS_SCHEMA, call_tool

_TOOL_NAMES = [t["function"]["name"] for t in TOOLS_SCHEMA]


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

Examples:
User: "What's 12 times 4?"
{{"thought": "Need the calculator.", "tool": "calculator", "args": {{"expression": "12 * 4"}}}}

User: "Save the word 'hello' to notes.txt"
{{"thought": "Writing the requested file.", "tool": "file_op", "args": {{"action": "write", "path": "notes.txt", "content": "hello"}}}}

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
    def __init__(self, llm: LocalLLM | None = None, max_steps: int | None = None):
        self.llm = llm or LocalLLM()
        self.max_steps = max_steps or config.AGENT_MAX_STEPS

    def run(self, user_input: str, verbose: bool = True) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

        for step in range(self.max_steps):
            raw = self.llm.chat_json(messages, RESPONSE_SCHEMA)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                # Shouldn't happen — the grammar guarantees valid JSON — but
                # don't crash the loop if it somehow does.
                return raw.strip()

            if "final_answer" in parsed:
                return str(parsed["final_answer"]).strip()

            tool_name = parsed.get("tool")
            tool_args = parsed.get("args", {})

            if verbose:
                print(f"[step {step + 1}] tool={tool_name} args={tool_args}")

            observation = call_tool(tool_name, tool_args)

            if verbose:
                print(f"[step {step + 1}] observation: {observation[:300]}")

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Tool result: {observation}"})

        return "I couldn't finish within the step limit — try rephrasing or simplifying the request."
