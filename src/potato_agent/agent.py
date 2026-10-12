"""A small, framework-free ReAct loop: LLM <-> tools, via a JSON protocol."""
from __future__ import annotations

import json
import re

from . import config
from .llm import LocalLLM
from .tools import call_tool, describe_tools

SYSTEM_PROMPT = """You are Potato Agent, a helpful assistant running entirely on CPU.

You can use tools to help answer the user. Available tools:
{tools}

Respond with ONE JSON object per turn, nothing else before or after it. Two shapes:

To use a tool:
{{"thought": "why you're doing this", "tool": "<tool_name>", "args": {{...}}}}

To give your final answer:
{{"thought": "brief reasoning", "final_answer": "the answer for the user"}}

Keep final_answer short and conversational — it will be spoken aloud. Only use a
tool when it's actually needed; for simple questions, answer directly."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class Agent:
    def __init__(self, llm: LocalLLM | None = None, max_steps: int | None = None):
        self.llm = llm or LocalLLM()
        self.max_steps = max_steps or config.AGENT_MAX_STEPS
        self.system_prompt = SYSTEM_PROMPT.format(tools=describe_tools())

    def run(self, user_input: str, verbose: bool = True) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]

        for step in range(self.max_steps):
            reply = self.llm.chat(messages)
            parsed = _extract_json(reply)

            if parsed is None:
                # Model didn't follow the protocol — treat raw text as the answer.
                return reply.strip()

            if "final_answer" in parsed:
                return str(parsed["final_answer"])

            tool_name = parsed.get("tool")
            tool_args = parsed.get("args", {})
            if not tool_name:
                return str(parsed.get("thought", reply)).strip()

            if verbose:
                print(f"[step {step + 1}] tool={tool_name} args={tool_args}")

            observation = call_tool(tool_name, tool_args)
            if verbose:
                print(f"[step {step + 1}] observation: {observation[:300]}")

            messages.append({"role": "assistant", "content": json.dumps(parsed)})
            messages.append({"role": "user", "content": f"Tool result: {observation}"})

        return "I couldn't finish within the step limit — try rephrasing or simplifying the request."
