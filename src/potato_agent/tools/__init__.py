"""Tool registry: name -> (description, run(args: dict) -> str)."""
from . import calculator, files, shell, web_search

_MODULES = [calculator, web_search, shell, files]

REGISTRY = {m.NAME: m for m in _MODULES}


def describe_tools() -> str:
    lines = [f"- {m.NAME}: {m.DESCRIPTION}" for m in _MODULES]
    return "\n".join(lines)


def call_tool(name: str, args: dict) -> str:
    module = REGISTRY.get(name)
    if module is None:
        return f"Error: unknown tool '{name}'. Available: {', '.join(REGISTRY)}"
    return module.run(args)
