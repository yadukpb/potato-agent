"""File read/write/list, sandboxed to the agent workspace directory."""
from pathlib import Path

from .. import config

NAME = "file_op"
DESCRIPTION = "Read, write, or list files inside the sandboxed workspace."
PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["read", "write", "list"]},
        "path": {"type": "string", "description": "Path relative to the workspace"},
        "content": {"type": "string", "description": "Content to write (only for the write action)"},
    },
    "required": ["action"],
}


def _resolve(path_str: str) -> Path:
    base = config.AGENT_WORKSPACE.resolve()
    target = (base / path_str).resolve()
    if base != target and base not in target.parents:
        raise ValueError(f"Path '{path_str}' escapes the workspace sandbox.")
    return target


def run(args: dict) -> str:
    action = args.get("action", "")
    path_str = args.get("path", "")
    try:
        if action == "list":
            base = config.AGENT_WORKSPACE if not path_str else _resolve(path_str)
            if not base.exists():
                return f"'{path_str or '.'}' does not exist."
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in base.iterdir())
            return "\n".join(entries) or "(empty)"

        if not path_str:
            return "Error: 'path' is required."
        target = _resolve(path_str)

        if action == "read":
            if not target.exists():
                return f"'{path_str}' does not exist."
            return target.read_text(errors="replace")[:8000]

        if action == "write":
            content = args.get("content", "")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            return f"Wrote {len(content)} chars to '{path_str}'."

        return f"Unknown action '{action}'. Use read, write, or list."
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error with file_op: {e}"
