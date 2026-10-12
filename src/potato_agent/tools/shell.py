"""Shell command execution — disabled by default, confirmed interactively when enabled.

This is the most dangerous tool in the registry: it lets the LLM run arbitrary
shell commands. It is OFF unless POTATO_ALLOW_SHELL=1 is set, and even then it
asks for a y/n confirmation before every command so a human stays in the loop.
"""
import subprocess

from .. import config

NAME = "run_shell_command"
DESCRIPTION = 'Run a shell command in the workspace directory. Args: {"command": "ls -la"}'

TIMEOUT_SECONDS = 30


def run(args: dict) -> str:
    command = args.get("command", "")
    if not command:
        return "Error: 'command' is required."

    if not config.SHELL_TOOL_ENABLED:
        return (
            "Shell tool is disabled. Set POTATO_ALLOW_SHELL=1 to enable it "
            "(commands still require interactive confirmation)."
        )

    print(f"\n[potato-agent] Agent wants to run: {command}")
    answer = input("Allow? [y/N] ").strip().lower()
    if answer != "y":
        return "Command rejected by user."

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(config.AGENT_WORKSPACE),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        output = result.stdout.strip()
        if result.stderr.strip():
            output += f"\n[stderr]\n{result.stderr.strip()}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {TIMEOUT_SECONDS}s."
    except Exception as e:
        return f"Error running command: {e}"
