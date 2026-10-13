"""run_shell_command is the most dangerous tool in the registry -- these
tests pin down the two safety properties that matter: off by default, and
even when on, nothing executes without an explicit 'y' at the prompt."""
import pytest

from potato_agent import config
from potato_agent.tools import shell


def test_missing_command_errors():
    assert shell.run({}) == "Error: 'command' is required."


def test_disabled_by_default_never_prompts(monkeypatch):
    monkeypatch.setattr(config, "SHELL_TOOL_ENABLED", False)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("should not prompt when disabled"))
    result = shell.run({"command": "echo hi"})
    assert "disabled" in result
    assert "POTATO_ALLOW_SHELL" in result


def test_enabled_but_declined_does_not_execute(monkeypatch):
    monkeypatch.setattr(config, "SHELL_TOOL_ENABLED", True)
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    ran = {"called": False}

    def fake_run(*a, **kw):
        ran["called"] = True

    monkeypatch.setattr(shell.subprocess, "run", fake_run)
    result = shell.run({"command": "rm -rf /"})
    assert result == "Command rejected by user."
    assert ran["called"] is False


def test_enabled_and_confirmed_executes(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SHELL_TOOL_ENABLED", True)
    monkeypatch.setattr(config, "AGENT_WORKSPACE", tmp_path)
    monkeypatch.setattr("builtins.input", lambda *a: "y")
    result = shell.run({"command": "echo hello"})
    assert "hello" in result


def test_timeout_is_reported_not_raised(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SHELL_TOOL_ENABLED", True)
    monkeypatch.setattr(config, "AGENT_WORKSPACE", tmp_path)
    monkeypatch.setattr("builtins.input", lambda *a: "y")
    monkeypatch.setattr(shell, "TIMEOUT_SECONDS", 1)
    result = shell.run({"command": "sleep 5"})
    assert "timed out" in result
