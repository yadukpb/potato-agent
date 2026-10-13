"""Env vars are a system boundary (user-supplied, easy to typo) -- config.py
should fail fast and clearly at import time rather than letting a bad value
surface later as a confusing error deep inside webrtcvad/llama.cpp/etc.

config.py itself runs its validation at import time, so these tests exercise
the underlying helpers directly rather than re-importing the module with
different env vars each time (which module caching makes awkward, and which
would also re-run AGENT_WORKSPACE.mkdir() repeatedly for no reason)."""
import pytest

from potato_agent.config import _env_bool, _env_int


def test_env_int_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv("POTATO_TEST_INT", raising=False)
    assert _env_int("POTATO_TEST_INT", "42") == 42


def test_env_int_parses_set_value(monkeypatch):
    monkeypatch.setenv("POTATO_TEST_INT", "7")
    assert _env_int("POTATO_TEST_INT", "42") == 7


def test_env_int_rejects_non_integer(monkeypatch):
    monkeypatch.setenv("POTATO_TEST_INT", "not-a-number")
    with pytest.raises(ValueError, match="POTATO_TEST_INT"):
        _env_int("POTATO_TEST_INT", "42")


def test_env_int_enforces_min_value(monkeypatch):
    monkeypatch.setenv("POTATO_TEST_INT", "0")
    with pytest.raises(ValueError, match=">= 1"):
        _env_int("POTATO_TEST_INT", "42", min_value=1)


def test_env_int_enforces_max_value(monkeypatch):
    monkeypatch.setenv("POTATO_TEST_INT", "4")
    with pytest.raises(ValueError, match="<= 3"):
        _env_int("POTATO_TEST_INT", "2", min_value=0, max_value=3)


def test_env_int_accepts_boundary_values(monkeypatch):
    monkeypatch.setenv("POTATO_TEST_INT", "3")
    assert _env_int("POTATO_TEST_INT", "2", min_value=0, max_value=3) == 3


def test_env_bool_parses_0_and_1(monkeypatch):
    monkeypatch.setenv("POTATO_TEST_BOOL", "1")
    assert _env_bool("POTATO_TEST_BOOL", "0") is True
    monkeypatch.setenv("POTATO_TEST_BOOL", "0")
    assert _env_bool("POTATO_TEST_BOOL", "1") is False


def test_env_bool_rejects_other_values(monkeypatch):
    monkeypatch.setenv("POTATO_TEST_BOOL", "true")
    with pytest.raises(ValueError, match="POTATO_TEST_BOOL"):
        _env_bool("POTATO_TEST_BOOL", "0")


def test_env_bool_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv("POTATO_TEST_BOOL", raising=False)
    assert _env_bool("POTATO_TEST_BOOL", "1") is True
