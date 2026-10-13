"""file_op is sandboxed to AGENT_WORKSPACE -- the important property to
pin down is that no path argument can escape that directory."""
import pytest

from potato_agent import config
from potato_agent.tools import files


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AGENT_WORKSPACE", tmp_path)
    yield tmp_path


def test_write_then_read_round_trip():
    write_result = files.run({"action": "write", "path": "notes.txt", "content": "hello"})
    assert "Wrote 5 chars" in write_result
    assert files.run({"action": "read", "path": "notes.txt"}) == "hello"


def test_read_missing_file_reports_cleanly():
    result = files.run({"action": "read", "path": "ghost.txt"})
    assert "does not exist" in result


def test_list_empty_workspace():
    assert files.run({"action": "list"}) == "(empty)"


def test_list_reflects_written_files():
    files.run({"action": "write", "path": "a.txt", "content": "x"})
    assert files.run({"action": "list"}) == "a.txt"


def test_write_creates_nested_directories():
    files.run({"action": "write", "path": "sub/dir/deep.txt", "content": "nested"})
    assert files.run({"action": "read", "path": "sub/dir/deep.txt"}) == "nested"


def test_rejects_parent_traversal_escape(workspace):
    result = files.run({"action": "write", "path": "../escape.txt", "content": "x"})
    assert "escapes the workspace sandbox" in result
    assert not (workspace.parent / "escape.txt").exists()


def test_rejects_absolute_path_escape():
    result = files.run({"action": "read", "path": "/etc/passwd"})
    assert "escapes the workspace sandbox" in result


def test_write_missing_path_errors():
    assert files.run({"action": "write", "content": "x"}) == "Error: 'path' is required."


def test_unknown_action_reports_cleanly():
    result = files.run({"action": "delete", "path": "a.txt"})
    assert "Unknown action" in result
