from potato_agent.tools import REGISTRY, TOOLS_SCHEMA, call_tool


def test_registry_contains_expected_tools():
    assert set(REGISTRY) == {"calculator", "web_search", "run_shell_command", "file_op"}


def test_tools_schema_matches_registry():
    schema_names = {t["function"]["name"] for t in TOOLS_SCHEMA}
    assert schema_names == set(REGISTRY)


def test_call_unknown_tool_reports_cleanly_instead_of_raising():
    result = call_tool("delete_everything", {})
    assert "unknown tool" in result
    assert "calculator" in result  # lists available tools


def test_call_tool_dispatches_to_the_right_module():
    assert call_tool("calculator", {"expression": "1 + 1"}) == "2"
