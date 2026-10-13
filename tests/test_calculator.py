"""calculator is the one tool that gets eval-like power without eval() --
these tests are really about proving the AST whitelist holds."""
from potato_agent.tools import calculator


def _run(expr):
    return calculator.run({"expression": expr})


def test_basic_arithmetic():
    assert _run("2 + 3") == "5"
    assert _run("2 * (3 + 4)") == "14"
    assert _run("10 / 4") == "2.5"
    assert _run("10 // 4") == "2"
    assert _run("10 % 4") == "2"
    assert _run("2 ** 10") == "1024"


def test_unary_operators():
    assert _run("-5 + 3") == "-2"
    assert _run("+5") == "5"


def test_missing_expression_errors_without_crashing():
    assert _run("").startswith("Error")


def test_division_by_zero_is_reported_not_raised():
    result = _run("1 / 0")
    assert "Error" in result


def test_rejects_name_lookups():
    # No ast.Name handling in _eval_node -- bare identifiers must fail closed.
    assert "Error" in _run("__import__")


def test_rejects_function_calls():
    assert "Error" in _run("__import__('os').system('echo pwned')")


def test_rejects_attribute_access():
    assert "Error" in _run("(1).__class__")


def test_rejects_string_literals():
    # ast.Constant branch only accepts int/float, so a string constant
    # should fall through to the "Unsupported expression" error.
    assert "Error" in _run("'a' + 'b'")
