"""Safe arithmetic evaluator — no eval(), only a whitelisted AST subset."""
import ast
import operator

NAME = "calculator"
DESCRIPTION = "Evaluate a math expression."
PARAMETERS = {
    "type": "object",
    "properties": {
        "expression": {"type": "string", "description": "A math expression, e.g. '2 * (3 + 4)'"},
    },
    "required": ["expression"],
}

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def run(args: dict) -> str:
    expr = args.get("expression", "")
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"
