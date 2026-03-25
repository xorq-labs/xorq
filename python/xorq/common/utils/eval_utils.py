import ast


_ALLOWED_NODES = frozenset(
    {
        ast.Expression,
        ast.Name,
        ast.Constant,
        ast.Attribute,
        # calls
        ast.Call,
        ast.keyword,
        ast.Starred,
        # operators
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.BoolOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Not,
        ast.Invert,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
        ast.And,
        ast.Or,
        # collections
        ast.List,
        ast.Tuple,
        ast.Dict,
        # context
        ast.Load,
        # subscript
        ast.Subscript,
        ast.Slice,
        # lambda
        ast.Lambda,
        ast.arguments,
        ast.arg,
    }
)


def safe_eval(code, namespace):
    """Evaluate *code* as a Python expression within *namespace*.

    The AST is walked before execution and only a whitelist of node types is
    permitted.  Dunder attribute access (``__foo__``) is rejected so that
    object-introspection escapes (``().__class__.__bases__`` etc.) cannot
    bypass the restricted namespace.
    """
    tree = ast.parse(code, mode="eval")
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            raise ValueError(f"disallowed expression: {type(node).__name__}")
        if (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("__")
            and node.attr.endswith("__")
        ):
            raise ValueError(f"dunder access not allowed: {node.attr}")
    return eval(compile(tree, "<code>", "eval"), namespace)  # noqa: S307
