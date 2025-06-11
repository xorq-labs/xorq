import xorq.expr.relations as rel
import xorq.expr.udf as udf


opaque_ops = (
    rel.Read,
    rel.CachedNode,
    rel.RemoteTable,
    rel.FlightUDXF,
    rel.FlightExpr,
    udf.ExprScalarUDF,
)


def walk_nodes(node_types, expr):
    def process_node(op):
        match op:
            case rel.RemoteTable():
                if isinstance(op, node_types):
                    yield op
                yield from walk_nodes(
                    node_types,
                    op.remote_expr,
                )
            case rel.CachedNode():
                if isinstance(op, node_types):
                    yield op
                yield from walk_nodes(
                    node_types,
                    op.parent,
                )
            case rel.FlightExpr():
                if isinstance(op, node_types):
                    yield op
                yield from walk_nodes(node_types, op.input_expr)
            case rel.FlightUDXF():
                if isinstance(op, node_types):
                    yield op
                yield from walk_nodes(node_types, op.input_expr)
            case udf.ExprScalarUDF():
                if isinstance(op, node_types):
                    yield op
                yield from walk_nodes(
                    node_types,
                    op.computed_kwargs_expr,
                )
            case rel.Read():
                if isinstance(op, node_types):
                    yield op
            case _:
                if isinstance(op, opaque_ops):
                    raise ValueError(f"unhandled opaque op {type(op)}")
                yield from op.find(opaque_ops + tuple(node_types))

    def inner(rest, seen):
        if not rest:
            return seen
        op = rest.pop()
        seen.add(op)
        new = process_node(op)
        rest.update(set(new).difference(seen))
        return inner(rest, seen)

    initial_op = expr.op() if hasattr(expr, "op") else expr
    rest = process_node(initial_op)
    nodes = inner(set(rest), set())
    return tuple(node for node in nodes if isinstance(node, node_types))


def replace_nodes(replacer, expr):
    def process_node(op):
        match op:
            case rel.RemoteTable():
                remote_expr = replace_nodes(replacer, op.remote_expr).to_expr()
                op = op.replace(replacer)
                kwargs = dict(zip(op.__argnames__, op.__args__))
                kwargs["remote_expr"] = remote_expr
                return op.__recreate__(kwargs)
            case rel.CachedNode():
                parent = replace_nodes(replacer, op.parent).to_expr()
                op = op.replace(replacer)
                kwargs = dict(zip(op.__argnames__, op.__args__))
                kwargs["parent"] = parent
                return op.__recreate__(kwargs)
            case rel.FlightExpr() | rel.FlightUDXF():
                input_expr = replace_nodes(replacer, op.input_expr).to_expr()
                op = op.replace(replacer)
                kwargs = dict(zip(op.__argnames__, op.__args__))
                kwargs["input_expr"] = input_expr
                return op.__recreate__(kwargs)
            case udf.ExprScalarUDF():
                computed_kwargs_expr = replace_nodes(
                    replacer, op.computed_kwargs_expr
                ).to_expr()
                op = op.replace(replacer)
                kwargs = dict(zip(op.__argnames__, op.__args__))
                kwargs["computed_kwargs_expr"] = computed_kwargs_expr
                return op.__recreate__(kwargs)
            case _:
                if isinstance(op, opaque_ops):
                    raise ValueError(f"unhandled opaque op {type(op)}")
                return op.replace(replacer)

    initial_op = expr.op() if hasattr(expr, "op") else expr
    op = process_node(initial_op)
    return op


def find_all_sources(expr):
    import xorq.vendor.ibis.expr.operations as ops

    node_types = (
        ops.DatabaseTable,
        ops.SQLQueryResult,
        rel.CachedNode,
        rel.Read,
        rel.RemoteTable,
        rel.FlightUDXF,
        rel.FlightExpr,
        # ExprScalarUDF has an expr we need to get to
        # FlightOperator has a dynamically generated connection: it should be passed a Profile instead
    )
    nodes = walk_nodes(node_types, expr)
    sources = tuple(
        source
        for (source, _) in set((node.source, node.source._profile) for node in nodes)
    )
    return sources
