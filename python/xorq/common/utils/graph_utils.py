import xorq.expr.relations as rel


def walk_nodes(node_types, expr):
    def process_node(op):
        match op:
            case rel.RemoteTable():
                yield op
                yield from walk_nodes(
                    node_types,
                    op.remote_expr,
                )
            case rel.CachedNode():
                yield op
                yield from walk_nodes(
                    node_types,
                    op.parent,
                )
            case rel.FlightExpr():
                yield op
                yield from walk_nodes(node_types, op.input_expr)

            case rel.FlightUDXF():
                yield op
                yield from walk_nodes(node_types, op.input_expr)

            case _:
                yield from op.find(node_types)

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
    return inner(set(rest), set())


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
