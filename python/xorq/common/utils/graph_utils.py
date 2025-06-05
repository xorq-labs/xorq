from typing import Any, Iterable, Optional, Tuple

import xorq.expr.relations as rel
import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.operations as ops
from xorq.vendor.ibis.expr.operations.core import Node


def _filter_none(values: Iterable[Optional[Node]]) -> Tuple[Node, ...]:
    return tuple(v for v in values if v is not None)


def to_node(maybe_expr: Any) -> Node:
    op_fn = getattr(maybe_expr, "op", None)
    if op_fn is None:
        return maybe_expr
    maybe_expr = op_fn()
    return maybe_expr


def children_of(node: Node) -> Tuple[Node, ...]:
    match node:
        case ops.Field():
            rel_node = node.rel
            if rel_node is None:
                return ()
            return _filter_none((to_node(rel_node),))

        case rel.RemoteTable():
            return _filter_none((to_node(node.remote_expr),))

        case rel.CachedNode():
            return (to_node(node.parent),)

        case rel.FlightExpr():
            return (to_node(node.input_expr),)

        case rel.FlightUDXF():
            return (to_node(node.input_expr),)

        case udf.ExprScalarUDF():
            exprs = node.computed_kwargs_expr
            if exprs is not None:
                single = to_node(exprs)
                return (single,) if single else ()
            return ()

        case rel.Read():
            return ()

        case _:
            raw_children = getattr(node, "__children__", ())
            return _filter_none(map(to_node, raw_children))


def walk_nodes(node_types, expr):
    visited = set()
    to_visit = [to_node(expr)]
    result = []

    while to_visit:
        node = to_visit.pop()
        node_id = id(node)

        if node_id in visited:
            continue
        visited.add(node_id)

        if isinstance(node, node_types):
            result.append(node)

        for child in children_of(node):
            if id(child) not in visited:
                to_visit.append(child)

    return tuple(result)


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
