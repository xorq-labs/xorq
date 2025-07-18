from typing import Any, Tuple

import xorq.expr.relations as rel
import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.operations as ops
from xorq import Expr
from xorq.vendor.ibis.expr.operations.core import Node


opaque_ops = (
    rel.Read,
    rel.CachedNode,
    rel.RemoteTable,
    rel.FlightUDXF,
    rel.FlightExpr,
    udf.ExprScalarUDF,
)


def to_node(maybe_expr: Any) -> Node:
    match maybe_expr:
        case Node():
            return maybe_expr
        case Expr():
            return maybe_expr.op()
        case _:
            raise ValueError(f"Don't know how to handle type {type(maybe_expr)}")


def gen_children_of(node: Node) -> Tuple[Node, ...]:
    match node:
        case ops.Field():
            rel_node = node.rel
            gen = () if rel_node is None else (to_node(rel_node),)

        case rel.RemoteTable():
            gen = (to_node(node.remote_expr),)

        case rel.CachedNode():
            gen = (to_node(node.parent),)

        case rel.FlightExpr() | rel.FlightUDXF():
            gen = (to_node(node.input_expr),)

        case udf.ExprScalarUDF():
            gen = (to_node(node.computed_kwargs_expr),)

        case rel.Read():
            gen = ()

        case _:
            raw_children = getattr(node, "__children__", ())
            # return _filter_none(map(to_node, raw_children))
            gen = map(to_node, raw_children)
    yield from filter(None, gen)


def walk_nodes(node_types, expr):
    visited = set()
    to_visit = [to_node(expr)]
    result = ()

    while to_visit:
        node = to_visit.pop()
        if node in visited:
            continue
        visited.add(node)
        if isinstance(node, node_types):
            result += (node,)

        to_visit += set(gen_children_of(node)).difference(visited)

    return result


def replace_nodes(replacer, expr):
    def do_recreate(op, _kwargs, **kwargs):
        kwargs = dict(zip(op.__argnames__, op.__args__)) | (_kwargs or {}) | kwargs
        return op.__recreate__(kwargs)

    def process_node(op, _kwargs):
        op = replacer(op, _kwargs)
        match op:
            case rel.RemoteTable():
                remote_expr = op.remote_expr.op().replace(process_node).to_expr()
                return do_recreate(op, _kwargs, remote_expr=remote_expr)
            case rel.CachedNode():
                parent = op.parent.op().replace(process_node).to_expr()
                return do_recreate(op, _kwargs, parent=parent)
            case rel.FlightExpr() | rel.FlightUDXF():
                input_expr = op.input_expr.op().replace(process_node).to_expr()
                return do_recreate(op, _kwargs, input_expr=input_expr)
            case udf.ExprScalarUDF():
                computed_kwargs_expr = (
                    op.computed_kwargs_expr.op().replace(process_node).to_expr()
                )
                return do_recreate(
                    op, _kwargs, computed_kwargs_expr=computed_kwargs_expr
                )
            case rel.Read():
                return op
            case _:
                if isinstance(op, opaque_ops):
                    raise ValueError(f"unhandled opaque op {type(op)}")
                return op

    initial_op = expr.op() if hasattr(expr, "op") else expr
    op = initial_op.replace(process_node)
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
