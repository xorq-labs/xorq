from typing import Any, OrderedDict, Tuple

import xorq.expr.relations as rel
import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.operations as ops
from xorq.vendor.ibis import Expr
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
            gen = map(to_node, raw_children)
    yield from filter(None, gen)


def bfs(node):
    from collections import deque  # noqa: PLC0415

    from xorq.vendor.ibis.common.graph import Graph  # noqa: PLC0415

    queue = deque((to_node(node),))
    dct = {}
    while queue:
        if (node := queue.popleft()) not in dct:
            children = tuple(gen_children_of(node))
            dct[node] = children
            queue.extend(children)
    return Graph(dct)


def walk_nodes(node_types, expr):
    # TODO should this function use an ordered set
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

        to_visit += (
            child
            for child in OrderedDict.fromkeys(gen_children_of(node))
            if child not in visited
        )

    return result


def replace_nodes(replacer, expr):
    # Cache results of opaque sub-expression traversals by their root node.
    # Sub-expression roots are often shared across multiple opaque nodes (e.g.
    # each pipeline step's ExprScalarUDF references accumulated sub-expressions
    # that overlap heavily), so without this memo each shared root gets
    # re-traversed once per reference — O(n²) for a depth-n pipeline.
    sub_expr_memo = {}

    def do_recreate(op, _kwargs, **kwargs):
        kwargs = dict(zip(op.__argnames__, op.__args__)) | (_kwargs or {}) | kwargs
        return op.__recreate__(kwargs)

    def _replace_sub(sub_op):
        if sub_op not in sub_expr_memo:
            sub_expr_memo[sub_op] = sub_op.replace(process_node).to_expr()
        return sub_expr_memo[sub_op]

    def process_node(op, _kwargs):
        op = replacer(op, _kwargs)
        match op:
            case rel.RemoteTable():
                remote_expr = _replace_sub(op.remote_expr.op())
                return do_recreate(op, _kwargs, remote_expr=remote_expr)
            case rel.CachedNode():
                parent = _replace_sub(op.parent.op())
                return do_recreate(op, _kwargs, parent=parent)
            case rel.FlightExpr() | rel.FlightUDXF():
                input_expr = _replace_sub(op.input_expr.op())
                return do_recreate(op, _kwargs, input_expr=input_expr)
            case udf.ExprScalarUDF():
                computed_kwargs_expr = _replace_sub(op.computed_kwargs_expr.op())
                with_cke = op.with_computed_kwargs_expr(computed_kwargs_expr)
                return do_recreate(with_cke, _kwargs)
            case rel.Read():
                return op
            case _:
                if isinstance(op, opaque_ops):
                    raise ValueError(f"unhandled opaque op {type(op)}")
                return op

    initial_op = expr.op() if hasattr(expr, "op") else expr
    op = initial_op.replace(process_node)
    return op


def replace_sources(source_mapping, expr):
    """Rewrite an expression graph, replacing backend sources.

    Every node that carries a ``source`` attribute (DatabaseTable, Read,
    RemoteTable, CachedNode, FlightExpr, FlightUDXF, SQLQueryResult, …) is
    recreated with the mapped replacement when its current source is found
    in *source_mapping*.  The mapping is keyed by backend identity (``id``).

    Sub-expressions reachable only through opaque fields (``remote_expr``,
    ``parent``, ``input_expr``, ``computed_kwargs_expr``) are rewritten
    recursively via the existing ``replace_nodes`` infrastructure.

    Parameters
    ----------
    source_mapping : dict[int, Any]
        ``{id(old_backend): new_backend, ...}``
    expr : Expr | Node
        The expression to rewrite.

    Returns
    -------
    Expr
        A new expression with sources replaced.
    """

    def _maybe_replace_cache(cache):
        """Rebuild a Cache object if its storage.source is in the mapping."""
        from attr import evolve  # noqa: PLC0415

        storage = getattr(cache, "storage", None)
        if storage is None:
            return cache
        source = getattr(storage, "source", None)
        if source is None or id(source) not in source_mapping:
            return cache
        new_storage = evolve(storage, source=source_mapping[id(source)])
        return evolve(cache, storage=new_storage)

    def replacer(node, kwargs):
        overrides = {}

        source = getattr(node, "source", None)
        if source is not None and id(source) in source_mapping:
            overrides["source"] = source_mapping[id(source)]

        cache = getattr(node, "cache", None)
        if cache is not None:
            new_cache = _maybe_replace_cache(cache)
            if new_cache is not cache:
                overrides["cache"] = new_cache

        if overrides or kwargs:
            merged = dict(zip(node.__argnames__, node.__args__))
            if kwargs:
                merged |= kwargs
            merged |= overrides
            return node.__recreate__(merged)
        return node

    return replace_nodes(replacer, expr).to_expr()


def get_ordered_unique_sources(nodes):
    # Use id() for deduplication because backend __hash__ collides for
    # same-class instances and __eq__ only differs by session-local idx.
    sources, seen = (), set()
    for source in (node.source for node in nodes):
        if id(source) not in seen:
            seen.add(id(source))
            sources += (source,)
    return sources


def find_all_sources(expr):
    import xorq.vendor.ibis.expr.operations as ops  # noqa: PLC0415

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
    sources = get_ordered_unique_sources(nodes)
    return sources
