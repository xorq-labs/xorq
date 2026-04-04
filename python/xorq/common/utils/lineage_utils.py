from __future__ import annotations

from functools import singledispatch
from itertools import count
from typing import Any, Callable, Tuple

import dask.base
from attrs import evolve, field, frozen
from attrs.validators import instance_of

import xorq.expr.relations as rel
import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.operations as ops
from xorq.common.utils.graph_utils import (
    bfs,
    gen_children_of,
    to_node,
)
from xorq.vendor.ibis.expr.operations.core import Node


__all__ = [
    "build_column_trees",
    "build_tree",
    "extract_lineage_dag",
]


@frozen
class TextTree:
    """Plain-text tree for displaying lineage."""

    label: str = field(validator=instance_of(str))
    children: Tuple["TextTree", ...] = field(
        factory=tuple, validator=instance_of(tuple)
    )

    def _lines(
        self, prefix: str = "", is_last: bool = True, is_root: bool = True
    ) -> tuple[str, ...]:
        if is_root:
            line = self.label
            child_prefix = ""
        else:
            connector = "└── " if is_last else "├── "
            line = prefix + connector + self.label
            child_prefix = prefix + ("    " if is_last else "│   ")
        return (line,) + tuple(
            grandchild_line
            for i, child in enumerate(self.children)
            for grandchild_line in child._lines(
                child_prefix, i == len(self.children) - 1, False
            )
        )

    def __str__(self) -> str:
        return "\n".join(self._lines())


@frozen
class GenericNode:
    op: Node = field(validator=instance_of(Node))
    children: Tuple["GenericNode", ...] = field(
        factory=tuple, validator=instance_of(tuple)
    )

    def map_children(
        self, fn: Callable[["GenericNode"], "GenericNode"]
    ) -> "GenericNode":
        return evolve(self, children=tuple(fn(c) for c in self.children))

    def clone(self, **changes: Any) -> "GenericNode":
        return evolve(self, **changes)


def _build_column_tree(
    node: Node, _results: dict[Node, GenericNode] | None = None
) -> GenericNode:
    if _results is None:
        _results = {}
    if node in _results:
        return _results[node]

    graph, _ = bfs(node).toposort()

    for n in graph:
        if n in _results:
            continue
        match n:
            case ops.Field(rel=ops.Project(values=values)) as field_node:
                # include the field and follow it into its mapped expression
                child = _results[to_node(values[field_node.name])]
                _results[n] = GenericNode(op=field_node, children=(child,))

            case ops.Field() as field_node:
                children = tuple(_results[c] for c in gen_children_of(field_node))
                _results[n] = GenericNode(op=field_node, children=children)

            case ops.Project() as proj:
                # Project is transparent: resolve to its parent's GenericNode
                _results[n] = _results[to_node(proj.parent)]

            case _:
                children = tuple(_results[c] for c in gen_children_of(n))
                _results[n] = GenericNode(op=n, children=children)

    return _results[node]


def build_column_trees(expr: Any) -> dict[str, GenericNode]:
    """Builds a lineage tree for each column in the expression."""
    op = to_node(expr)
    cols = getattr(op, "values", None) or getattr(op, "fields", {})
    shared: dict[Node, GenericNode] = {}
    return {k: _build_column_tree(to_node(v), shared) for k, v in cols.items()}


@singledispatch
def format_node(node: Node) -> str:
    return node.__class__.__name__


@format_node.register
def _(node: ops.Field) -> str:
    return f"Field:{node.name}"


@format_node.register
def _(node: rel.RemoteTable) -> str:
    return f"RemoteTable:{node.name}"


@format_node.register
def _(node: rel.CachedNode) -> str:
    store = getattr(node.cache, "kind", "cache")
    return f"Cache[{store}] {getattr(node, 'name', '')}"


@format_node.register
def _(node: rel.FlightExpr) -> str:
    return f"FlightExpr ({node.input_expr})"


@format_node.register
def _(node: udf.ExprScalarUDF) -> str:
    return "ExprScalarUDF"


@format_node.register
def _(node: ops.WindowFunction) -> str:
    parts = []
    if node.order_by:
        parts.append(f"order_by: {node.order_by}")
    if node.group_by:
        parts.append(f"group_by: {node.group_by}")
    if node.start:
        parts.append(f"start: {node.start}")
    if node.end:
        parts.append(f"end: {node.end}")

    if parts:
        details = "\n ".join(parts)
        return f"WindowFunction:\n {details}"
    return "WindowFunction"


@format_node.register
def _(node: ops.Literal) -> str:
    return f"Literal: {node.value}"


def build_tree(
    node: GenericNode,
    *,
    dedup: bool = True,
    max_depth: int | None = None,
) -> TextTree:
    seen: dict[str, int] = {}
    seq = count(1)
    token_memo: dict[int, str] = {}

    def _token(g: GenericNode) -> str:
        gid = id(g)
        if gid in token_memo:
            return token_memo[gid]
        op = g.op
        tok = dask.base.tokenize(
            (
                getattr(op, "name", None),
                getattr(op, "schema", None),
                tuple(_token(c) for c in g.children),
            )
        )
        token_memo[gid] = tok
        return tok

    def _to_tree(g: GenericNode, depth: int) -> TextTree:
        if max_depth is not None and depth > max_depth:
            return TextTree("…")

        digest = _token(g) if dedup else None
        if digest is not None and digest in seen:
            ref = seen[digest]
            return TextTree(f"↻ see #{ref}")

        ref = next(seq)
        if digest is not None:
            seen[digest] = ref

        label = format_node(g.op)
        if dedup:
            label += f" #{ref}"
        children = tuple(_to_tree(child, depth + 1) for child in g.children)
        return TextTree(label, children=children)

    return _to_tree(node, 0)


def extract_lineage_dag(expr: Any) -> dict:
    """Extract a full DAG representation of expression lineage.

    Uses ``bfs`` and ``gen_children_of`` from :mod:`graph_utils` for a
    complete traversal that correctly follows opaque sub-expressions
    (RemoteTable, CachedNode, FlightExpr, ExprScalarUDF, etc.).

    Returns
    -------
    dict
        ``{"nodes": [...], "edges": [...], "root": "node_0"}``

        Each node dict has keys: ``id``, ``op``, ``name``, ``label`` (the
        rich ``format_node`` string), and optionally ``schema`` (a mapping
        of column name to ``{"dtype": str, "nullable": bool}``).

        Each edge dict has keys: ``source`` (upstream node id) and ``target``
        (downstream node id).
    """
    from xorq.expr.relations import Tag  # noqa: PLC0415
    from xorq.vendor.ibis.expr.operations.relations import Relation  # noqa: PLC0415

    root_node = to_node(expr)
    graph = bfs(root_node)

    node_to_id: dict[Node, str] = {}
    nodes: list[dict] = []

    for i, (node, _children) in enumerate(graph.items()):
        node_id = f"node_{i}"
        node_to_id[node] = node_id

        node_data: dict[str, Any] = {
            "id": node_id,
            "op": node.__class__.__name__,
            "name": getattr(node, "name", "") or "",
            "label": format_node(node),
        }

        if isinstance(node, Relation):
            try:
                schema = node.schema
                node_data["schema"] = {
                    col_name: {
                        "dtype": str(dtype),
                        "nullable": dtype.nullable,
                    }
                    for col_name, dtype in schema.items()
                }
            except (AttributeError, TypeError):
                pass

            if isinstance(node, Tag):
                meta = dict(node.metadata) if node.metadata else {}
                node_data["tag"] = meta

        nodes.append(node_data)

    # Build edges after all IDs are assigned (children appear after parents in BFS).
    edges: list[dict] = []
    for node, children in graph.items():
        node_id = node_to_id[node]
        for child in children:
            child_id = node_to_id.get(child)
            if child_id is not None:
                edges.append({"source": child_id, "target": node_id})

    return {
        "nodes": nodes,
        "edges": edges,
        "root": node_to_id.get(root_node),
    }
