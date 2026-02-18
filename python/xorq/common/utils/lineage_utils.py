from __future__ import annotations

from functools import lru_cache, singledispatch
from itertools import count
from typing import Any, Callable, Tuple

import dask.base
from attrs import evolve, field, frozen
from attrs.validators import instance_of

import xorq.expr.relations as rel
import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.operations as ops
from xorq.common.utils.graph_utils import (
    gen_children_of,
    to_node,
)
from xorq.vendor.ibis.expr.operations.core import Node


__all__ = [
    "build_column_trees",
    "build_tree",
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


def _build_column_tree(node: Node) -> GenericNode:
    match node:
        case ops.Field(rel=ops.Project(values=values)) as field_node:
            # include the field and recurse into its mapped expression
            mapped = values[field_node.name]
            child = _build_column_tree(to_node(mapped))
            return GenericNode(op=field_node, children=(child,))

        case ops.Field() as field_node:
            children = tuple(
                _build_column_tree(to_node(child))
                for child in gen_children_of(field_node)
            )
            return GenericNode(op=field_node, children=children)

        case ops.Project() as proj:
            return _build_column_tree(to_node(proj.parent))

        case _:
            children = tuple(
                _build_column_tree(to_node(child)) for child in gen_children_of(node)
            )
            return GenericNode(op=node, children=children)


def build_column_trees(expr: Any) -> dict[str, GenericNode]:
    """Builds a lineage tree for each column in the expression."""
    op = to_node(expr)
    cols = getattr(op, "values", None) or getattr(op, "fields", {})
    return {k: _build_column_tree(to_node(v)) for k, v in cols.items()}


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


@lru_cache
def _token_node(g: GenericNode) -> str:
    """unique token for the node based on its operation and children (useful in
    deduplication)"""
    op = g.op
    return dask.base.tokenize(
        (
            getattr(op, "name", None),  # is name always set?
            getattr(op, "schema", None),
            tuple(_token_node(c) for c in g.children),
        )
    )


def build_tree(
    node: GenericNode,
    *,
    dedup: bool = True,
    max_depth: int | None = None,
) -> TextTree:
    seen: dict[str, int] = {}
    seq = count(1)

    def _to_tree(g: GenericNode, depth: int) -> TextTree:
        if max_depth is not None and depth > max_depth:
            return TextTree("…")

        digest = _token_node(g) if dedup else None
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
