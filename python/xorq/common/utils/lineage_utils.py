from __future__ import annotations

from functools import lru_cache, singledispatch
from itertools import count
from typing import Any, Callable, Dict, Optional, Tuple

import dask.base
from attrs import evolve, field, frozen
from attrs.validators import instance_of
from rich import print as rprint
from rich.tree import Tree
from toolz import curry, pipe

import xorq.expr.relations as rel
import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.operations as ops
from xorq.common.utils.graph_utils import children_of, to_node
from xorq.vendor.ibis.expr.operations.core import Node


__all__ = [
    "build_lineage_tree",
    "build_column_trees",
    "build_tree",
    "print_tree",
]


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


@curry
def maybe_get_project_expr(field_node: ops.Field) -> Optional[Any]:
    rel_node = field_node.rel
    if isinstance(rel_node, ops.Project):
        _, mapping = rel_node.args
        return mapping.get(field_node.name)
    return None


@curry
def build_lineage_tree(node: Node) -> GenericNode:
    if isinstance(node, ops.Field):
        return pipe(
            node,
            maybe_get_project_expr(),
            lambda expr: build_lineage_tree(to_node(expr)) if expr else None,
            lambda child_node: GenericNode(
                op=node,
                children=(
                    (child_node,)
                    if child_node
                    else ((build_lineage_tree(to_node(node.rel)),) if node.rel else ())
                ),
            ),
        )

    return GenericNode(
        op=node,
        children=tuple(
            build_lineage_tree(to_node(child)) for child in children_of(node)
        ),
    )


def build_column_trees(expr: Any) -> Dict[str, GenericNode]:
    op = to_node(expr)
    cols: Dict[str, Any] = getattr(op, "values", getattr(op, "fields", {}))
    return {k: build_lineage_tree(to_node(v)) for k, v in cols.items()}


@frozen
class ColorScheme:
    colors: Dict[str, str] = {
        "table": "[#658594]",  # dragonBlue2 (muted ocean blue)
        "cached_table": "[#8a739a]",  # dragonViolet (soft purple)
        "field": "[#8ea4a2]",  # dragonAsh (sage green)
        "literal": "[#b6927b]",  # dragonOrange2 (warm earth)
        "project": "[#c5c9c5]",  # dragonWhite (soft cloud)
        "filter": "[#87a987]",  # springViolet (forest green)
        "join": "[bold #a292a3]",  # springViolet2 (muted lavender)
        "aggregate": "[#c4b28a]",  # dragonYellow (wheat)
        "sort": "[#7d7c61]",  # comet (olive)
        "limit": "[#43436c]",  # dragonInk (deep twilight)
        "value": "[#b98d7b]",  # dragonOrange (clay)
        "binary": "[#7d957d]",  # dragonGreen2 (moss)
        "window": "[bold #8a739a]",  # bold dragonViolet
        "udf": "[#7e9cd8]",  # waveBlue1 (accent blue)
        "default": "[#a6a69c]",  # fujiGray (natural stone)
    }

    def get(self, category: str) -> str:
        return self.colors.get(category, self.colors["default"])


default_palette = ColorScheme()


def _category(node: Node) -> str:
    if isinstance(node, (ops.InMemoryTable, ops.UnboundTable, ops.DatabaseTable)):
        return "table"

    if isinstance(node, rel.RemoteTable):
        return "remote_table"
    if isinstance(node, rel.FlightExpr):
        return "flight"
    if isinstance(node, rel.FlightUDXF):
        return "udxf"
    if isinstance(node, udf.ExprScalarUDF):
        return "udxf"
    if isinstance(node, rel.CachedNode):
        return "cached_table"
    if isinstance(node, rel.Read):
        return "table"

    if isinstance(node, ops.Field):
        return "field"
    if isinstance(node, ops.Literal):
        return "literal"
    if isinstance(node, ops.Project):
        return "project"
    if isinstance(node, ops.Filter):
        return "filter"
    if isinstance(node, ops.JoinChain):
        return "join"

    if isinstance(node, ops.Aggregate):
        return "aggregate"
    if isinstance(node, ops.Sort):
        return "sort"
    if isinstance(node, ops.Limit):
        return "limit"
    if isinstance(node, (ops.BinaryOp)):
        return "binary"
    if isinstance(node, ops.WindowFunction):
        return "window"

    if isinstance(node, ops.ValueOp):
        return "value"
    return "default"


@singledispatch
def format_node(node: Node, config: Dict[str, Any] | None = None) -> str:
    config = config or {}
    palette: ColorScheme = config.get("palette", default_palette)
    cat = _category(node)
    color = palette.get(cat)
    return f"{color}{node.__class__.__name__}[/]"


@format_node.register
def _(node: ops.Field, cfg: Dict[str, Any] | None = None) -> str:
    palette: ColorScheme = (cfg or {}).get("palette", default_palette)
    col = palette.get("field")
    return f"{col}Field:{node.name}[/]"


@format_node.register
def _(node: rel.RemoteTable, cfg: Dict[str, Any] | None = None) -> str:
    palette: ColorScheme = (cfg or {}).get("palette", default_palette)
    col = palette.get("remote_table")
    return f"{col}RemoteTable:{node.name}[/]"


@format_node.register
def _(node: rel.CachedNode, cfg: Dict[str, Any] | None = None) -> str:
    palette: ColorScheme = (cfg or {}).get("palette", default_palette)
    col = palette.get("cached_table")
    store = getattr(node.storage, "kind", "cache")
    return f"{col}Cache[{store}] {node.name or ''}[/]"


@format_node.register
def _(node: rel.FlightExpr, cfg: Dict[str, Any] | None = None) -> str:
    palette: ColorScheme = (cfg or {}).get("palette", default_palette)
    col = palette.get("flight")
    return f"{col}FlightExpr ({node.input_expr})[/]"


@format_node.register
def _(node: udf.ExprScalarUDF, cfg: Dict[str, Any] | None = None) -> str:
    palette: ColorScheme = (cfg or {}).get("palette", default_palette)
    col = palette.get("udxf")
    return f"{col}ExprScalarUDF[/]"


@format_node.register
def _(node: ops.WindowFunction, cfg: Dict[str, Any] | None = None) -> str:
    palette: ColorScheme = (cfg or {}).get("palette", default_palette)
    col = palette.get("window")

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
        return f"{col}WindowFunction:\n {details}[/]"
    else:
        return f"{col}WindowFunction[/]"


@format_node.register
def _(node: ops.Literal, cfg: Dict[str, Any] | None = None) -> str:
    palette: ColorScheme = (cfg or {}).get("palette", default_palette)
    col = palette.get("literal")
    return f"{col}Literal: {node.value}[/]"


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
    palette: ColorScheme | None = None,
    dedup: bool = True,
    max_depth: int | None = None,
) -> Tree:
    cfg = {"palette": palette or default_palette}

    seen: dict[str, int] = {}
    seq = count(1)

    def _to_tree(g: GenericNode, depth: int) -> Tree:
        if max_depth is not None and depth > max_depth:
            return Tree("[dim]…[/]")

        digest = _token_node(g) if dedup else None
        if digest is not None and digest in seen:
            ref = seen[digest]
            return Tree(f"[italic dim]↻ see #{ref}[/]")

        ref = next(seq)
        if digest is not None:
            seen[digest] = ref

        label = format_node(g.op, cfg)
        if dedup:
            label += f" [grey37]#{ref}[/]"
        branch = Tree(label)

        for child in g.children:
            branch.add(_to_tree(child, depth + 1))
        return branch

    return _to_tree(node, 0)


def print_tree(
    node: GenericNode,
    *,
    palette: ColorScheme | None = None,
    dedup: bool = True,
    max_depth: int | None = None,
) -> None:
    rprint(build_tree(node, palette=palette, dedup=dedup, max_depth=max_depth))
