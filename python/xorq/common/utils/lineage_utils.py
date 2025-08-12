from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache, singledispatch
from itertools import count
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Protocol,
    Tuple,
)

import dask.base
from attrs import evolve, field, frozen, validators
from toolz import curry

import xorq.expr.relations as rel
import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.operations as ops
from xorq.common.utils.graph_utils import gen_children_of, to_node
from xorq.vendor.ibis.expr.operations.core import Node


try:
    from rich import print as rprint
    from rich.tree import Tree as RichTree

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    rprint = print
    RichTree = None


__all__ = [
    "build_column_trees",
    "build_tree_expr",
    "TreeExpr",
    "FormatterConfig",
    "TreeFormatter",
    "PlainTextFormatter",
    "RichFormatter",
    "make_default_formatter",
    "maybe_build_tree_expr",
    "do_print_tree",
    "print_tree",
]


@frozen
class GenericNode:
    """Immutable representation of a lineage tree node."""

    op: Node = field(validator=validators.instance_of(Node))
    children: Tuple["GenericNode", ...] = field(
        factory=tuple, validator=validators.instance_of(tuple)
    )

    def __attrs_post_init__(self):
        """Validate children after initialization."""
        for i, child in enumerate(self.children):
            if not isinstance(child, GenericNode):
                raise TypeError(f"Child {i} must be GenericNode, got {type(child)}")

    def with_children(self, children: Tuple["GenericNode", ...]) -> "GenericNode":
        """Return new node with different children."""
        return self.clone(children=children)

    def map_children(
        self, fn: Callable[["GenericNode"], "GenericNode"]
    ) -> "GenericNode":
        """Transform children using provided function."""
        return self.with_children(tuple(fn(c) for c in self.children))

    def clone(self, **changes: Any) -> "GenericNode":
        """Create new instance with specified changes."""
        return evolve(self, **changes)

    def pipe(self, fn: Callable[["GenericNode"], Any]) -> Any:
        """Pipe this node through a function for composition."""
        return fn(self)


@frozen
class FormatterConfig:
    """Immutable configuration for tree formatting."""

    colors: Tuple[Tuple[str, str], ...] = field(
        factory=lambda: (
            ("table", "blue"),
            ("cached_table", "magenta"),
            ("field", "green"),
            ("literal", "yellow"),
            ("project", "white"),
            ("filter", "cyan"),
            ("join", "bright_magenta"),
            ("aggregate", "bright_yellow"),
            ("sort", "bright_black"),
            ("limit", "bright_blue"),
            ("value", "bright_red"),
            ("binary", "bright_green"),
            ("window", "bright_magenta"),
            ("udf", "bright_cyan"),
            ("remote_table", "bright_blue"),
            ("flight", "bright_white"),
            ("udxf", "magenta"),
            ("default", "white"),
        ),
        validator=validators.instance_of(tuple),
    )

    show_node_ids: bool = field(default=True, validator=validators.instance_of(bool))
    max_depth: Optional[int] = field(
        default=None, validator=validators.optional(validators.instance_of(int))
    )
    dedup: bool = field(default=True, validator=validators.instance_of(bool))
    indent: str = field(default="  ", validator=validators.instance_of(str))

    show_literal_values: bool = field(
        default=True, validator=validators.instance_of(bool)
    )
    truncate_literals: int = field(default=50, validator=validators.instance_of(int))
    show_field_details: bool = field(
        default=True, validator=validators.instance_of(bool)
    )

    def __attrs_post_init__(self):
        """Validate color mappings after initialization."""
        for item in self.colors:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(
                    f"Color mapping must be tuple of (category, color), got {item}"
                )
            category, color = item
            if not isinstance(category, str) or not isinstance(color, str):
                raise ValueError(f"Category and color must be strings, got {item}")

    def with_colors(self, **color_updates: str) -> "FormatterConfig":
        """Return new config with updated colors."""
        color_dict = dict(self.colors)
        color_dict.update(color_updates)
        return self.clone(colors=tuple(color_dict.items()))

    def with_depth_limit(self, max_depth: int) -> "FormatterConfig":
        """Return new config with depth limit."""
        return self.clone(max_depth=max_depth)

    def with_dedup(self, enabled: bool = True) -> "FormatterConfig":
        """Return new config with deduplication setting."""
        return self.clone(dedup=enabled)

    def with_node_ids(self, enabled: bool = True) -> "FormatterConfig":
        """Return new config with node ID display setting."""
        return self.clone(show_node_ids=enabled)

    def clone(self, **changes: Any) -> "FormatterConfig":
        """Create new config with specified changes."""
        return evolve(self, **changes)

    def get_color(self, category: str) -> str:
        """Get color for category with fallback to default."""
        color_dict = dict(self.colors)
        return color_dict.get(category, color_dict["default"])

    def pipe(self, fn: Callable[["FormatterConfig"], Any]) -> Any:
        """Pipe this config through a function for composition."""
        return fn(self)


_NODE_CATEGORY_MAPPING = {
    ops.Field: "field",
    ops.Literal: "literal",
    ops.Project: "project",
    ops.Filter: "filter",
    ops.Aggregate: "aggregate",
    ops.Sort: "sort",
    ops.Limit: "limit",
    ops.BinaryOp: "binary",
    ops.ValueOp: "value",
    ops.JoinChain: "join",
    ops.WindowFunction: "window",
    ops.InMemoryTable: "table",
    ops.UnboundTable: "table",
    ops.DatabaseTable: "table",
}

# Special node type mapping
_SPECIAL_NODE_MAPPING = {
    rel.RemoteTable: "remote_table",
    rel.FlightExpr: "flight",
    rel.FlightUDXF: "udxf",
    udf.ExprScalarUDF: "udf",
    rel.CachedNode: "cached_table",
    rel.Read: "table",
}


def _get_node_category(node: Node) -> str:
    """Determine node category using lookup tables."""
    # Check direct type mapping first
    node_type = type(node)
    if node_type in _NODE_CATEGORY_MAPPING:
        return _NODE_CATEGORY_MAPPING[node_type]

    # Check special types
    for special_type, category in _SPECIAL_NODE_MAPPING.items():
        if isinstance(node, special_type):
            return category

    return "default"


def maybe_truncate_text(text: str, limit: int) -> str:
    """Truncate text if it exceeds limit."""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def maybe_get_node_name(node: Node) -> Optional[str]:
    """Extract name from node if available."""
    return getattr(node, "name", None)


def maybe_get_node_value(node: Node) -> Optional[Any]:
    """Extract value from node if available."""
    return getattr(node, "value", None)


@singledispatch
def format_node_label(node: Node, config: FormatterConfig) -> str:
    """Format node label based on node type."""
    return node.__class__.__name__


@format_node_label.register(ops.Field)
def _format_field(node: ops.Field, config: FormatterConfig) -> str:
    """Format field node with optional details."""
    if config.show_field_details:
        return f"Field:{node.name}"
    return node.name


@format_node_label.register(ops.Literal)
def _format_literal(node: ops.Literal, config: FormatterConfig) -> str:
    """Format literal node with optional value display."""
    if not config.show_literal_values:
        return "Literal"

    value_str = str(node.value)
    truncated = maybe_truncate_text(value_str, config.truncate_literals)
    return f"Literal: {truncated}"


@format_node_label.register(rel.RemoteTable)
def _format_remote_table(node: rel.RemoteTable, config: FormatterConfig) -> str:
    """Format remote table node."""
    return f"RemoteTable:{node.name}"


@format_node_label.register(rel.CachedNode)
def _format_cached_node(node: rel.CachedNode, config: FormatterConfig) -> str:
    """Format cached node with storage info."""
    store = getattr(node.storage, "kind", "cache")
    name = getattr(node, "name", "")
    return f"Cache[{store}] {name}".strip()


@format_node_label.register(ops.WindowFunction)
def _format_window_function(node: ops.WindowFunction, config: FormatterConfig) -> str:
    """Format window function with details."""
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


class TreeNode(Protocol):
    """Protocol for tree nodes that can be formatted."""

    def add(self, child: "TreeNode") -> "TreeNode": ...  # Rich Tree uses add()
    def __str__(self) -> str: ...


class TreeFormatter(ABC):
    """Abstract base for tree formatters with functional interface."""

    def __init__(self, config: Optional[FormatterConfig] = None):
        self.config = config or FormatterConfig()

    @abstractmethod
    def create_tree(self, label: str) -> TreeNode:
        """Create a new tree with the given root label."""
        pass

    @abstractmethod
    def format_color(self, text: str, color: str) -> str:
        """Apply color formatting to text."""
        pass

    @abstractmethod
    def format_style(self, text: str, style: str) -> str:
        """Apply style formatting to text."""
        pass

    @abstractmethod
    def do_print_tree(self, tree: TreeNode) -> None:
        """Print the tree (side effect)."""
        pass

    def with_config(self, config: FormatterConfig) -> "TreeFormatter":
        """Return new formatter with different config."""
        return type(self)(config)

    def pipe(self, fn: Callable[["TreeFormatter"], Any]) -> Any:
        """Pipe this formatter through a function."""
        return fn(self)


class PlainTextTreeNode:
    """Simple tree node for plain text output."""

    def __init__(self, label: str, indent: str = "  "):
        self.label = label
        self.children: list[PlainTextTreeNode] = []
        self.indent = indent

    def add(self, child: "PlainTextTreeNode") -> "PlainTextTreeNode":
        """Add child and return the added child (to match Rich Tree API)."""
        self.children.append(child)
        return child

    def add_child(self, child: "PlainTextTreeNode") -> None:
        """Legacy method for compatibility."""
        self.add(child)

    def set_label(self, label: str) -> None:
        self.label = label

    def to_string(self, depth: int = 0, prefix: str = "") -> str:
        """Convert tree to string representation."""
        result = prefix + self.label + "\n"

        for i, child in enumerate(self.children):
            is_last = i == len(self.children) - 1
            child_prefix = prefix + ("└── " if is_last else "├── ")
            result += child.to_string(depth + 1, child_prefix)

        return result

    def __str__(self) -> str:
        return self.to_string()


class PlainTextFormatter(TreeFormatter):
    """Plain text formatter - no dependencies required."""

    def create_tree(self, label: str) -> PlainTextTreeNode:
        return PlainTextTreeNode(label, self.config.indent)

    def format_color(self, text: str, color: str) -> str:
        return text  # No color in plain text

    def format_style(self, text: str, style: str) -> str:
        style_mapping = {
            "bold": lambda t: f"**{t}**",
            "italic": lambda t: f"*{t}*",
            "dim": lambda t: f"({t})",
        }
        return style_mapping.get(style, lambda t: t)(text)

    def do_print_tree(self, tree: PlainTextTreeNode) -> None:
        """Print tree to stdout (side effect)."""
        print(tree.to_string())


class RichFormatter(TreeFormatter):
    """Rich-based formatter with graceful fallback to plain text."""

    def __init__(self, config: Optional[FormatterConfig] = None):
        super().__init__(config)
        self._warned = False

    def create_tree(self, label: str) -> TreeNode:
        if RICH_AVAILABLE:
            return RichTree(label)
        else:
            if not self._warned:
                print("Note: Rich not available, using plain text formatting")
                self._warned = True
            return PlainTextTreeNode(label, self.config.indent)

    def format_color(self, text: str, color: str) -> str:
        if not RICH_AVAILABLE:
            return text  # No color in fallback mode

        if color.startswith("#"):
            return f"[{color}]{text}[/]"
        return f"[{color}]{text}[/]"

    def format_style(self, text: str, style: str) -> str:
        if not RICH_AVAILABLE:
            # Use plain text styling
            style_mapping = {
                "bold": lambda t: f"**{t}**",
                "italic": lambda t: f"*{t}*",
                "dim": lambda t: f"({t})",
            }
            return style_mapping.get(style, lambda t: t)(text)

        return f"[{style}]{text}[/]"

    def do_print_tree(self, tree: TreeNode) -> None:
        """Print tree using Rich or plain text fallback."""
        if RICH_AVAILABLE and RichTree and isinstance(tree, RichTree):
            rprint(tree)
        elif isinstance(tree, PlainTextTreeNode):
            tree_formatter = PlainTextFormatter(self.config)
            tree_formatter.do_print_tree(tree)
        else:
            # Fallback - convert to string
            print(str(tree))


@frozen
class TreeExpr:
    """Immutable expression for building trees with deferred execution."""

    node: GenericNode = field(validator=validators.instance_of(GenericNode))
    config: FormatterConfig = field(factory=FormatterConfig)
    formatter: Optional[TreeFormatter] = field(default=None)

    def with_config(self, config: FormatterConfig) -> "TreeExpr":
        """Return new expression with different config."""
        return self.clone(config=config)

    def with_formatter(self, formatter: TreeFormatter) -> "TreeExpr":
        """Return new expression with different formatter."""
        return self.clone(formatter=formatter)

    def with_depth_limit(self, max_depth: int) -> "TreeExpr":
        """Return new expression with depth limit."""
        return self.with_config(self.config.with_depth_limit(max_depth))

    def with_dedup(self, enabled: bool = True) -> "TreeExpr":
        """Return new expression with deduplication setting."""
        return self.with_config(self.config.with_dedup(enabled))

    def with_colors(self, **colors: str) -> "TreeExpr":
        """Return new expression with custom colors."""
        return self.with_config(self.config.with_colors(**colors))

    def clone(self, **changes: Any) -> "TreeExpr":
        """Create new expression with changes."""
        return evolve(self, **changes)

    def build(self) -> TreeNode:
        """Build the tree structure (deferred execution)."""
        formatter = self.formatter or make_default_formatter(self.config)
        return _build_tree_recursive(self.node, formatter, self.config)

    def execute(self) -> None:
        """Execute the tree building and print (side effect)."""
        tree = self.build()
        formatter = self.formatter or make_default_formatter(self.config)
        formatter.do_print_tree(tree)

    def pipe(self, fn: Callable[["TreeExpr"], Any]) -> Any:
        """Pipe this expression through a function."""
        return fn(self)


def make_default_formatter(
    config: Optional[FormatterConfig] = None, force_plain: bool = False
) -> TreeFormatter:
    """Factory function for default formatter with options."""
    config = config or FormatterConfig()

    if force_plain or not RICH_AVAILABLE:
        return PlainTextFormatter(config)
    return RichFormatter(config)


def make_plain_formatter(
    config: Optional[FormatterConfig] = None,
) -> PlainTextFormatter:
    """Factory function for plain text formatter."""
    return PlainTextFormatter(config or FormatterConfig())


def make_rich_formatter(config: Optional[FormatterConfig] = None) -> RichFormatter:
    """Factory function for Rich formatter."""
    return RichFormatter(config or FormatterConfig())


@lru_cache(maxsize=1024)
def _get_node_token(node: GenericNode) -> str:
    """Get unique token for node deduplication."""
    op = node.op
    return dask.base.tokenize(
        (
            getattr(op, "name", None),
            getattr(op, "schema", None),
            tuple(_get_node_token(c) for c in node.children),
        )
    )


def _build_tree_recursive(
    node: GenericNode,
    formatter: TreeFormatter,
    config: FormatterConfig,
    seen: Optional[Dict[str, int]] = None,
    seq: Optional[count] = None,
    depth: int = 0,
) -> TreeNode:
    """Recursively build tree structure."""

    if seen is None:
        seen = {}
    if seq is None:
        seq = count(1)

    if config.max_depth is not None and depth > config.max_depth:
        return formatter.create_tree(formatter.format_style("…", "dim"))

    if config.dedup:
        token = _get_node_token(node)
        if token in seen:
            ref_num = seen[token]
            label = formatter.format_style(f"↻ see #{ref_num}", "italic")
            return formatter.create_tree(label)

        ref_num = next(seq)
        seen[token] = ref_num

    base_label = format_node_label(node.op, config)
    category = _get_node_category(node.op)
    colored_label = formatter.format_color(base_label, config.get_color(category))

    final_label = colored_label
    if config.show_node_ids and config.dedup:
        ref_text = formatter.format_style(f" #{seen.get(token, next(seq))}", "dim")
        final_label += ref_text

    tree_node = formatter.create_tree(final_label)

    for child in node.children:
        child_tree = _build_tree_recursive(
            child, formatter, config, seen, seq, depth + 1
        )
        tree_node.add(child_tree)

    return tree_node


def _build_column_tree_recursive(node: Node) -> GenericNode:
    """Recursively build column lineage tree."""
    match node:
        case ops.Field(rel=ops.Project(values=values)) as field_node:
            mapped = values[field_node.name]
            child = _build_column_tree_recursive(to_node(mapped))
            return GenericNode(op=field_node, children=(child,))

        case ops.Field() as field_node:
            children = tuple(
                _build_column_tree_recursive(to_node(child))
                for child in gen_children_of(field_node)
            )
            return GenericNode(op=field_node, children=children)

        case ops.Project() as proj:
            return _build_column_tree_recursive(to_node(proj.parent))

        case _:
            children = tuple(
                _build_column_tree_recursive(to_node(child))
                for child in gen_children_of(node)
            )
            return GenericNode(op=node, children=children)


def build_column_trees(expr: Any) -> Dict[str, GenericNode]:
    """Build lineage trees for each column in expression."""
    op = to_node(expr)
    cols = getattr(op, "values", None) or getattr(op, "fields", {})
    return {k: _build_column_tree_recursive(to_node(v)) for k, v in cols.items()}


def maybe_build_tree_expr(
    expr: Any, column_name: Optional[str] = None
) -> Optional[TreeExpr]:
    """Maybe build tree expression from expression and column name."""
    try:
        if column_name:
            trees = build_column_trees(expr)
            if column_name not in trees:
                return None
            return TreeExpr(node=trees[column_name])
        else:
            node = GenericNode(op=to_node(expr))
            return TreeExpr(node=node)
    except Exception:
        return None


def build_tree_expr(
    node: GenericNode,
    config: Optional[FormatterConfig] = None,
    formatter_type: str = "auto",
) -> TreeExpr:
    """Build tree expression from node with formatter selection."""
    config = config or FormatterConfig()

    if formatter_type == "plain":
        formatter = make_plain_formatter(config)
    elif formatter_type == "rich":
        formatter = make_rich_formatter(config)
    else:  # auto
        formatter = make_default_formatter(config)

    return TreeExpr(node=node, config=config, formatter=formatter)


def do_print_tree(
    node: GenericNode,
    formatter: Optional[TreeFormatter] = None,
    config: Optional[FormatterConfig] = None,
) -> None:
    """Print tree (side effect function)."""
    tree_expr = build_tree_expr(node, config)
    if formatter:
        tree_expr = tree_expr.with_formatter(formatter)
    tree_expr.execute()


@curry
def with_depth_limit(max_depth: int, tree_expr: TreeExpr) -> TreeExpr:
    """Curried function to set depth limit."""
    return tree_expr.with_depth_limit(max_depth)


@curry
def with_colors(color_map: Dict[str, str], tree_expr: TreeExpr) -> TreeExpr:
    """Curried function to set colors."""
    return tree_expr.with_colors(**color_map)


@curry
def with_formatter_type(formatter_type: str, tree_expr: TreeExpr) -> TreeExpr:
    """Curried function to set formatter type."""
    formatter_factories = {
        "rich": make_rich_formatter,
        "plain": make_plain_formatter,
        "default": make_default_formatter,
    }
    factory = formatter_factories.get(formatter_type, make_default_formatter)
    return tree_expr.with_formatter(factory(tree_expr.config))


def print_tree(
    node: GenericNode,
    max_depth: Optional[int] = None,
    dedup: bool = True,
    colors: Optional[Dict[str, str]] = None,
    formatter_type: str = "auto",
) -> None:
    """Print tree with functional pipeline."""
    config = FormatterConfig()
    if max_depth is not None:
        config = config.with_depth_limit(max_depth)
    if not dedup:
        config = config.with_dedup(False)
    if colors:
        config = config.with_colors(**colors)

    tree_expr = build_tree_expr(node, config, formatter_type)
    tree_expr.execute()
