from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from functools import singledispatch
from itertools import count
from typing import Any, Callable, Tuple

from attrs import evolve, field, frozen
from attrs.validators import instance_of

import xorq.expr.relations as rel
import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.operations as ops
from xorq.common.utils.content_hash import content_hash
from xorq.common.utils.dasher import tokenize
from xorq.common.utils.graph_utils import (
    bfs,
    find_all_sources,
    gen_children_flight_leaf,
    gen_children_of,
    to_node,
)
from xorq.vendor.ibis.expr.operations.core import Node


__all__ = [
    "CAPABILITY_REGISTRY",
    "LineageDAG",
    "LineageRow",
    "base_kind",
    "build_column_trees",
    "build_tree",
    "compact_lineage_rows",
    "extract_lineage_dag",
    "format_compact_lineage",
    "schema_diff",
]


# ── Boundary taxonomy ──────────────────────────────────────────────────
#
# A "boundary" is a semantically-meaningful hinge in the lineage: an ingestion,
# a cache, an engine crossing, a process crossing (Flight), a tag, or a table
# source. ``compact()`` keeps exactly these and collapses everything between them
# into a ``via`` run. ``_boundary_kind`` below is the single source of truth for
# the set: every ``graph_utils.opaque_ops`` type (the descent boundaries) plus the
# non-opaque display boundaries (tables, joins, tags) registers a kind.


def _backend_label(node: Node) -> str | None:
    source = getattr(node, "source", None)
    if source is None:
        return None
    return getattr(source, "name", type(source).__name__)


@singledispatch
def _boundary_kind(node: Node) -> str | None:
    """Taxonomy string for a boundary node, or ``None`` if not a boundary."""
    return None


@_boundary_kind.register
def _(node: rel.CachedNode) -> str:
    return "cache"


@_boundary_kind.register
def _(node: rel.CacheTag) -> str:
    return "pin"


@_boundary_kind.register
def _(node: rel.RemoteTable) -> str:
    return "engine_crossing"


@_boundary_kind.register
def _(node: rel.FlightUDXF) -> str:
    return "flight_udxf"


@_boundary_kind.register
def _(node: rel.FlightExpr) -> str:
    return "flight_expr"


@_boundary_kind.register
def _(node: udf.ExprScalarUDF) -> str:
    return "udf"


@_boundary_kind.register
def _(node: rel.Read) -> str:
    # Read subclasses DatabaseTable; register it first-class so it wins dispatch.
    return "ingestion"


@_boundary_kind.register
def _(node: ops.DatabaseTable) -> str:
    return "table"


@_boundary_kind.register
def _(node: ops.InMemoryTable) -> str:
    return "table"


@_boundary_kind.register
def _(node: ops.UnboundTable) -> str:
    return "unbound"


@_boundary_kind.register
def _(node: ops.JoinChain) -> str:
    return "join"


@_boundary_kind.register
def _(node: rel.HashingTag) -> str:
    # Taxonomy: a catalog tag qualifies its kind with the tag value
    # (``tag:catalog-source``). ``base_kind`` strips the qualifier for registry
    # lookups and kind queries.
    metadata = getattr(node, "metadata", None) or {}
    tag = metadata.get("tag")
    return f"tag:{tag}" if tag else "tag"


@_boundary_kind.register
def _(node: rel.Tag) -> str:
    # CacheTag subclasses Tag but registers its own "pin" kind above.
    return "tag"


def base_kind(kind: str | None) -> str | None:
    """Taxonomy kind without its qualifier (``tag:catalog-source`` -> ``tag``)."""
    return None if kind is None else kind.split(":", 1)[0]


def _schema_dict(schema: Any) -> dict[str, str] | None:
    if schema is None:
        return None
    return {k: str(v) for k, v in schema.items()}


_SECRET_HINTS = ("password", "passwd", "secret", "token", "credential", "api_key")


def _redact(key: str, value: Any) -> Any:
    """Drop credential-looking read kwargs; strip userinfo from URI values.

    A ``Read``'s kwargs are stored verbatim in the sidecar, which is committed
    next to the build -- a ``postgres://user:pw@host/db`` uri must not travel
    with it.
    """
    if any(hint in key.lower() for hint in _SECRET_HINTS):
        return "***"
    if isinstance(value, str) and "://" in value:
        scheme, _, rest = value.partition("://")
        if "@" in rest:
            return f"{scheme}://***@{rest.rpartition('@')[2]}"
    return value


_READ_PATH_KEYS = (
    "path",
    "hash_path",
    "paths",
    "source",
    "source_list",
    "uri",
    "file",
    "glob",
)


def _read_path_or_uri(kwargs: Mapping[str, Any]) -> Any:
    for key in _READ_PATH_KEYS:
        if (value := kwargs.get(key)) is not None:
            return _redact(key, _to_jsonable(value))
    return None


def _read_format(method_name: str | None) -> str | None:
    """``read_parquet`` -> ``parquet`` (the source kind for non-file readers)."""
    if not method_name:
        return None
    return method_name.removeprefix("deferred_").removeprefix("read_") or None


def _namespace_extras(node: Node) -> dict[str, Any]:
    namespace = getattr(node, "namespace", None)
    return {
        "database": getattr(namespace, "database", None),
        "catalog": getattr(namespace, "catalog", None),
    }


_PREDICATE_SYMBOLS = {
    "Equals": "==",
    "NotEquals": "!=",
    "Less": "<",
    "LessEqual": "<=",
    "Greater": ">",
    "GreaterEqual": ">=",
}


def _predicate_ref(value: Any) -> str:
    """Column name for a field reference, op type name for anything else.

    Deliberately never renders a literal's *value*: the summary is for display
    and the sidecar is committed.
    """
    if isinstance(value, ops.Field):
        return value.name
    return type(value).__name__


def _predicate_summary(node: ops.JoinChain) -> str | None:
    parts = []
    for link in getattr(node, "rest", ()) or ():
        how = getattr(link, "how", "join")
        refs = ", ".join(
            f"{_predicate_ref(getattr(pred, 'left', pred))} "
            f"{_PREDICATE_SYMBOLS.get(type(pred).__name__, type(pred).__name__)} "
            f"{_predicate_ref(getattr(pred, 'right', pred))}"
            if hasattr(pred, "left")
            else type(pred).__name__
            for pred in getattr(link, "predicates", ()) or ()
        )
        parts.append(f"{how}: {refs}" if refs else str(how))
    return "; ".join(parts) or None


def _udf_signature(node: Node) -> str | None:
    dtypes = tuple(
        str(getattr(getattr(node, name, None), "dtype", "?"))
        for name in getattr(node, "__argnames__", ())
    )
    out = getattr(node, "dtype", None)
    if not dtypes or out is None:
        return None
    return f"inputs: [{', '.join(dtypes)}] -> output: {out}"


def schema_diff(
    before: Mapping[str, str] | None, after: Mapping[str, str] | None
) -> dict[str, Any] | None:
    """Column-level delta between two ``{col: type}`` schemas.

    Computed at extraction time and stored in the sidecar (rather than
    recomputed per TUI open) so external readers get the transition for free.
    A rename is only inferred when exactly one column was dropped and one added
    *and* they share a dtype that is unique to the pair -- anything more
    ambiguous stays reported as an add plus a drop.
    """
    if not before or not after:
        return None
    added = {k: v for k, v in after.items() if k not in before}
    removed = {k: before[k] for k in before if k not in after}
    retyped = {
        k: [before[k], after[k]] for k in before if k in after and before[k] != after[k]
    }
    renamed: dict[str, str] = {}
    if len(added) == 1 and len(removed) == 1:
        ((new_col, new_type),) = added.items()
        ((old_col, old_type),) = removed.items()
        if new_type == old_type:
            renamed = {old_col: new_col}
            added, removed = {}, {}
    if not (added or removed or retyped or renamed):
        return None
    return {
        "added": added,
        "removed": sorted(removed),
        "retyped": retyped,
        "renamed": renamed,
    }


def _server_factory_label(make_server: Any) -> str | None:
    """Name of the factory that builds the Flight server.

    The callable itself is never serialized and never *called* (that would start
    a server); only its name travels. No attempt is made to infer a flavour
    (``mtls``/``plain``) from that name: the defaults are a bare ``FlightServer``
    class for ``FlightExpr`` and a ``make_mtls_server`` closure for
    ``FlightUDXF``, so name-sniffing yields a class name as often as a flavour.
    """
    if make_server is None:
        return None
    return getattr(make_server, "__name__", None) or type(make_server).__name__


def _flight_command(node: rel.FlightExpr) -> str | None:
    """The command string the FlightExpr client sends on the wire.

    ``FlightExpr.to_rbr`` registers an ``UnboundExprExchanger`` over
    ``unbound_expr`` and exchanges on *its* ``command``, so the exchanger is the
    only authority for the value -- recomputing the format string here would rot
    silently. Constructing one is pure (walk + rename + tokenize, no server, no
    connection); a failure degrades to no field rather than breaking extraction.
    """
    unbound_expr = getattr(node, "unbound_expr", None)
    if unbound_expr is None:
        return None
    from xorq.flight.exchanger import UnboundExprExchanger  # noqa: PLC0415

    try:
        return UnboundExprExchanger(unbound_expr).command
    except Exception:
        return None


@singledispatch
def _boundary_extras(node: Node) -> dict[str, Any]:
    """Per-kind inline facts stored on a boundary node (intrinsic 1:1 facts)."""
    return {}


def _cache_extras(cache: Any) -> dict[str, Any]:
    """Where a cache writes and how it keys, without touching the filesystem.

    ``cache_path`` is the storage's *relative* path only. The resolved directory
    is ``base_path / relative_path`` with ``base_path`` defaulting to the user's
    cache dir -- an absolute, per-machine path that must not travel in a sidecar
    committed next to the build. The per-entry parquet path is deliberately not
    derived either: it needs ``cache.calc_key(expr)``, which stats the source
    data under ``ModificationTimeStrategy``. Storages with no path
    (``SourceStorage``) get none; their location is the ``backend``.
    """
    if cache is None:
        return {}
    storage = getattr(cache, "storage", None)
    strategy = getattr(cache, "strategy", None)
    relative_path = getattr(storage, "relative_path", None)
    return {
        "cache_kind": type(cache).__name__,
        "cache_path": None if relative_path is None else str(relative_path),
        "cache_key_prefix": getattr(strategy, "key_prefix", None),
    }


@_boundary_extras.register
def _(node: rel.CachedNode) -> dict[str, Any]:
    return {"backend": _backend_label(node), **_cache_extras(node.cache)}


@_boundary_extras.register
def _(node: rel.CacheTag) -> dict[str, Any]:
    # A pin's identity is its cache key: the frozen read's table name.
    return {
        "backend": _backend_label(node),
        "cache_key": getattr(node.parent, "name", None),
        **_cache_extras(getattr(node, "cache", None)),
    }


def _source_backend(remote_expr: Any) -> str | None:
    """Backend the data came from, for the visual ``duckdb → xorq`` hop.

    ``remote_expr``'s *root* op usually has no ``source`` of its own (it is a
    ``Project``/``Filter``), so the origin has to be resolved by walking to the
    leaves. A multi-source input has no single origin to name: report none rather
    than picking one arbitrarily.
    """
    if remote_expr is None:
        return None
    sources = find_all_sources(remote_expr, execution_only=True)
    if len(sources) != 1:
        return None
    (source,) = sources
    return getattr(source, "name", type(source).__name__)


@_boundary_extras.register
def _(node: rel.RemoteTable) -> dict[str, Any]:
    # The crossing has two ends: ``source`` is where the data lands,
    # ``remote_expr``'s backend is where it came from.
    return {
        "backend": _backend_label(node),
        "remote_name": getattr(node, "name", None),
        "source_backend": _source_backend(getattr(node, "remote_expr", None)),
    }


@_boundary_extras.register
def _(node: rel.Read) -> dict[str, Any]:
    kwargs = dict(getattr(node, "read_kwargs", None) or ())
    method_name = getattr(node, "method_name", None)
    extras: dict[str, Any] = {
        "backend": _backend_label(node),
        "read_kind": method_name,
        "format": _read_format(method_name),
        "path_or_uri": _read_path_or_uri(kwargs),
    }
    if kwargs:
        extras["read_kwargs"] = {
            str(k): _redact(str(k), v) for k, v in _to_jsonable(kwargs).items()
        }
    return extras


@_boundary_extras.register
def _(node: ops.DatabaseTable) -> dict[str, Any]:
    return {
        "backend": _backend_label(node),
        "table_name": getattr(node, "name", None),
        **_namespace_extras(node),
    }


@_boundary_extras.register
def _(node: ops.UnboundTable) -> dict[str, Any]:
    return {"table_name": getattr(node, "name", None), **_namespace_extras(node)}


@_boundary_extras.register
def _(node: ops.JoinChain) -> dict[str, Any]:
    return {
        "n_inputs": 1 + len(getattr(node, "rest", ()) or ()),
        "predicates_summary": _predicate_summary(node),
    }


@_boundary_extras.register
def _(node: udf.ExprScalarUDF) -> dict[str, Any]:
    return {
        "udf_name": getattr(node, "__func_name__", None) or type(node).__name__,
        "udf_signature": _udf_signature(node),
    }


@_boundary_extras.register
def _(node: rel.FlightUDXF) -> dict[str, Any]:
    udxf = getattr(node, "udxf", None)
    input_expr = getattr(node, "input_expr", None)
    schema_in = input_expr.schema() if input_expr is not None else None
    schema_in_observed = _schema_dict(schema_in)
    schema_out = _schema_dict(getattr(node, "schema", None))
    return {
        "process_boundary": True,
        "udxf_class": getattr(udxf, "__name__", type(udxf).__name__ if udxf else None),
        "udxf_command": getattr(udxf, "command", None),
        "server_factory": _server_factory_label(getattr(node, "make_server", None)),
        "schema_in_required": _schema_dict(getattr(udxf, "schema_in_required", None)),
        "schema_in_observed": schema_in_observed,
        "schema_out": schema_out,
        # The UDXF is the one boundary where the schema actually changes; store
        # the delta so readers (TUI, audit tools) do not recompute it.
        "schema_diff": schema_diff(schema_in_observed, schema_out),
        "do_instrument_reader": getattr(node, "do_instrument_reader", None),
    }


@_boundary_extras.register
def _(node: rel.FlightExpr) -> dict[str, Any]:
    input_expr = getattr(node, "input_expr", None)
    schema_in = input_expr.schema() if input_expr is not None else None
    make_server = getattr(node, "make_server", None)
    return {
        "process_boundary": True,
        "server_factory": _server_factory_label(make_server),
        "flight_command": _flight_command(node),
        "input_schema": _schema_dict(schema_in),
        "output_schema": _schema_dict(getattr(node, "schema", None)),
        "do_instrument_reader": getattr(node, "do_instrument_reader", None),
    }


# ── Capability registry (code, keyed by boundary_kind) ─────────────────
#
# ``capabilities(node) = CAPABILITY_REGISTRY[kind]`` is what a kind *can*
# expose; ``available(node)`` scans ``overlays`` for what is actually present.
# Nothing here is stored per node.
CAPABILITY_REGISTRY: dict[str, tuple[str, ...]] = {
    "ingestion": ("schema",),
    "cache": ("schema", "cache_path"),
    "pin": ("schema", "tag_metadata", "cache_key"),
    "engine_crossing": ("schema",),
    "table": ("schema",),
    "unbound": ("schema",),
    "join": ("schema",),
    "udf": ("schema",),
    "tag": ("tag_metadata",),
    "flight_udxf": (
        "schema_in_required",
        "schema_in_observed",
        "schema_out",
        "schema_diff",
        "nested",
    ),
    "flight_expr": ("input_schema", "output_schema", "flight_command", "nested"),
}

_FLIGHT_TYPES = (rel.FlightExpr, rel.FlightUDXF)


@frozen
class LineageRow:
    """One rendered line, in pieces: tree glyphs, label, boundary kind, ``via``.

    A renderer (``catalog/tui.py``) styles the pieces separately; ``text`` is the
    plain-text line, so ``str(TextTree)`` and a styled render never drift.
    """

    prefix: str = field(validator=instance_of(str))
    label: str = field(validator=instance_of(str))
    kind: str | None = field(default=None)
    via: Tuple[str, ...] = field(factory=tuple, validator=instance_of(tuple))

    @property
    def via_suffix(self) -> str:
        return _via_suffix(list(self.via))

    @property
    def text(self) -> str:
        return self.prefix + self.label + self.via_suffix


@frozen
class TextTree:
    """Plain-text tree for displaying lineage."""

    label: str = field(validator=instance_of(str))
    children: Tuple["TextTree", ...] = field(
        factory=tuple, validator=instance_of(tuple)
    )
    # Only the compact lineage tree sets these; column trees leave them empty.
    kind: str | None = field(default=None)
    via: Tuple[str, ...] = field(factory=tuple, validator=instance_of(tuple))

    def rows(
        self, prefix: str = "", is_last: bool = True, is_root: bool = True
    ) -> tuple[LineageRow, ...]:
        if is_root:
            line_prefix = ""
            child_prefix = ""
        else:
            line_prefix = prefix + ("└── " if is_last else "├── ")
            child_prefix = prefix + ("    " if is_last else "│   ")
        row = LineageRow(
            prefix=line_prefix, label=self.label, kind=self.kind, via=self.via
        )
        return (row,) + tuple(
            descendant
            for i, child in enumerate(self.children)
            for descendant in child.rows(
                child_prefix, i == len(self.children) - 1, False
            )
        )

    def _lines(
        self, prefix: str = "", is_last: bool = True, is_root: bool = True
    ) -> tuple[str, ...]:
        return tuple(row.text for row in self.rows(prefix, is_last, is_root))

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
    # Do NOT stringify input_expr (it is extracted as a nested sub-DAG); the old
    # ``FlightExpr ({node.input_expr})`` label duplicated the whole sub-tree.
    return f"FlightExpr:{getattr(node, 'name', '')}"


@format_node.register
def _(node: rel.FlightUDXF) -> str:
    cls = getattr(getattr(node, "udxf", None), "__name__", "udxf")
    return f"FlightUDXF[{cls}]"


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


LINEAGE_VERSION = 2
ROOT_SCOPE = "root"


def _norm_edge(edge: Any) -> dict:
    """Normalise a stored edge to ``{from, to, scope}``.

    Tolerates legacy 2-tuples/lists (``[from, to]``) by defaulting the scope to
    ``"root"``; a 3-tuple carries an explicit scope; a dict is taken as-is.
    """
    if isinstance(edge, Mapping):
        return {
            "from": edge["from"],
            "to": edge["to"],
            "scope": edge.get("scope", ROOT_SCOPE),
        }
    frm, to, *rest = edge
    return {"from": frm, "to": to, "scope": rest[0] if rest else ROOT_SCOPE}


@frozen
class LineageDAG:
    """Typed container for a serialisable, self-contained lineage DAG.

    Storage model (a): ONE shared node table (outer + all nested Flight lineage),
    deduped by ``snapshot_hash``; edges are scope-tagged ``{from, to, scope}``
    (``"root"`` for the outer graph, the owning Flight label for nested edges).
    ``compact()`` and ``scope()`` are computed *views* over this table, not
    stored sub-documents.
    """

    nodes: tuple[dict, ...] = field(validator=instance_of(tuple))
    edges: tuple[dict, ...] = field(validator=instance_of(tuple))
    root: str = field(validator=instance_of(str))
    overlays: dict = field(factory=dict, validator=instance_of(dict))
    version: int = field(default=LINEAGE_VERSION, validator=instance_of(int))

    @property
    def by_id(self) -> dict[str, dict]:
        return {n["id"]: n for n in self.nodes}

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "root": self.root,
            "nodes": list(self.nodes),
            "edges": [dict(_norm_edge(e)) for e in self.edges],
            "overlays": dict(self.overlays),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> LineageDAG:
        # Tolerate-and-degrade: legacy sidecars (integer ids, 2-tuple edges, no
        # version/overlays) load without migration -- they render, just without
        # the enriched dimensions.
        if "root" not in raw:
            raise KeyError("lineage dict missing required key 'root'")
        return cls(
            nodes=tuple(raw.get("nodes", ())),
            edges=tuple(_norm_edge(e) for e in raw.get("edges", ())),
            root=raw["root"],
            overlays=dict(raw.get("overlays", {})),
            version=int(raw.get("version", 1)),
        )

    # ── query API ──────────────────────────────────────────────────────

    def resolve(self, handle: Any) -> tuple[dict, ...]:
        """Resolve a handle to node dict(s).

        Accepts a ``snapshot_hash``, an ``@label`` id, a tag string, a
        ``boundary_kind``, or a predicate callable over node dicts.
        """
        by_id = self.by_id
        if callable(handle):
            return tuple(n for n in self.nodes if handle(n))
        if isinstance(handle, str) and handle in by_id:
            return (by_id[handle],)
        matches = tuple(
            n
            for n in self.nodes
            if n.get("snapshot_hash") == handle
            or handle in (n.get("boundary_kind"), base_kind(n.get("boundary_kind")))
        )
        if matches:
            return matches
        return tuple(
            n
            for n in self.nodes
            if isinstance(n.get("tag_metadata"), Mapping)
            and n["tag_metadata"].get("tag") == handle
        )

    def boundaries(self, kind: str | None = None) -> tuple[dict, ...]:
        """Boundary nodes, optionally of one *kind*.

        A qualified kind matches on either form: ``kind="tag"`` selects
        ``tag:catalog-source`` too, ``kind="tag:catalog-source"`` only that one.
        """
        return tuple(
            n
            for n in self.nodes
            if n.get("is_boundary")
            and (
                kind is None
                or kind in (n.get("boundary_kind"), base_kind(n.get("boundary_kind")))
            )
        )

    def capabilities(self, node: dict) -> tuple[str, ...]:
        """What dimensions this node's kind *can* expose (code registry)."""
        return CAPABILITY_REGISTRY.get(base_kind(node.get("boundary_kind")), ())

    def available(self, node: dict) -> tuple[str, ...]:
        """What overlay dimensions actually reference this node (present)."""
        nid = node["id"]
        return tuple(
            dim
            for dim, payload in self.overlays.items()
            if _overlay_touches(payload, nid)
        )

    def expand(self, selector: Any, *dims: str) -> dict:
        """Filter stored overlays for the resolved node(s) along ``dims``.

        Static: expand is a filter over what is already serialised, never a
        recompute.  Returns ``{dim: payload}``.
        """
        targets = {n["id"] for n in self.resolve(selector)}
        wanted = dims or tuple(self.overlays)
        out: dict[str, Any] = {}
        for dim in wanted:
            payload = self.overlays.get(dim)
            if payload is not None:
                out[dim] = _overlay_filter(payload, targets)
        return out

    def _scope_root(self, scope: str) -> str | None:
        """Root id of *scope*: the DAG root, or the Flight node's ``nested_root``."""
        if scope == ROOT_SCOPE:
            return self.root
        return self.by_id.get(scope, {}).get("nested_root")

    def scope(self, flight_label: str) -> dict:
        """Nested subgraph view for a Flight boundary: nodes/edges in its scope."""
        edges = tuple(
            e for e in map(_norm_edge, self.edges) if e["scope"] == flight_label
        )
        root = self._scope_root(flight_label)
        # Seed with the scope root: a nested lineage that is a *single* node (the
        # Flight input is a bare table) has no edges at all, and deriving the id
        # set from edges alone would drop the very node ``root`` points at.
        ids = {e["from"] for e in edges} | {e["to"] for e in edges} | {root} - {None}
        by_id = self.by_id
        return {
            "nodes": tuple(by_id[i] for i in ids if i in by_id),
            "edges": edges,
            "root": root,
            "scope": flight_label,
        }

    def compact(self, scope: str = ROOT_SCOPE) -> dict:
        """Boundary-only view: collapse non-boundary runs into ``via`` on the
        surviving boundary→boundary edges.

        A small generic adjacency BFS over the id-graph (stored labels + edges),
        NOT the op-graph -- at render time there are no ops.
        """
        by_id = self.by_id
        adj: dict[str, list[str]] = defaultdict(list)
        in_scope: set[str] = set()
        for e in self.edges:
            e = _norm_edge(e)
            if e["scope"] == scope:
                adj[e["from"]].append(e["to"])
                in_scope |= {e["from"], e["to"]}

        # A legacy/partial sidecar can name a scope with no recorded root; the
        # view then holds only what its edges reach.
        scope_root = self._scope_root(scope)
        kept = {i for i in in_scope if by_id.get(i, {}).get("is_boundary")}
        if scope_root is not None:
            kept.add(scope_root)

        # (from, to) -> collapsed run. Two boundaries can be joined by several
        # non-boundary paths (a Sort reaching its parent both directly and through
        # its SortKey, say); keep the most direct one so the view stays one line
        # per boundary pair.
        runs: dict[tuple[str, str], tuple[str, ...]] = {}
        for boundary in kept:
            stack = [(child, ()) for child in adj.get(boundary, ())]
            seen: set[str] = set()
            while stack:
                cur, via = stack.pop()
                if cur in kept:
                    key = (boundary, cur)
                    if key not in runs or len(via) < len(runs[key]):
                        runs[key] = via
                    continue
                if cur in seen:
                    continue
                seen.add(cur)
                via_next = via + (by_id[cur].get("type", "?"),)
                for child in adj.get(cur, ()):
                    stack.append((child, via_next))

        return {
            "nodes": tuple(by_id[i] for i in sorted(kept) if i in by_id),
            "edges": tuple(
                {"from": frm, "to": to, "via": list(via), "scope": scope}
                for (frm, to), via in sorted(runs.items())
            ),
            "root": scope_root,
            "scope": scope,
        }


def _overlay_touches(payload: Any, node_id: str) -> bool:
    if isinstance(payload, Mapping):
        return node_id in payload or any(
            _overlay_touches(v, node_id) for v in payload.values()
        )
    if isinstance(payload, (list, tuple)):
        return node_id in payload or any(_overlay_touches(v, node_id) for v in payload)
    return payload == node_id


def _overlay_filter(payload: Any, targets: set[str]) -> Any:
    """Restrict an overlay payload to entries touching ``targets``.

    Overlays are a deferred scaffold (columns/engines/cost slot in here); this
    keeps mapping payloads whose keys are target node ids and leaves other shapes
    untouched.
    """
    if not targets:
        return payload
    if isinstance(payload, Mapping):
        return {k: v for k, v in payload.items() if k in targets}
    return payload


def _to_jsonable(v: Any) -> Any:
    """Convert frozen tag metadata to JSON-friendly primitives.

    Tag metadata may carry FrozenDict / FrozenOrderedDict / nested tuples
    (e.g. BSL's serialized dimensions and measures). Stringifying them with
    repr() loses structure for downstream consumers like `xorq catalog show
    --json | jq`; recursing through Mapping/Sequence preserves it.
    """
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, Mapping):
        return {str(k): _to_jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return [_to_jsonable(x) for x in sorted(v, key=str)]
    return str(v)


def _op_slug(node: Node) -> str:
    name = type(node).__name__
    return "".join(("_" + c.lower()) if c.isupper() else c for c in name).lstrip("_")


def _arg_token(value: Any, node_hash: Callable[[Node], str]) -> Any:
    """Hashable, address-free rendering of one op argument."""
    if isinstance(value, Node):
        return node_hash(value)
    if isinstance(value, Mapping):
        return tuple((str(k), _arg_token(v, node_hash)) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return tuple(_arg_token(v, node_hash) for v in value)
    return str(value)


def _structural_token(node: Node, node_hash: Callable[[Node], str]) -> str:
    """Deterministic identity from an op's type and arguments.

    ``repr()`` of an ibis op is address-based, so identity is rebuilt from
    ``__argnames__``; child nodes contribute their own (memoized) hash, keeping
    this linear in graph size.
    """
    return tokenize(
        (type(node).__name__,)
        + tuple(
            (name, _arg_token(getattr(node, name, None), node_hash))
            for name in getattr(node, "__argnames__", ())
        )
    )


def make_node_hasher() -> Callable[[Node], str]:
    """Return a memoized ``node -> hash`` function for one extraction pass.

    Relations hash with :func:`content_hash` -- the identity ``ibis_yaml`` stores
    as ``snapshot_hash``, so a relation cross-references between expr.yaml and the
    sidecar by hash. Everything else (``Field``, ``Literal``, ``SortKey``,
    ``JoinLink``, …) gets a structural token: ``content_hash`` snapshot-normalizes
    and may compile to SQL, which is both wasted work and outright unsupported for
    some value ops (a ``SortKey`` cannot be re-projected), and none of them appear
    as expr.yaml nodes anyway. Both forms are deterministic, so dedup is stable
    across runs.
    """
    memo: dict[Node, str] = {}

    def node_hash(node: Node) -> str:
        if (hashed := memo.get(node)) is not None:
            return hashed
        hashed = None
        if isinstance(node, ops.Relation):
            try:
                hashed = content_hash(node)
            except Exception:  # noqa: BLE001 - lineage must never break a build
                hashed = None
        if hashed is None:
            hashed = _structural_token(node, node_hash)
        memo[node] = hashed
        return hashed

    return node_hash


def _node_dict(node: Node, label: str, snapshot_hash: str) -> dict:
    d: dict[str, Any] = {
        "id": label,
        "snapshot_hash": snapshot_hash,
        "type": type(node).__name__,
        "label": format_node(node),
    }
    kind = _boundary_kind(node)
    if kind is not None:
        d["is_boundary"] = True
        d["boundary_kind"] = kind
        for k, v in _boundary_extras(node).items():
            if v is not None:
                d[k] = v
    else:
        d["is_boundary"] = False
    schema = _schema_dict(getattr(node, "schema", None))
    if schema is not None:
        d["schema"] = schema
    if isinstance(node, rel.Tag):
        d["tag_metadata"] = _to_jsonable(node.metadata)
    return d


def extract_lineage_dag(expr: Any) -> LineageDAG:
    """Extract a compact, self-contained lineage DAG from an expression.

    The outer walk treats ``FlightExpr``/``FlightUDXF`` as leaves
    (``gen_children_flight_leaf``): a Flight node's ``input_expr`` is NOT
    flattened into the outer graph but recursed as a *nested* sub-DAG whose nodes
    live in the same shared table under a ``scope`` tag (the owning Flight
    label).  Nodes are deduped across scopes by ``content_hash`` (the durable
    ``snapshot_hash``, shared with expr.yaml); the ``@op_N`` label is a readable
    within-doc handle.  BFS dedup guards nested-of-nested recursion.
    """
    root = to_node(expr)

    hash_to_label: dict[str, str] = {}
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    edge_seen: set[tuple[str, str, str]] = set()
    expanded_flights: set[str] = set()
    counter = count(1)
    node_hash = make_node_hasher()

    def register(node: Node) -> str:
        h = node_hash(node)
        label = hash_to_label.get(h)
        if label is None:
            label = f"@{_op_slug(node)}_{next(counter)}"
            hash_to_label[h] = label
            nodes[label] = _node_dict(node, label, h)
        return label

    def add_edge(frm: str, to: str, scope: str) -> None:
        key = (frm, to, scope)
        if key not in edge_seen:
            edge_seen.add(key)
            edges.append({"from": frm, "to": to, "scope": scope})

    def walk(subroot: Node, scope: str) -> str:
        graph = bfs(subroot, children=gen_children_flight_leaf)
        labels = {node: register(node) for node in graph}
        for node, children in graph.items():
            for child in children:
                add_edge(labels[node], labels[child], scope)
        # Expand each Flight node's input_expr as a nested sub-DAG (once).
        for node, label in labels.items():
            input_expr = getattr(node, "input_expr", None)
            if not isinstance(node, _FLIGHT_TYPES) or input_expr is None:
                continue
            if label in expanded_flights:
                continue
            expanded_flights.add(label)
            nodes[label]["nested_root"] = walk(to_node(input_expr), label)
        return labels[subroot]

    root_label = walk(root, ROOT_SCOPE)

    return LineageDAG(
        nodes=tuple(nodes.values()),
        edges=tuple(edges),
        root=root_label,
        overlays={},
        version=LINEAGE_VERSION,
    )


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
        tok = tokenize(
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


# ── compact-view rendering (TUI) ───────────────────────────────────────


def _schema_diff_suffix(diff: Mapping[str, Any] | None) -> str:
    """One-line column delta: ``(+score, -raw, b→c, a: int64→float64)``."""
    if not diff:
        return ""
    parts = (
        [f"+{col}" for col in diff.get("added") or ()]
        + [f"-{col}" for col in diff.get("removed") or ()]
        + [f"{old}→{new}" for old, new in (diff.get("renamed") or {}).items()]
        + [
            f"{col}: {before}→{after}"
            for col, (before, after) in (diff.get("retyped") or {}).items()
        ]
    )
    if not parts:
        return ""
    shown = ", ".join(parts[:3]) + ("…" if len(parts) > 3 else "")
    return f"   ({shown})"


def _compact_node_label(node: dict) -> str:
    """Per-kind one-line label for a boundary node in the compact view."""
    kind = base_kind(node.get("boundary_kind"))
    if kind == "flight_udxf":
        cls = node.get("udxf_class", "udxf")
        n_in = len(node.get("schema_in_observed") or {})
        n_out = len(node.get("schema_out") or {})
        diff = _schema_diff_suffix(node.get("schema_diff"))
        return f"UDXF[{cls}] : {n_in}→{n_out} cols{diff}"
    if kind == "flight_expr":
        n_in = len(node.get("input_schema") or {})
        n_out = len(node.get("output_schema") or {})
        return f"FlightExpr : {n_in}→{n_out} cols"
    if kind == "cache":
        return f"Cache[{node.get('cache_kind', 'cache')}]"
    if kind == "pin":
        key = node.get("cache_key") or ""
        return f"Pin[{key[:8]}]" if key else "Pin"
    if kind == "engine_crossing":
        target = node.get("backend", "?")
        source = node.get("source_backend")
        # A same-backend crossing (into_backend onto its own source) has no hop
        # to show; rendering "duckdb → duckdb" would be noise.
        if source and source != target:
            return f"RemoteTable[{source} → {target}]"
        return f"→ {target}"
    if kind == "ingestion":
        backend = node.get("backend")
        return f"Read[{backend}]" if backend else "Read"
    if kind in ("table", "unbound"):
        backend = node.get("backend")
        name = node.get("table_name") or ""
        return f"{backend}:{name}" if backend else (name or node.get("label", "table"))
    if kind == "join":
        return f"Join({node.get('n_inputs', '?')} inputs)"
    if kind == "udf":
        return f"UDF[{node.get('udf_name', '?')}]"
    if kind == "tag":
        tm = node.get("tag_metadata") or {}
        return f"Tag[{tm.get('tag', '')}]" if isinstance(tm, Mapping) else "Tag"
    return node.get("label", node.get("type", "?"))


def _via_suffix(via: list) -> str:
    if not via:
        return ""
    shown = ", ".join(via[:3]) + ("…" if len(via) > 3 else "")
    return f"   via [{shown}]"


def _compact_lineage_tree(dag: LineageDAG) -> TextTree | None:
    """Compact boundary tree, or ``None`` when there is nothing to render.

    Non-boundary runs collapse into ``via [...]`` on the surviving edges; Flight
    boundaries carry their nested input lineage as an indented sub-tree. Each
    node keeps its ``kind`` so a renderer can style per boundary kind without
    re-parsing the label.
    """
    by_id = dag.by_id

    def build(view: dict, nid: str, seen: frozenset, marker: str = "") -> TextTree:
        node = by_id.get(nid, {"label": nid, "type": "?"})
        children_of: dict = defaultdict(list)
        for edge in view["edges"]:
            children_of[edge["from"]].append((edge["to"], edge.get("via", [])))

        kids: list[TextTree] = []

        nested_root = node.get("nested_root")
        if nested_root is not None:
            # A Flight boundary's input lineage lives in its own scope: render it
            # as a sub-tree marked with the crossing arrow at its root.
            kids.append(
                build(
                    dag.compact(scope=nid),
                    nested_root,
                    frozenset({nested_root}),
                    "↳ ",
                )
            )

        for child_id, via in children_of.get(nid, ()):
            child = by_id[child_id]
            if child_id in seen:
                kids.append(
                    TextTree(
                        f"↻ {_compact_node_label(child)}",
                        kind=base_kind(child.get("boundary_kind")),
                    )
                )
                continue
            subtree = build(view, child_id, seen | {child_id})
            kids.append(evolve(subtree, via=tuple(via)))

        return TextTree(
            marker + _compact_node_label(node),
            tuple(kids),
            kind=base_kind(node.get("boundary_kind")),
        )

    view = dag.compact()
    if not view["nodes"] or view["root"] is None:
        return None
    return build(view, view["root"], frozenset({view["root"]}))


def compact_lineage_rows(dag: LineageDAG) -> tuple[LineageRow, ...]:
    """One row per rendered line, with the tree glyphs, label, kind and ``via``
    kept apart so a renderer can style each piece (see ``catalog/tui.py``)."""
    tree = _compact_lineage_tree(dag)
    return () if tree is None else tree.rows()


def format_compact_lineage(dag: LineageDAG) -> str:
    """Render a :class:`LineageDAG` as a compact boundary-only text tree."""
    tree = _compact_lineage_tree(dag)
    return "(empty)" if tree is None else str(tree)
