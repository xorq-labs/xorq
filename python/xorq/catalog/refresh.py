"""Rebuild a loaded build over the sources a drift report found changed.

Runs after ``load_expr``: changed sources get their live schema and every op
above them is recreated, so signature validation re-runs. A ``CacheTag`` pin
is kept as recorded.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import xorq.vendor.ibis.expr.operations as ops
from xorq.catalog.drift import (
    RECORDED_SCHEMA_KEYS,
    LeafReport,
    iter_leaf_reports,
    make_profile,
    unchecked_leaves,
)
from xorq.catalog.enums import LeafKind, RebaseTag, Verdict
from xorq.catalog.inspection import BuildRecord, SourceLeaf, join_read_path
from xorq.common.exceptions import (
    InternalError,
    SchemaRefreshError,
    UnmatchedSourceError,
    XorqError,
)
from xorq.common.utils.graph_utils import (
    _opaque_lookup,
    opaque_ops,
    recreate_opaque,
    replace_nodes,
    to_node,
    walk_nodes,
)
from xorq.common.utils.node_utils import recreate, update_read_kwargs
from xorq.expr.relations import (
    CachedNode,
    CacheTag,
    FlightExpr,
    FlightUDXF,
    HashingTag,
    PyIcebergTable,
    Read,
    RemoteTable,
    Tag,
    TeeNode,
    snapshot_pin,
)
from xorq.ibis_yaml.enums import ReadKwarg
from xorq.vendor.ibis.backends.profiles import Profile
from xorq.vendor.ibis.common.annotations import ValidationError
from xorq.vendor.ibis.common.graph import Node
from xorq.vendor.ibis.expr.schema import Schema
from xorq.vendor.ibis.expr.types import Expr


# Ops storing a schema their parent determines -> the parent field.
SCHEMA_FOLLOWS_PARENT = {
    CachedNode: "parent",
    RemoteTable: "remote_expr",
    TeeNode: "parent",
    Tag: "parent",
}

# (kind, name, snapshot_id, schema): a source's identity without its profile.
SourceIdentity = tuple[str, str, int | None, Schema]
# (kind, profile content hash, name, snapshot_id, schema).
LeafKey = tuple[str, str | None, str, int | None, Schema]

# Exact source op class -> the `op` name it serializes as in expr.yaml.
SOURCE_LEAF_KINDS: dict[type[Node], LeafKind] = {
    ops.DatabaseTable: LeafKind.DATABASE_TABLE,
    PyIcebergTable: LeafKind.DATABASE_TABLE,
    Read: LeafKind.READ,
}


def leaf_profile_key(leaf: SourceLeaf, record: BuildRecord) -> str | None:
    """The content hash of the profile ``leaf`` records, or ``None`` if none."""
    profile_dict = record.get_profile_dict(leaf)
    return None if profile_dict is None else make_profile(profile_dict).content_hash


def leaf_key(leaf: SourceLeaf, record: BuildRecord) -> LeafKey:
    """``leaf``'s identity, matched against ``op_key``.

    Profile content hash, not backend name, so one table name on two
    connections is two sources; the snapshot pin, so a pinned and a current
    read of one table are two sources whose live schemas may differ.
    """
    return (
        str(leaf.kind),
        leaf_profile_key(leaf, record),
        leaf.name,
        leaf.snapshot_id,
        leaf.recorded,
    )


def source_identity(node: Node) -> SourceIdentity | None:
    """``op_key`` minus the profile; never touches ``node.source``.

    Exact class: ``CachedNode``/``RemoteTable`` subclass ``DatabaseTable``.
    """
    match kind := SOURCE_LEAF_KINDS.get(type(node)):
        case LeafKind.DATABASE_TABLE:
            name = node.to_expr().get_name()
        case LeafKind.READ:
            # Same fallback as `SourceLeaf`.
            path = dict(node.read_kwargs).get(ReadKwarg.hash_path) or node.name
            name = join_read_path(path)
        case _:
            return None
    return (kind, name, snapshot_pin(node), node.schema)


def op_key(node: Node) -> LeafKey | None:
    """``node``'s ``leaf_key``, or ``None`` when it is not a source.

    Profile name carries a session-local index; content hash does not.
    """
    if (identity := source_identity(node)) is None:
        return None
    (kind, name, snapshot_id, schema) = identity
    profile = Profile.from_con(node.source)
    profile_key = None if profile is None else profile.content_hash
    return (kind, profile_key, name, snapshot_id, schema)


def with_live_schema(node: Node, schema: Schema) -> Node:
    """``node`` carrying ``schema``, including a read's schema ``read_kwargs``.

    duckdb's ``types`` override can't be derived from a schema, so it is dropped.
    A key recorded as ``None`` is a bound default, not an instruction (duckdb's
    ``read_json`` binds ``columns=None``), so it stays unset.
    """
    if not isinstance(node, Read):
        return recreate(node, schema=schema)
    recorded = dict(node.read_kwargs)
    instructions = tuple(
        (key, schema)
        for key in RECORDED_SCHEMA_KEYS - {ReadKwarg.types}
        if recorded.get(key) is not None
    )
    read_kwargs = tuple(
        (key, value)
        for key, value in update_read_kwargs(node.read_kwargs, instructions)
        if key != ReadKwarg.types
    )
    return recreate(node, schema=schema, read_kwargs=read_kwargs)


def rebuild(node: Node, build: Callable[[], Node]) -> Node:
    """``build()``, an op rejecting its new inputs named after ``node``.

    Other errors are rewrite bugs, not drift, and propagate unchanged.
    """
    try:
        return build()
    except (SchemaRefreshError, InternalError):
        raise
    except (ValidationError, XorqError) as e:
        raise SchemaRefreshError(type(node).__name__, e) from e


def rebind_unbound(unbound_expr: Any, schema: Schema) -> Any:
    """``unbound_expr`` rebuilt over its one ``UnboundTable`` carrying ``schema``.

    Each op above is recreated, so one that no longer fits is named.
    Found via ``__children__``, the edges ``replace`` follows.
    """
    if not (tables := to_node(unbound_expr).find(ops.UnboundTable)):
        raise ValueError("unbound_expr has no UnboundTable reachable via __children__")
    (table, *_) = tables
    moved = recreate(table, schema=schema)

    def replacer(node: Node, kwargs: dict | None) -> Node:
        if node == table:
            return moved
        if kwargs is None:
            return node
        return rebuild(node, lambda: node.__recreate__(kwargs))

    return to_node(unbound_expr).replace(replacer).to_expr()


def revalidate_flight(node: Node, overrides: dict) -> dict:
    """``overrides`` checked against a moved ``input_expr``.

    Flight ops validate only in ``from_expr``/``from_exprs``, not ``__init__``.
    A ``FlightExpr``'s ``unbound_expr`` is rebuilt over the new input schema.
    """
    if (input_expr := overrides.get("input_expr")) is None:
        return overrides
    # Bare `ValueError`: too broad for `rebuild` to catch.
    try:
        match node:
            case FlightUDXF():
                return overrides | {
                    "schema": FlightUDXF.validate_schema(input_expr, node.udxf)
                }
            case FlightExpr():
                unbound_expr = rebind_unbound(node.unbound_expr, input_expr.schema())
                FlightExpr.validate_schema(input_expr, unbound_expr)
                return overrides | {
                    "unbound_expr": unbound_expr,
                    "schema": unbound_expr.schema(),
                }
    except ValueError as e:
        raise SchemaRefreshError(type(node).__name__, e) from e
    return overrides


def recreate_over(node: Node, overrides: dict) -> Node:
    """``node`` recreated with ``overrides``; a stored schema follows its parent."""
    if (attr := _opaque_lookup(node, SCHEMA_FOLLOWS_PARENT)) is not None:
        parent = overrides.get(attr, getattr(node, attr))
        overrides = overrides | {"schema": to_node(parent).schema}
    return recreate_opaque(node, revalidate_flight(node, overrides))


def recreate_changed(node: Node, kwargs: dict) -> Node:
    """``node`` over the inputs ``replace_nodes`` rewrote; ``node`` if none moved."""
    changed = {
        name: value
        for name, value in kwargs.items()
        if not (isinstance(value, Expr) and value.op() is to_node(getattr(node, name)))
    }
    return rebuild(node, lambda: recreate_over(node, changed)) if changed else node


def refuse(
    offenders: Iterable[tuple[str, str]],
    *,
    separator: str = ", ",
    error_type: type[SchemaRefreshError] = SchemaRefreshError,
) -> None:
    """Raise one ``error_type`` for ``(kind, detail)`` pairs, labeled with
    their distinct kinds in order."""
    kinds, details = zip(*offenders)
    op_name = ", ".join(dict.fromkeys(map(str, kinds)))
    raise error_type(op_name, LookupError(separator.join(details)))


def with_renames(node: Node, renames: Mapping[str, str]) -> Node:
    """``node`` under a marked rename that gives each live column its
    expression name; ``node`` itself when there is none.

    ``renames`` maps expression name -> live name, as ``Table.rename`` takes
    it. The rename is a Project listing every column of ``node``, so the
    ``RebaseTag.RENAME`` mark above it holds the mapping: a later refresh
    rebuilds the rename from it instead of recreating a stale column list.
    """
    if not renames:
        return node
    # `Table.rename` over a live column no rename moves away would keep that
    # column and silently drop the renamed one: refuse it. A chain or swap
    # (`a <- x, b <- a`) moves it, and is fine.
    moved = set(renames.values())
    if clashes := [
        name for name in renames if name in node.schema and name not in moved
    ]:
        refuse(
            (
                type(node).__name__,
                f"{name!r} is still a live column, and no rename moves it away",
            )
            for name in clashes
        )
    renamed = node.to_expr().rename(dict(renames))
    marked = renamed.hashing_tag(
        RebaseTag.RENAME, renames=tuple(sorted(renames.items()))
    )
    return to_node(marked)


def marked_renames(node: Node) -> dict[str, str] | None:
    """The mapping a ``with_renames`` mark holds, or ``None`` if ``node`` isn't one."""
    if isinstance(node, HashingTag) and node.tag == RebaseTag.RENAME:
        return dict(node.metadata["renames"])
    return None


def compose_renames(
    held: Mapping[str, str], declared: Mapping[str, str]
) -> dict[str, str]:
    """``held`` (expression name -> recorded column), then ``declared``
    (recorded column -> live column): expression name -> live column.

    ``declared`` names only source columns, as ``check-sources`` prints them;
    one ``held`` doesn't cover keeps its name in the expression.
    """
    composed = {name: declared.get(column, column) for name, column in held.items()}
    covered = set(held.values())
    composed |= {old: new for old, new in declared.items() if old not in covered}
    return {name: column for name, column in composed.items() if name != column}


def marked_projects(expr: Any) -> frozenset[Node]:
    """The rename under each ``with_renames`` mark over a source, opaque
    sub-expressions included."""
    return frozenset(
        mark.parent
        for mark in walk_nodes(HashingTag, expr)
        if marked_renames(mark) is not None
        and isinstance(mark.parent, ops.Project)
        and source_identity(mark.parent.parent) is not None
    )


def refresh_schemas(
    expr: Any,
    live: Mapping[LeafKey, Schema],
    renames: Mapping[LeafKey, Mapping[str, str]] = MappingProxyType({}),
) -> Any:
    """``expr`` rebuilt over ``live`` (``leaf_key`` -> live schema).

    ``renames`` (``leaf_key`` -> recorded name -> live name) puts a marked
    rename above a refreshed source, so every op over it still sees its
    recorded names. A source already under a mark gets that rename rebuilt
    over its live columns, composed with any new ``renames``. Raises if a key
    matches no source, or is renamed but not refreshed: that source would
    stay stale silently.
    """
    if stray := [key for key in renames if key not in live]:
        refuse(
            (
                (kind, f"{name} is renamed but not refreshed")
                for (kind, _, name, _, _) in stray
            ),
            error_type=UnmatchedSourceError,
        )
    if not live:
        return expr
    matched: set[LeafKey] = set()
    # Screen before `op_key`: resolving a profile connects a lazy backend.
    candidates = {
        (kind, name, snapshot_id, schema)
        for (kind, _, name, snapshot_id, schema) in live
    }
    # Recreated bottom-up, the rename under a mark would keep its stale column
    # list over the refreshed source: the walk stops there, and the mark
    # rebuilds it from the mapping it holds.
    projects = marked_projects(expr)

    def refreshed_key(node: Node) -> LeafKey | None:
        if source_identity(node) in candidates and (key := op_key(node)) in live:
            return key
        return None

    def replacer(node: Node, kwargs: dict | None) -> Node:
        if (held := marked_renames(node)) is not None and node.parent in projects:
            source = node.parent.parent
            if (key := refreshed_key(source)) is None:
                # Its source is unchanged: the rename stands as recorded.
                return node
            matched.add(key)
            composed = compose_renames(held, renames.get(key, {}))
            return rebuild(
                node,
                lambda: with_renames(with_live_schema(source, live[key]), composed),
            )
        if (key := refreshed_key(node)) is not None:
            matched.add(key)
            return rebuild(
                node,
                lambda: with_renames(
                    with_live_schema(node, live[key]), renames.get(key, {})
                ),
            )
        # An opaque op is rebuilt by `recreate_changed`, over its rewritten edges.
        if kwargs is None or isinstance(node, opaque_ops):
            return node
        return rebuild(node, lambda: recreate_over(node, kwargs))

    refreshed = replace_nodes(
        replacer,
        expr,
        recreate=recreate_changed,
        stop=lambda node: isinstance(node, CacheTag),
        filter=lambda node: node not in projects,
    ).to_expr()
    if unmatched := [key for key in live if key not in matched]:
        refuse(
            (
                (kind, f"{name} matched no source of the loaded expression")
                for (kind, _, name, _, _) in unmatched
            ),
            error_type=UnmatchedSourceError,
        )
    return refreshed


def live_schemas(
    record: BuildRecord, reports: Iterable[LeafReport]
) -> dict[LeafKey, Schema]:
    """The ``refresh_schemas`` mapping for the changed leaves in ``reports``.

    Raises, naming every offender, on an uncomparable source or on a key seen
    at two schemas (two reads of one path with different options).
    """
    found: dict[LeafKey, dict[Schema, LeafReport]] = {}
    uncomparable = []
    for report in reports:
        match report.verdict:
            case Verdict.EQUAL | Verdict.CHANGED:
                key = leaf_key(report.leaf, record)
                found.setdefault(key, {})[report.live] = report
            case _:
                uncomparable.append(report)
    ambiguous = [
        next(iter(by_live.values())) for by_live in found.values() if len(by_live) > 1
    ]
    if uncomparable or ambiguous:
        refuse(
            [
                (
                    report.leaf.kind,
                    f"{report.leaf.name} is {report.verdict}"
                    + (f": {report.error}" if report.error else ""),
                )
                for report in uncomparable
            ]
            + [
                (
                    report.leaf.kind,
                    f"{report.leaf.name} is read more than once, and its reads "
                    "disagree on its live schema",
                )
                for report in ambiguous
            ],
            separator="; ",
        )
    return {
        key: live
        for key, by_live in found.items()
        for (live, report) in by_live.items()
        if report.verdict == Verdict.CHANGED
    }


def check_refreshable(record: BuildRecord) -> None:
    """Raise, naming each, if ``record`` has external sources the sweep skips."""
    if unchecked := unchecked_leaves(record):
        refuse(
            (leaf.kind, f"{leaf.name} cannot be probed without writing to it")
            for leaf in unchecked
        )


def refresh_build(
    build_path: str | Path, con_cache: dict | None = None, **kwargs: Any
) -> Any:
    """Load ``build_path`` rebuilt over its drifted sources; sweeps before loading."""
    from xorq.ibis_yaml.compiler import load_expr  # noqa: PLC0415

    record = BuildRecord.from_build_dir(build_path)
    check_refreshable(record)
    live = live_schemas(record, iter_leaf_reports(record, con_cache))
    return refresh_schemas(load_expr(build_path, **kwargs), live)
