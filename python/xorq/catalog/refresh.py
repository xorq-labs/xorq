"""Rebuild a loaded build over the sources a drift report found changed.

Runs after ``load_expr``: changed sources get their live schema and every op
above them is recreated, so signature validation re-runs. A ``CacheTag`` pin
is kept as recorded.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from xorq.catalog.drift import (
    RECORDED_SCHEMA_KEYS,
    LeafReport,
    iter_leaf_reports,
    make_profile,
    unchecked_leaves,
)
from xorq.catalog.enums import LeafKind, Verdict
from xorq.catalog.inspection import BuildRecord, SourceLeaf, join_read_path
from xorq.common.exceptions import InternalError, SchemaRefreshError, XorqError
from xorq.common.utils.graph_utils import (
    OPAQUE_SPECS,
    _opaque_lookup,
    _require_expr_args_recorded,
    _require_registered_if_expr_bearing,
    to_node,
)
from xorq.common.utils.node_utils import recreate, update_read_kwargs
from xorq.expr.relations import (
    CachedNode,
    CacheTag,
    FlightExpr,
    FlightUDXF,
    Read,
    RemoteTable,
    Tag,
    TeeNode,
)
from xorq.ibis_yaml.enums import ReadKwarg
from xorq.vendor.ibis.backends.profiles import Profile
from xorq.vendor.ibis.common.annotations import ValidationError
from xorq.vendor.ibis.common.graph import Node
from xorq.vendor.ibis.expr.schema import Schema


# Ops storing a schema their parent determines -> the parent field.
SCHEMA_FOLLOWS_PARENT = {
    CachedNode: "parent",
    RemoteTable: "remote_expr",
    TeeNode: "parent",
    Tag: "parent",
}


def leaf_profile_key(leaf: SourceLeaf, record: BuildRecord) -> str | None:
    """The content hash of the profile ``leaf`` records, or ``None`` if none."""
    profile_dict = record.get_profile_dict(leaf)
    return None if profile_dict is None else make_profile(profile_dict).content_hash


def leaf_key(leaf: SourceLeaf, record: BuildRecord) -> tuple:
    """``leaf``'s identity, matched against ``op_key``.

    Profile content hash, not backend name, so one table name on two
    connections is two sources.
    """
    return (
        str(leaf.kind),
        leaf_profile_key(leaf, record),
        leaf.name,
        leaf.recorded,
    )


def source_identity(node: Node) -> tuple | None:
    """``op_key`` minus the profile; never touches ``node.source``.

    Exact type name: ``CachedNode``/``RemoteTable`` subclass ``DatabaseTable``.
    """
    match type(node).__name__:
        case LeafKind.DATABASE_TABLE:
            name = node.to_expr().get_name()
        case LeafKind.READ:
            # Same fallback as `SourceLeaf`.
            path = dict(node.read_kwargs).get(ReadKwarg.hash_path) or node.name
            name = join_read_path(path)
        case _:
            return None
    return (type(node).__name__, name, node.schema)


def op_key(node: Node) -> tuple | None:
    """``node``'s ``leaf_key``, or ``None`` when it is not a source.

    Profile name carries a session-local index; content hash does not.
    """
    if (identity := source_identity(node)) is None:
        return None
    (kind, name, schema) = identity
    profile = Profile.from_con(node.source)
    profile_key = None if profile is None else profile.content_hash
    return (kind, profile_key, name, schema)


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


def revalidate_flight(node: Node, overrides: dict) -> dict:
    """``overrides`` checked against a moved ``input_expr``.

    Flight ops validate only in ``from_expr``/``from_exprs``, not ``__init__``.
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
                FlightExpr.validate_schema(input_expr, node.unbound_expr)
    except ValueError as e:
        raise SchemaRefreshError(type(node).__name__, e) from e
    return overrides


def recreate_over(node: Node, overrides: dict) -> Node:
    """``node`` recreated with ``overrides``; a stored schema follows its parent."""
    if (attr := _opaque_lookup(node, SCHEMA_FOLLOWS_PARENT)) is not None:
        parent = overrides.get(attr, getattr(node, attr))
        overrides = overrides | {"schema": to_node(parent).schema}
    return recreate(node, **revalidate_flight(node, overrides))


def refuse(offenders: Iterable[tuple[str, str]], separator: str = ", ") -> None:
    """Raise one ``SchemaRefreshError`` for ``(kind, detail)`` pairs, labeled
    with their distinct kinds in order."""
    kinds, details = zip(*offenders)
    op_name = ", ".join(dict.fromkeys(map(str, kinds)))
    raise SchemaRefreshError(op_name, LookupError(separator.join(details)))


def refresh_schemas(expr: Any, live: Mapping[tuple, Schema]) -> Any:
    """``expr`` rebuilt over ``live`` (``leaf_key`` -> live schema).

    Raises if a key matches no source: that source would stay stale silently.
    """
    if not live:
        return expr
    memo: dict[Node, Node] = {}
    matched: set[tuple] = set()
    # Screen before `op_key`: resolving a profile connects a lazy backend.
    candidates = {(kind, name, schema) for (kind, _, name, schema) in live}

    def rewrite(node: Node) -> Node:
        if node not in memo:
            memo[node] = node.replace(replacer)
        return memo[node]

    def replacer(node: Node, kwargs: dict | None) -> Node:
        if isinstance(node, CacheTag):
            return node
        if source_identity(node) in candidates and (key := op_key(node)) in live:
            matched.add(key)
            return rebuild(node, lambda: with_live_schema(node, live[key]))
        overrides = dict(kwargs or {})
        rebound = node
        # `replace_nodes`'s tripwires: an unregistered Expr field would be
        # skipped here and keep its stale schema silently.
        if (spec := _opaque_lookup(node, OPAQUE_SPECS)) is None:
            _require_registered_if_expr_bearing(node)
        else:
            _require_expr_args_recorded(node)
            for edge in spec.descend_edges:
                sub = to_node(getattr(node, edge))
                if (new := rewrite(sub)) is sub:
                    continue
                if spec.rebind is not None:
                    rebound = rebuild(node, lambda: spec.rebind(node, new.to_expr()))
                else:
                    as_node = isinstance(getattr(node, edge), Node)
                    overrides[edge] = new if as_node else new.to_expr()
        if not overrides and rebound is node:
            return node
        return rebuild(node, lambda: recreate_over(rebound, overrides))

    refreshed = rewrite(to_node(expr)).to_expr()
    if unmatched := [key for key in live if key not in matched]:
        refuse(
            (kind, f"{name} matched no source of the loaded expression")
            for (kind, _, name, _) in unmatched
        )
    return refreshed


def live_schemas(record: BuildRecord, reports: Iterable[LeafReport]) -> dict:
    """The ``refresh_schemas`` mapping for the changed leaves in ``reports``.

    Raises, naming every offender, on an uncomparable source or on a key seen
    at two schemas (two reads of one path with different options).
    """
    found: dict[tuple, dict[Schema, LeafReport]] = {}
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
