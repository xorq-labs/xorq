"""Re-derive a loaded expression onto the sources a drift report says moved.

The loader is not involved. A build loads exactly as it always has, against the
schemas it recorded, and ``refresh_schemas`` rewrites the loaded graph
afterwards: every source ``check-sources`` reported ``changed`` is rebuilt with
the schema it has now, and every op above it is rebuilt through
``__recreate__``, which re-runs its signature validation. An op that can no
longer be constructed over its new inputs raises ``SchemaRefreshError`` naming
itself -- a ``Field`` over a dropped column, a ``Mean`` over a column that is
now a string.

Only drifted sources move. A source the report found ``equal`` keeps its node,
so the subtrees above it rebuild to equal ops and their cache keys stay put.
The live schemas come from the report rather than from a second probe, so what
``check-sources`` said and what the rewrite rebuilt over are one snapshot, and
every question about which sources may be probed without writing to them stays
in ``catalog.drift``.

A pin is drift-exempt: a ``CacheTag`` is returned as recorded, whatever its
subtree would have become.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from xorq.catalog.drift import (
    LeafReport,
    iter_leaf_reports,
    leaf_con_name,
    unchecked_leaves,
)
from xorq.catalog.enums import LeafKind, Verdict
from xorq.catalog.inspection import BuildRecord, SourceLeaf
from xorq.common.exceptions import SchemaRefreshError
from xorq.common.utils.graph_utils import OPAQUE_SPECS, _opaque_lookup, to_node
from xorq.common.utils.node_utils import recreate, update_read_kwargs
from xorq.expr.relations import CachedNode, CacheTag, Read, RemoteTable, Tag, TeeNode
from xorq.ibis_yaml.enums import ReadKwarg
from xorq.vendor.ibis.common.graph import Node
from xorq.vendor.ibis.expr.schema import Schema


# Ops that store a schema their parent determines, and the field holding that
# parent. Rebuilt over a moved parent they take its schema, or they would
# advertise columns they no longer produce (`Tag`, `CachedNode`) or fail an
# integrity check over a source that merely gained a column (`TeeNode`).
SCHEMA_FOLLOWS_PARENT = {
    CachedNode: "parent",
    RemoteTable: "remote_expr",
    TeeNode: "parent",
    Tag: "parent",
}


def join_path(path: Any) -> str:
    """A read path spelled the way ``SourceLeaf.name`` spells it."""
    return ", ".join(map(str, path)) if isinstance(path, (list, tuple)) else str(path)


def leaf_key(leaf: SourceLeaf, record: BuildRecord) -> tuple:
    """What identifies ``leaf`` among the loaded expression's sources.

    Matched against ``op_key``. The recorded schema is part of it, so a node
    already rebuilt over its live schema can never match a second time.
    """
    return (str(leaf.kind), leaf_con_name(leaf, record), leaf.name, leaf.recorded)


def op_key(node: Node) -> tuple | None:
    """``node``'s ``leaf_key``, or ``None`` when it is not a source.

    Matched on the exact type name, because `CachedNode` and `RemoteTable` are
    `DatabaseTable` subclasses that are not sources. The profile name is not
    usable: it carries a session-local index, so a loaded source never spells
    the one its record does.
    """
    match type(node).__name__:
        case LeafKind.DATABASE_TABLE:
            namespace = node.namespace
            parts = (namespace.catalog, namespace.database, node.name)
            name = ".".join(part for part in parts if part)
        case LeafKind.READ:
            # `SourceLeaf`'s fallback, so a read without a `hash_path` still
            # spells the name its record does.
            path = dict(node.read_kwargs).get(ReadKwarg.hash_path) or node.name
            name = join_path(path)
        case _:
            return None
    return (type(node).__name__, node.source.name, name, node.schema)


def with_live_schema(node: Node, schema: Schema) -> Node:
    """``node`` carrying ``schema``.

    A read also records its schema in ``read_kwargs``, as an instruction the
    read method obeys; left stale, the node would advertise the live columns
    and then read the recorded ones. duckdb's per-column ``types`` override
    cannot be reproduced from a schema, so it is dropped.
    """
    if not isinstance(node, Read):
        return recreate(node, schema=schema)
    recorded = dict(node.read_kwargs)
    instructions = tuple(
        (key, schema)
        for key in (ReadKwarg.schema, ReadKwarg.columns)
        if key in recorded
    )
    read_kwargs = tuple(
        (key, value)
        for key, value in update_read_kwargs(node.read_kwargs, instructions)
        if key != ReadKwarg.types
    )
    return recreate(node, schema=schema, read_kwargs=read_kwargs)


def rebuild(node: Node, build: Callable[[], Node]) -> Node:
    """``build()``, with a failure named after ``node``.

    A ``SchemaRefreshError`` from deeper down passes through untouched, so the
    name that survives is the deepest op that could not be rebuilt.
    """
    try:
        return build()
    except SchemaRefreshError:
        raise
    except Exception as e:
        raise SchemaRefreshError(type(node).__name__, e) from e


def recreate_over(node: Node, overrides: dict) -> Node:
    """``node`` rebuilt with ``overrides``, its stored schema following its parent."""
    if (attr := _opaque_lookup(node, SCHEMA_FOLLOWS_PARENT)) is not None:
        parent = overrides.get(attr, getattr(node, attr))
        overrides = overrides | {"schema": to_node(parent).schema}
    return recreate(node, **overrides)


def refresh_schemas(expr: Any, live: Mapping[tuple, Schema]) -> Any:
    """``expr`` rebuilt over ``live``, a ``leaf_key`` -> live schema mapping.

    Bottom-up through ``Node.replace``, descending the opaque edges
    ``OPAQUE_SPECS`` names on its write side. A replacer that returned an
    untouched op as-is would discard the rebuilt children ``replace`` hands it,
    so every op a change reached is recreated from them.

    Every key in ``live`` has to match a source: one that matched nothing would
    leave its source on the recorded schema, and the result would look
    refreshed without being so. It raises instead, named after the leaf's kind.
    """
    if not live:
        return expr
    memo: dict[Node, Node] = {}
    matched: set[tuple] = set()

    def rewrite(node: Node) -> Node:
        if node not in memo:
            memo[node] = node.replace(replacer)
        return memo[node]

    def replacer(node: Node, kwargs: dict | None) -> Node:
        if isinstance(node, CacheTag):
            return node
        if (key := op_key(node)) is not None and key in live:
            matched.add(key)
            return rebuild(node, lambda: with_live_schema(node, live[key]))
        overrides = dict(kwargs or {})
        rebound = node
        if (spec := _opaque_lookup(node, OPAQUE_SPECS)) is not None:
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
        (kind, _, name, _) = unmatched[0]
        cause = LookupError(f"{name} matched no source of the loaded expression")
        raise SchemaRefreshError(kind, cause)
    return refreshed


def live_schemas(record: BuildRecord, reports: Iterable[LeafReport]) -> dict:
    """The ``refresh_schemas`` mapping for the leaves ``reports`` found changed.

    A source that could not be compared is not something to rebuild over, so
    every verdict but ``equal`` and ``changed`` raises, named after its kind.
    """
    live = {}
    for report in reports:
        match report.verdict:
            case Verdict.EQUAL:
                continue
            case Verdict.CHANGED:
                live[leaf_key(report.leaf, record)] = report.live
            case verdict:
                detail = f": {report.error}" if report.error else ""
                cause = LookupError(f"{report.leaf.name} is {verdict}{detail}")
                raise SchemaRefreshError(str(report.leaf.kind), cause)
    return live


def check_refreshable(record: BuildRecord) -> None:
    """Raise when ``record`` has an external source the sweep will not probe.

    ``iter_leaf_reports`` covers ``checkable_leaves`` only, so a source it
    leaves out -- a read with no registered inference, bound to an ingesting
    backend -- would keep its recorded schema while the refresh reports
    success. ``check-sources`` names such a leaf as unchecked; a refresh cannot
    stand behind a schema it never looked at, so it refuses instead.
    """
    if unchecked := unchecked_leaves(record):
        leaf = unchecked[0]
        cause = LookupError(f"{leaf.name} cannot be probed without writing to it")
        raise SchemaRefreshError(str(leaf.kind), cause)


def refresh_build(
    build_path: str | Path, con_cache: dict | None = None, **kwargs: Any
) -> Any:
    """Load the build at ``build_path`` and rebuild it over its drifted sources.

    The sweep runs first, on the guarded connections ``catalog.drift`` opens,
    and a source it cannot compare, or will not probe, stops the refresh
    before anything loads.
    """
    from xorq.ibis_yaml.compiler import load_expr  # noqa: PLC0415

    record = BuildRecord.from_build_dir(build_path)
    check_refreshable(record)
    live = live_schemas(record, iter_leaf_reports(record, con_cache))
    return refresh_schemas(load_expr(build_path, **kwargs), live)
