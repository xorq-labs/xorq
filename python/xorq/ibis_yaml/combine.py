"""Combine build artifacts (or already-loaded exprs) via join/union.

Shared core for the ``xorq join``/``xorq union`` and ``xorq catalog
join``/``xorq catalog union`` CLI commands: `join_exprs`/`union_exprs`
operate on already-resolved `Expr` objects (used directly by the catalog
tier, whose entries resolve to exprs via `CatalogEntry.load_expr`), while
`join_builds`/`union_builds` additionally load/build from plain build-path
strings (used by the top-level tier).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import click


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from xorq.api import Expr


@contextmanager
def combine_errors() -> "Iterator[None]":
    """Map `join_exprs`/`union_exprs`'s exceptions to clean CLI errors.

    Shared by both CLI tiers (`xorq join`/`union` and `xorq catalog
    join`/`union`) so this mapping can't drift out of sync between call
    sites the way it already had (catalog `join` was missing the
    `RelationError` arm `union` had). Covers the exceptions actually
    observed escaping `join_exprs`/`union_exprs` uncaught:
    - `ValueError`/`XorqTypeError`: bad user input (a malformed predicate
      spec, a typo'd `--on`/`--left-on`/`--right-on` column -- the latter
      raises `XorqTypeError`, a `TypeError` subclass, not `ValueError`).
    - `RelationError`/`IntegrityError`: the join/union is well-formed but
      the tables don't fit together (schema mismatch, or an lname/rname
      collision left unresolved).
    - `KeyError`: an `--lname`/`--rname` template referencing a field that
      doesn't exist (e.g. `--rname "{bogus}"` raises `KeyError('bogus')`
      from the bare `str.format` call inside ibis's join disambiguation).
    """
    from xorq.common.exceptions import (  # noqa: PLC0415
        IntegrityError,
        RelationError,
        XorqTypeError,
    )

    try:
        yield
    except (ValueError, XorqTypeError) as e:
        raise click.BadParameter(str(e)) from None
    except (RelationError, IntegrityError) as e:
        raise click.ClickException(str(e)) from e
    except KeyError as e:
        raise click.BadParameter(
            f"unknown placeholder {e} in --lname/--rname template"
        ) from None


def _split_columns(value: str) -> tuple[str, ...]:
    return tuple(c.strip() for c in value.split(",") if c.strip())


def _build_join_predicates(
    on: str | None,
    left_on: str | None,
    right_on: str | None,
    how: str,
) -> tuple[str | tuple[str, str], ...]:
    """Turn CLI-style --on/--left-on/--right-on into ibis's `predicates` shape."""
    if on is not None and (left_on is not None or right_on is not None):
        raise ValueError("--on is mutually exclusive with --left-on/--right-on")
    if (left_on is None) != (right_on is None):
        raise ValueError("--left-on and --right-on must be given together")

    if how == "cross":
        if on is not None or left_on is not None:
            raise ValueError(
                "--how=cross takes no predicates; omit --on/--left-on/--right-on"
            )
        return ()

    if on is not None:
        columns = _split_columns(on)
        if not columns:
            raise ValueError("--on must name at least one non-empty column")
        return columns

    if left_on is not None:
        left_cols = _split_columns(left_on)
        right_cols = _split_columns(right_on)
        if not left_cols or not right_cols:
            raise ValueError(
                "--left-on/--right-on must each name at least one non-empty column"
            )
        if len(left_cols) != len(right_cols):
            raise ValueError(
                f"--left-on has {len(left_cols)} column(s) but "
                f"--right-on has {len(right_cols)}"
            )
        return tuple(zip(left_cols, right_cols))

    raise ValueError("must specify --on or --left-on/--right-on, unless --how=cross")


def _rebind_same_profile_sources(*exprs: "Expr") -> tuple["Expr", ...]:
    """Collapse same-profile backend connections across *exprs* onto one object.

    Independently-loaded exprs being joined/unioned ("multi-root" combining)
    can carry several backend connection objects that share the same
    `Profile` content (con_name + kwargs, ignoring the session-local `idx`)
    -- e.g. two `xo.connect()` calls, or two `load_expr()` calls of the same
    source, each of which always constructs a fresh connection object
    (`Profile.get_con()` has no caching). Rewriting every reference in a
    same-profile group onto one connection object is a pure graph rewrite
    (the same one `replace_sources` already does for `normalize_profiles`,
    just deduping instead of reindexing) and safe for:
    - A plain file/table `Read` (e.g. `deferred_read_parquet`): same profile
      means the same physical resource.
    - A `RemoteTable` (`.into_backend(...)`): lazy by construction --
      `into_backend` just stores `(source=con, remote_expr)`, no
      registration happens until `read_record_batches` runs at execution
      time -- so repointing `.source` onto a same-profile connection is as
      safe as for a `Read`. Confirmed live with two `xo.connect()` hub
      connections and a `.into_backend()` on one side: rewriting `.source`
      and executing produces the correct joined result. (An earlier version
      of this excluded `RemoteTable` after hitting `AttributeError: 'Backend'
      object has no attribute 'read_record_batches'` -- that backend
      (`xo.datafusion.connect()`, not the `xo.connect()` hub) doesn't
      implement `read_record_batches` at all and would fail identically
      with zero rebinding involved; `con_name` is part of the profile
      content-key, so a backend lacking a capability is never grouped with
      one that has it in the first place.)

    Same profile does *not* imply interchangeable for anything else that
    carries a backend:
    - An in-memory `DatabaseTable` (`.create_table(...)`): two connections
      with identical params are still two independent sessions, each with
      its own registered data. `replace_sources(transfer_tables=False)`
      already refuses to rewrite these.
    - `CachedNode`, `FlightExpr`/`FlightUDXF`, or `TeeNode`: each may tie a
      node to state living only on the specific connection instance that
      produced it, unverified either way -- left excluded conservatively;
      rewriting `.source` there is not blocked by `replace_sources` itself
      (it only guards `DatabaseTable`).

    So a backend is only rebind-eligible if *every* node referencing it
    (across all of *exprs*) is a `Read` or `RemoteTable`; a backend touched
    by any other node type is left alone entirely, as both a merge source
    and a merge target. `replace_sources` failing for an eligible group
    (e.g. a `DatabaseTable` sharing that backend after all) is caught per
    expr and treated as "can't rebind that one", falling through to the
    standard multi-backend error rather than leaking that internal
    ValueError.
    """
    import xorq.expr.relations as rel  # noqa: PLC0415
    from xorq.common.utils.dasher import tokenize  # noqa: PLC0415
    from xorq.common.utils.graph_utils import (  # noqa: PLC0415
        BACKEND_LEAF_NODE_TYPES,
        find_all_sources,
        replace_sources,
        walk_nodes,
    )

    # Walk the same node-type list find_all_sources uses (not a hand-copied
    # subset) so a future backend-bearing leaf type is unsafe by default --
    # added there, it's excluded here automatically -- rather than silently
    # skipping this exclusion check until someone remembers to update both.
    safe_node_types = (rel.Read, rel.RemoteTable)
    unsafe_ids = {
        id(source)
        for expr in exprs
        for node in walk_nodes(BACKEND_LEAF_NODE_TYPES, expr)
        if not isinstance(node, safe_node_types)
        for source in (
            node.writer.cons if isinstance(node, rel.TeeNode) else (node.source,)
        )
    }

    backends = [
        b for expr in exprs for b in find_all_sources(expr) if id(b) not in unsafe_ids
    ]
    if len(backends) < 2:
        return exprs

    def content_key(backend):
        profile = backend._profile  # xorq-style: disable=protected-access
        return tokenize({k: v for k, v in profile.as_dict().items() if k != "idx"})

    groups: dict[str, list] = {}
    for backend in backends:
        groups.setdefault(content_key(backend), []).append(backend)

    source_mapping = {
        id(duplicate): canonical
        for canonical, *duplicates in groups.values()
        for duplicate in duplicates
    }
    if not source_mapping:
        return exprs

    def rebind_one(expr):
        try:
            return replace_sources(source_mapping, expr)
        except ValueError:
            return expr

    return tuple(rebind_one(expr) for expr in exprs)


def _reconcile_backends(
    *exprs: "Expr", rebind_backends: bool
) -> tuple["Expr", ...]:
    """Require *exprs* to jointly resolve to a single backend, rebinding
    same-profile connections onto one object first (unless
    `rebind_backends=False`).

    Raises `ValueError` if, after rebinding, more than one distinct backend
    remains across *exprs* -- a genuine multi-engine combination this does
    not attempt to solve automatically (see `_rebind_same_profile_sources`).
    Runs before the exprs are combined so a schema mismatch in `Table.join`/
    `Table.union` doesn't mask a backend mismatch (or vice versa).
    """
    if rebind_backends:
        exprs = _rebind_same_profile_sources(*exprs)

    seen: dict[int, object] = {}
    for expr in exprs:
        backends, _ = expr._find_backends()  # xorq-style: disable=protected-access
        for backend in backends:
            seen.setdefault(id(backend), backend)
    if len(seen) > 1:
        named = ", ".join(sorted(type(b).__name__ for b in seen.values()))
        raise ValueError(
            f"cannot combine exprs bound to {len(seen)} different backend "
            f"connections ({named}); bridge them onto one connection yourself "
            "with `.into_backend(...)` first"
        )
    return exprs


def join_exprs(
    left: "Expr",
    right: "Expr",
    *,
    on: str | None = None,
    left_on: str | None = None,
    right_on: str | None = None,
    how: str = "inner",
    lname: str = "",
    rname: str = "{name}_right",
    rebind_backends: bool = True,
) -> "Expr":
    """Join two already-resolved exprs; a thin wrapper around `Table.join`."""
    left, right = _reconcile_backends(left, right, rebind_backends=rebind_backends)
    predicates = _build_join_predicates(on, left_on, right_on, how)
    joined = left.join(right, predicates, how=how, lname=lname, rname=rname)
    # `Table.join` returns a lazy, "unfinished" `Join` chain object: its
    # `lname`/`rname` name-collision check lives in a side-channel Python
    # attribute (`_collisions`), not the op graph, and only fires when a
    # `finished`-wrapped method (e.g. `.execute()`) is called on it directly.
    # `build_expr`/`load_expr` don't go through one of those, so an
    # unresolved collision would silently round-trip through YAML with the
    # colliding column dropped -- force resolution now so it raises here,
    # every time, instead of only for callers who happen to `.execute()` the
    # object returned from this call before doing anything else with it.
    return joined._finish()  # xorq-style: disable=protected-access


def union_exprs(
    *exprs: "Expr", distinct: bool = False, rebind_backends: bool = True
) -> "Expr":
    """Union two-or-more already-resolved exprs; a thin wrapper around `Table.union`."""
    if len(exprs) < 2:
        raise ValueError("union requires at least 2 sources")
    exprs = _reconcile_backends(*exprs, rebind_backends=rebind_backends)
    first, *rest = exprs
    return first.union(*rest, distinct=distinct)


def join_builds(
    left_path: str | Path,
    right_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    builds_dir: str | Path = "builds",
    relocate_reads: bool = True,
    on: str | None = None,
    left_on: str | None = None,
    right_on: str | None = None,
    how: str = "inner",
    lname: str = "",
    rname: str = "{name}_right",
    rebind_backends: bool = True,
) -> Path:
    """Load two build artifacts, join them, and write the result as a new build."""
    from xorq.ibis_yaml.compiler import build_expr, load_expr  # noqa: PLC0415

    # Shared so a same-profile connection in both builds is one object from
    # the moment each is loaded, not two independently-constructed ones that
    # `join_exprs`'s rebind then has to reconcile after the fact. Gated on
    # rebind_backends -- same policy toggle, applied proactively here instead
    # of reactively in `join_exprs` -- so `--no-rebind-backends` still gets
    # the fully-isolated-connections behavior it promises.
    con_cache: dict | None = {} if rebind_backends else None
    left = load_expr(left_path, cache_dir=cache_dir, con_cache=con_cache)
    right = load_expr(right_path, cache_dir=cache_dir, con_cache=con_cache)
    expr = join_exprs(
        left, right, on=on, left_on=left_on, right_on=right_on, how=how,
        lname=lname, rname=rname, rebind_backends=rebind_backends,
    )
    return build_expr(
        expr, builds_dir=builds_dir, cache_dir=cache_dir, relocate_reads=relocate_reads
    )


def union_builds(
    *paths: str | Path,
    cache_dir: str | Path | None = None,
    builds_dir: str | Path = "builds",
    relocate_reads: bool = True,
    distinct: bool = False,
    rebind_backends: bool = True,
) -> Path:
    """Load two-or-more build artifacts, union them, and write the result as a new build."""
    from xorq.ibis_yaml.compiler import build_expr, load_expr  # noqa: PLC0415

    con_cache: dict | None = {} if rebind_backends else None
    exprs: Sequence[Expr] = [
        load_expr(path, cache_dir=cache_dir, con_cache=con_cache) for path in paths
    ]
    expr = union_exprs(*exprs, distinct=distinct, rebind_backends=rebind_backends)
    return build_expr(
        expr, builds_dir=builds_dir, cache_dir=cache_dir, relocate_reads=relocate_reads
    )


def _read_build_library_version(build_path: str | Path) -> str:
    from xorq.ibis_yaml.enums import DumpFiles  # noqa: PLC0415

    metadata_path = Path(build_path) / DumpFiles.build_metadata
    metadata = json.loads(metadata_path.read_text())
    return metadata["current_library_version"]


def assert_matching_library_versions(*build_paths: str | Path) -> None:
    """Raise `BuildVersionMismatchError` if the builds pin different xorq versions."""
    from xorq.common.exceptions import BuildVersionMismatchError  # noqa: PLC0415

    versions = {str(p): _read_build_library_version(p) for p in build_paths}
    if len(set(versions.values())) > 1:
        detail = ", ".join(f"{p}={v}" for p, v in versions.items())
        raise BuildVersionMismatchError(
            f"builds recorded different xorq library versions: {detail}. "
            "Pass --ignore-library-version-mismatch to proceed anyway."
        )
