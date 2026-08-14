"""``Read`` and per-backend ``DatabaseTable`` normalizers, plus the
DatabaseTable dispatcher.

Each per-backend normalizer extracts file paths from the backend-specific
plan / DDL representation and stats them — restoring the
``ModificationTimeStrategy`` invalidation semantics that xorq_dasher 0.1.0's
ep_str-only DT rule loses.
"""

from __future__ import annotations

import contextvars
import functools
import pathlib
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

from xorq_dasher.rules.expr import (
    normalize_cached_node,
    normalize_databasetable,
    normalize_remote_table,
)

from xorq.common.constants import (
    DATAFUSION_BACKEND_NAMES,
    READ_IDENTITY_KEYS,
    REMOTE_SCHEMES,
)
from xorq.common.enums import BackendName
from xorq.common.utils.dasher._canonical import (
    normalize_memory_databasetable_canonical,
)
from xorq.common.utils.dasher._gap_rules import normalize_ibis_schema
from xorq.common.utils.dasher._opaque import (
    _MISSING,
    _rename_unbound_xorq,
    require_normalize_method,
)
from xorq.common.utils.dasher._paths import (
    _extract_datafusion_plan_paths,
    _extract_duckdb_file_paths,
    _normalize_path_stat,
    _stat_or_canonical,
)


if TYPE_CHECKING:
    from xorq.vendor.ibis.expr import operations as ops


# Per-outer-call memo for ``_databasetable_dispatcher``.  Cross-engine nested
# expressions cause the same underlying ``DatabaseTable`` to be normalized
# many times (``walk_nodes(DatabaseTable, op)`` descends through opaque
# sub-expressions, so each recursive ``_normalize_expr_xorq`` invocation
# returns the same deep DTs again).  Each call hits ``to_pyarrow_batches``
# on the underlying table — for an 8-level ``into_backend`` chain this is
# 1280 conversions of the same data.  The memo collapses it back to one
# per unique DT, per outer tokenize call.
_dt_normalize_memo: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "_xorq_dt_normalize_memo", default=None
)


def _normalize_read_xorq(read):
    """xorq-flavored Read normalizer.

    xorq stores the read path under the canonical ``hash_path`` key (defer_utils
    renames backend-specific kwargs), so the dasher 0.1.0 Read rule (which looks
    up ``path/paths/source/source_list``) does not match. This restores the
    legacy xorq behavior covering http(s), cloud, build-bundle relative, and
    local-filesystem paths.
    """

    read_kwargs = dict(read.read_kwargs)
    if "hash_path" not in read_kwargs:
        # path-less Read (e.g. an API-backed source): identity comes entirely
        # from the registered normalize_method, which receives the op itself
        return ("xorq.Read", read.schema, require_normalize_method(read)(read))
    path = read_kwargs["hash_path"]
    if path is None:
        raise ValueError(
            f"Read op {getattr(read, 'name', read)!r} has hash_path=None in "
            f"read_kwargs (keys: {sorted(read_kwargs)!r}); "
            f"normalize_filenames must run before tokenization."
        )
    match path:
        case list() | tuple() if len(path) == 1:
            path = str(path[0])
        case list() | tuple():
            tpls = tuple(
                _normalize_single_path(str(p), read_kwargs, read) for p in path
            )
            tpls += _read_extra_kwargs(read)
            return ("xorq.Read", read.schema, tpls)
        case str() | pathlib.Path():
            path = str(path)
        case _:
            raise NotImplementedError(f"Don't know how to deal with path {path!r}")
    tpls = (_normalize_single_path(path, read_kwargs, read),)
    tpls += _read_extra_kwargs(read)
    return ("xorq.Read", read.schema, tpls)


def _normalize_single_path(path, read_kwargs, read):
    """Normalize a single string path for Read tokenization."""

    if path.startswith(REMOTE_SCHEMES):
        stat_kwargs = {k: v for k, v in read_kwargs.items() if k != "hash_path"}
        return _normalize_path_stat(path, **stat_kwargs)
    elif not pathlib.Path(path).is_absolute() and path == read_kwargs.get("read_path"):
        return (("build-relative-path", path),)
    elif (p := pathlib.Path(path)).exists():
        return read.normalize_method(p)
    else:
        raise FileNotFoundError(f"local path does not exist: {path!r}")


def _read_extra_kwargs(read):
    return tuple((k, v) for k, v in read.read_kwargs if k in READ_IDENTITY_KEYS)


def _normalize_duckdb_databasetable_xorq(dt):
    """DuckDB DT normalizer with catalog-extract path canonicalization.

    Dasher 0.1.0's ``normalize_duckdb_file_databasetable`` returns the raw
    DDL string, which embeds the absolute path DuckDB sees — for tables
    rehydrated from a catalog zip, that path lives under a per-load
    ``xorq-catalog-<random>/`` tempdir and leaks into the token. Parse paths
    out, canonicalize, then stat-or-pass-through (see :func:`_stat_or_canonical`).
    """

    import sqlglot as sg  # noqa: PLC0415

    name = sg.table(dt.name, quoted=dt.source.compiler.quoted).sql(
        dialect=dt.source.name
    )
    ((_, plan),) = dt.source.raw_sql(f"EXPLAIN SELECT * FROM {name}").fetchall()
    lines = plan.split("\n")
    if len(lines) < 2:
        raise ValueError(f"unexpected EXPLAIN output for {dt.name!r}: {plan!r}")
    scan_line = lines[1]
    scan_match = re.match(r"\s*│\s*(\w+)\s*│\s*", scan_line)
    if scan_match is None:
        raise ValueError(
            f"unrecognized EXPLAIN scan line for {dt.name!r}: {scan_line!r}"
        )
    scan_kind = scan_match.group(1)
    if scan_kind in ("ARROW_SCAN", "PANDAS_SCAN"):
        return normalize_memory_databasetable_canonical(dt)
    if scan_kind in ("READ_PARQUET", "READ_CSV", "SEQ_SCAN"):
        sql_name = sg.exp.convert(dt.name).sql(dialect=dt.source.name)
        (sql_ddl,) = dt.source.con.sql(
            f"select sql from duckdb_views() where view_name = {sql_name} "
            f"UNION select sql from duckdb_tables() where table_name = {sql_name}"
        ).fetchone()
        paths = _extract_duckdb_file_paths(sql_ddl)
        if paths:
            file_metadata = tuple((p, _stat_or_canonical(p)) for p in sorted(paths))
            return (
                "ibis.DatabaseTable.duckdb.file",
                normalize_ibis_schema(dt.schema),
                file_metadata,
            )
        # Fallback to the raw-DDL form when we can't parse paths (preserves
        # dasher 0.1.0 behavior).
        return (
            "ibis.DatabaseTable.duckdb.file",
            normalize_ibis_schema(dt.schema),
            sql_ddl,
        )
    raise NotImplementedError(scan_line)


def _normalize_datafusion_databasetable_xorq(dt):
    """Datafusion DT normalizer that stats Parquet/CSV files for content sensitivity.

    Dasher 0.1.0's rule returns just ``(schema, ep_str)`` for parquet/csv-backed
    tables; ep_str captures the path but no mtime/size, so file edits don't
    invalidate ``ModificationTimeStrategy`` cache keys (the test in
    ``test_parquet_cache_storage``). Mirror the legacy xorq behavior: extract
    file paths from the plan and stat them.
    """

    table = dt.source.con.table(dt.name)
    ep_str = str(table.execution_plan())
    is_file = ep_str.startswith(("ParquetExec:", "CsvExec:")) or re.match(
        r"DataSourceExec:.+file_type=(csv|parquet)", ep_str
    )
    if is_file:
        paths = _extract_datafusion_plan_paths(ep_str)
        if paths:
            file_metadata = tuple((p, _stat_or_canonical(p)) for p in sorted(paths))
            return (
                "ibis.DatabaseTable.datafusion.file",
                normalize_ibis_schema(dt.schema),
                file_metadata,
            )
        raise ValueError(
            f"no parquet/csv paths extractable from execution plan: {ep_str!r}"
        )
    if ep_str.startswith(("MemoryExec:", "DataSourceExec:")):
        return normalize_memory_databasetable_canonical(dt)
    if "PyRecordBatchProviderExec" in ep_str:
        return (
            "ibis.DatabaseTable.datafusion.recordbatch",
            normalize_ibis_schema(dt.schema),
            dt.name,
        )
    if ep_str.startswith("EmptyExec"):
        raise ValueError("No data to cache")
    raise ValueError(f"unrecognized DataFusion execution plan: {ep_str!r}")


# BigQuery project/dataset identifiers are letters, digits, underscores and
# hyphens, plus the '.' and ':' of a legacy domain-scoped project id (e.g.
# `google.com:my-project`); anything else (notably a backtick) cannot appear in
# a valid namespace and would corrupt the backtick-quoted __TABLES__ reference
_BQ_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:\-]+")


def _bigquery_last_modified_query(namespace: ops.Namespace, table_name: str) -> str:
    """Build the ``__TABLES__.last_modified_time`` lookup for a DatabaseTable.

    ``table_name`` is compared as a GoogleSQL string literal, which escapes with
    a backslash (BigQuery has no ``''`` quote-doubling), so both ``\\`` and ``'``
    are backslash-escaped — backslash first, or an escaped quote would be double
    escaped. The dataset path is a backtick-quoted identifier, so each of its
    components is validated against the BigQuery identifier grammar (a name with
    a backtick would otherwise break out of the quoting).
    """
    components = tuple(part for part in (namespace.catalog, namespace.database) if part)
    for part in components:
        if not _BQ_IDENTIFIER.fullmatch(part):
            raise ValueError(f"invalid BigQuery identifier in namespace: {part!r}")
    dataset = ".".join(components)
    table_id = table_name.replace("\\", "\\\\").replace("'", "\\'")
    return (
        "SELECT last_modified_time "
        f"FROM `{dataset}.__TABLES__` "
        f"WHERE table_id = '{table_id}'"
    )


def _normalize_bigquery_databasetable_xorq(dt: ops.DatabaseTable) -> tuple:
    """BigQuery DT normalizer keyed on ``__TABLES__.last_modified_time``.

    xorq_dasher 0.1.0's ``normalize_bigquery_databasetable`` unpacks the
    result with ``((last_modified_time,),) = ...to_dataframe()``, which
    iterates the DataFrame by *column label* rather than by row and so raises
    ``ValueError: too many values to unpack`` for every BigQuery table. Read
    the scalar out of the frame directly instead, and qualify ``__TABLES__``
    with the catalog so tables outside the billing project resolve.
    """
    import pandas as pd  # noqa: PLC0415
    from google.api_core.exceptions import GoogleAPICallError  # noqa: PLC0415

    query = _bigquery_last_modified_query(dt.namespace, dt.name)
    base = (
        "ibis.DatabaseTable.bigquery",
        dt.name,
        normalize_ibis_schema(dt.schema),
        dt.namespace,
    )
    try:
        df = dt.source.raw_sql(query).to_dataframe()
    except GoogleAPICallError:
        # anonymous session tables (e.g. read_parquet) live in a dataset whose
        # __TABLES__ isn't necessarily queryable, so the lookup can *raise*
        # (NotFound/BadRequest) rather than return an empty frame; fall back to
        # a stable structural token in that case too
        return base
    if df.empty:
        # a queryable __TABLES__ that simply has no row for this table_id
        # (also possible for session/temp tables); same structural fallback —
        # the namespace already makes the token distinct
        return base
    (last_modified_time,) = df["last_modified_time"]
    if pd.isna(last_modified_time):
        # external/federated tables report a NULL last_modified_time
        return base
    # a numpy scalar has no dasher normalizer; hand back a native int
    return (*base, int(last_modified_time))


def _databasetable_dispatcher(dt: ops.DatabaseTable) -> tuple:
    """Dispatch DatabaseTable subclasses to their specific normalizers.

    xorq_dasher 0.1.0's normalize_databasetable does not handle the
    ``xorq_datafusion`` backend name (only ``xorq``) and its DatabaseTable
    rule outranks the more-specific Read/CachedNode/RemoteTable rules in
    MRO-with-earliest-match-wins lookup. This wrapper restores the
    most-specific-wins behavior xorq depends on.

    Memoized per outer call via :data:`_dt_normalize_memo` — see the
    contextvar's docstring for the perf rationale.  Result is a pure
    function of ``dt`` (the per-subclass normalizers don't consult the
    active hasher), so the memo doesn't need to key on it.
    """
    memo = _dt_normalize_memo.get()
    if memo is not None and dt in memo:
        return memo[dt]
    result = _dispatch_databasetable(dt)
    if memo is not None:
        memo[dt] = result
    return result


def normalize_flight_expr(dt: ops.DatabaseTable) -> tuple:
    """Identity of a ``FlightExpr``, deliberately excluding ``dt.name``.

    A ``FlightExpr`` carries input_expr / make_connection that the plain
    datafusion path would silently flatten away. Inlines the dasher 0.1.0 logic
    but uses ``_rename_unbound_xorq`` (whose op.replace callback signs
    ``(node, _kwargs)`` correctly — dasher 0.1.0's ``_rename_unbound`` uses
    ``**kwargs`` and crashes recreating ops with required positional fields like
    ``Field``).

    ``dt.name`` is omitted because it defaults to ``gen_name()`` — a fresh uuid4
    per process — so folding it in makes the token non-reproducible (gh-2229;
    see :func:`view_rules`).
    """
    return (
        "xorq.FlightExpr",
        dt.input_expr,
        _rename_unbound_xorq(dt.unbound_expr.op()).to_expr(),
        dt.make_connection,
    )


def normalize_flight_udxf(dt: ops.DatabaseTable) -> tuple:
    """Identity of a ``FlightUDXF``, deliberately excluding ``dt.name``.

    The qualname is what distinguishes two exchangers that inherit rather than
    override ``exchange_f``, and it is deterministic (``name or
    process_df.__name__``, never a ``gen_name()``). ``dt.udxf`` is already the
    exchanger *class* — do not write ``type(dt.udxf).__qualname__``, which reads
    the metaclass and yields ``"ABCMeta"`` for every UDXF
    (``test_flight_udxf_qualname_is_the_class_not_the_metaclass`` pins this).

    See :func:`normalize_flight_expr` for why ``dt.name`` is excluded.
    """
    return (
        "xorq.FlightUDXF",
        dt.input_expr,
        dt.udxf.__qualname__,
        getattr(dt.udxf, "exchange_f", _MISSING),
        dt.make_connection,
    )


class ViewRule(NamedTuple):
    """One row of :func:`view_rules`: an op type and its normalizer per regime."""

    op_type: type
    normalizer: Callable
    snapshot_normalizer: Callable


@functools.cache
def view_rules() -> tuple[ViewRule, ...]:
    """The single source of truth for ``DatabaseTable``-subclass normalization.

    Two dispatch tables normalize these ops — the global one
    (:func:`_dispatch_databasetable`) and the snapshot one
    (``SnapshotStrategy.normalize_databasetable``) — because dasher's
    MRO-with-earliest-match-wins lookup would otherwise pick the broader
    ``DatabaseTable`` rule over the more specific subclasses.  They used to be
    hand-mirrored ``match`` statements, and they drifted: the snapshot copy
    omitted ``FlightExpr``/``FlightUDXF`` and its ``case _`` fallback folded
    their ``gen_name()`` uuid4 into the key, making ``xorq build``
    non-reproducible and every ``SnapshotStrategy`` cache miss per process
    (gh-2229).  The identical bug had already been fixed once in the dask era
    (gh-610) and returned with the dasher rewrite, so the table is the fix:
    adding a row serves both regimes at once (the ADR-0016 pattern —
    table-driven dispatch with registration tripwires).

    Two columns because the regimes legitimately differ for exactly one op:
    ``Read`` takes stat-based identity globally and path-only identity under
    snapshot.  Every other row is deliberately the same callable in both
    columns, which is the invariant that drift used to break silently.

    Order is significant — :func:`lookup_view_normalizer` takes the first
    ``isinstance`` match, so a subtype must precede its supertype.

    Resolved lazily (and cached) because the snapshot column lives in
    ``xorq.caching.strategy``, which imports this module.
    """
    from xorq.caching.strategy import snapshot_normalize_read  # noqa: PLC0415
    from xorq.expr.relations import (  # noqa: PLC0415
        CachedNode,
        FlightExpr,
        FlightUDXF,
        Read,
        RemoteTable,
    )

    return (
        ViewRule(Read, _normalize_read_xorq, snapshot_normalize_read),
        ViewRule(CachedNode, normalize_cached_node, normalize_cached_node),
        ViewRule(RemoteTable, normalize_remote_table, normalize_remote_table),
        ViewRule(FlightExpr, normalize_flight_expr, normalize_flight_expr),
        ViewRule(FlightUDXF, normalize_flight_udxf, normalize_flight_udxf),
    )


def lookup_view_normalizer(dt: ops.DatabaseTable, *, snapshot: bool) -> Callable | None:
    """Return the normalizer for ``dt``, or ``None`` to use the caller's fallback.

    Raises ``NotImplementedError`` for an unhandled ``DatabaseTableView``: a
    name-folding fallback is right for a genuine backend table (there ``name``
    *is* the identity) and wrong for a view, whose ``name`` defaults to a
    per-process ``gen_name()`` uuid4 (gh-2229; see :func:`view_rules`).  Both
    dispatch tables route through here so the guard covers both.

    Not made redundant by ``test_view_rules.py``'s static exhaustiveness check:
    that check reads ``DatabaseTableView.__subclasses__()``, which only sees
    imported modules, and xorq imports backends lazily.
    """
    for rule in view_rules():
        if isinstance(dt, rule.op_type):
            return rule.snapshot_normalizer if snapshot else rule.normalizer
    from xorq.expr.relations import DatabaseTableView  # noqa: PLC0415

    if isinstance(dt, DatabaseTableView):
        raise NotImplementedError(
            f"{type(dt).__name__} is a DatabaseTableView with no normalizer; "
            "add a row to xorq.common.utils.dasher._relations.view_rules "
            "rather than letting a fallback fold its generated name into the key"
        )
    return None


def unhandled_view_op_types() -> tuple[type, ...]:
    """``DatabaseTableView`` subclasses (recursively) with no :func:`view_rules` row.

    Only sees subclasses whose defining module has been imported; see the
    caveat on :func:`lookup_view_normalizer`.
    """
    from xorq.expr.relations import DatabaseTableView  # noqa: PLC0415

    handled = tuple(rule.op_type for rule in view_rules())

    def descendants(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from descendants(sub)

    return tuple(
        cls
        for cls in dict.fromkeys(descendants(DatabaseTableView))
        if not issubclass(cls, handled)
    )


def _dispatch_databasetable(dt: ops.DatabaseTable) -> tuple:
    # DatabaseTable-subclass dispatch is shared with
    # ``SnapshotStrategy.normalize_databasetable`` via ``view_rules`` so the two
    # regimes cannot drift on which ops they cover (gh-2229); returning None
    # here means "not a view, fall through to the per-backend chain below".
    normalizer = lookup_view_normalizer(dt, snapshot=False)
    if normalizer is not None:
        return normalizer(dt)
    # For datafusion-backed file tables, dasher's normalize_datafusion_
    # databasetable stops at ep_str — which captures the path but no stat —
    # so file edits don't invalidate the cache key. _normalize_datafusion_
    # databasetable_xorq stats the underlying files to restore mtime sensitivity.
    if dt.source.name in DATAFUSION_BACKEND_NAMES:
        return _normalize_datafusion_databasetable_xorq(dt)
    if dt.source.name == BackendName.DUCKDB:
        return _normalize_duckdb_databasetable_xorq(dt)
    # xorq_dasher 0.1.0's bigquery normalizer unpacks its result frame by
    # column label and crashes on every table; use the fixed xorq version.
    if dt.source.name == BackendName.BIGQUERY:
        return _normalize_bigquery_databasetable_xorq(dt)
    # pandas-backend tables and in-memory sqlite are memory-resident:
    # xorq_dasher's dispatch hashes the IPC bytes of their
    # ``to_pyarrow_batches()`` stream, which is pyarrow-version-coupled
    # (issue #2191) — route them to the canonical form instead.
    if dt.source.name == BackendName.PANDAS:
        return normalize_memory_databasetable_canonical(dt)
    if dt.source.name == BackendName.SQLITE and dt.source.is_in_memory():
        return normalize_memory_databasetable_canonical(dt)
    # All remaining backends fall through to ``xorq_dasher``
    # ``normalize_databasetable`` (bigquery is handled above and never reaches
    # here), which is itself a per-backend dispatch table postgres calls
    # ``get_postgres_n_reltuples``, snowflake calls
    # ``get_snowflake_last_modification_time``, pyiceberg calls
    # ``get_iceberg_snapshots_ids``, file-backed sqlite calls
    # ``get_sqlite_stats`` (memory-backed is intercepted above),
    # trino/gizmosql fall back to ``normalize_remote_databasetable``.
    # Data-sensitivity is preserved upstream, not blindly flattened to
    # schema+name, see xorq_dasher/rules/expr.py::normalize_databasetable.
    result = normalize_databasetable(dt)
    if isinstance(result, tuple) and result[:1] == ("ibis.MemoryDatabaseTable",):
        # Safety net: dasher resolved this table to its memory rule, whose
        # token hashes the to_pyarrow_batches() IPC stream — the
        # pyarrow-version-coupled form (#2191). Known memory backends are
        # intercepted above before dasher runs; this catches renamed or
        # future ones (dasher still maps a backend *named* "xorq" to that
        # rule) at the cost of one redundant materialization.
        return normalize_memory_databasetable_canonical(dt)
    return result


__all__ = [
    "ViewRule",
    "_databasetable_dispatcher",
    "_normalize_bigquery_databasetable_xorq",
    "_normalize_datafusion_databasetable_xorq",
    "_normalize_duckdb_databasetable_xorq",
    "_normalize_read_xorq",
    "lookup_view_normalizer",
    "normalize_flight_expr",
    "normalize_flight_udxf",
    "unhandled_view_op_types",
    "view_rules",
]
