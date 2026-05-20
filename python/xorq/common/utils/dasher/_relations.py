"""``Read`` and per-backend ``DatabaseTable`` normalizers, plus the
DatabaseTable dispatcher.

Each per-backend normalizer extracts file paths from the backend-specific
plan / DDL representation and stats them — restoring the
``ModificationTimeStrategy`` invalidation semantics that xorq_dasher 0.1.0's
ep_str-only DT rule loses.
"""

from __future__ import annotations

import contextvars

from xorq.common.utils.dasher._opaque import _rename_unbound_xorq
from xorq.common.utils.dasher._paths import (
    _extract_datafusion_plan_paths,
    _extract_duckdb_file_paths,
    _normalize_path_stat,
    _stat_or_canonical,
)


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
    import pathlib  # noqa: PLC0415

    read_kwargs = dict(read.read_kwargs)
    path = read_kwargs.get("hash_path")
    if path is None:
        raise ValueError(
            f"Read op {getattr(read, 'name', read)!r} has no 'hash_path' in "
            f"read_kwargs (keys: {sorted(read_kwargs)!r}); "
            f"normalize_filenames must run before tokenization."
        )
    if isinstance(path, (list, tuple)) and len(path) == 1:
        path = path[0]

    def _one(p):
        s = str(p)
        if s.startswith(("http://", "https://", "s3://", "gs://", "gcs://")):
            stat_kwargs = {k: v for k, v in read_kwargs.items() if k != "hash_path"}
            return _normalize_path_stat(s, **stat_kwargs)
        if not pathlib.Path(s).is_absolute() and s == read_kwargs.get("read_path"):
            # Build-bundled Read: relative read_path is already a content hash.
            return (("build-relative-path", s),)
        pp = pathlib.Path(s)
        if pp.exists():
            return read.normalize_method(pp)
        raise NotImplementedError(f'Don\'t know how to deal with path "{p}"')

    if isinstance(path, (list, tuple)):
        tpls = tuple(t for p in path for t in _one(p))
    elif isinstance(path, (str, pathlib.Path)):
        tpls = _one(path)
    else:
        raise NotImplementedError(f'Don\'t know how to deal with path "{path}"')
    tpls += tuple(
        (k, v) for k, v in read.read_kwargs if k in ("mode", "schema", "temporary")
    )
    return ("xorq.Read", read.schema, tpls)


def _normalize_duckdb_databasetable_xorq(dt):
    """DuckDB DT normalizer with catalog-extract path canonicalization.

    Dasher 0.1.0's ``normalize_duckdb_file_databasetable`` returns the raw
    DDL string, which embeds the absolute path DuckDB sees — for tables
    rehydrated from a catalog zip, that path lives under a per-load
    ``xorq-catalog-<random>/`` tempdir and leaks into the token. Parse paths
    out, canonicalize, then stat-or-pass-through (see :func:`_stat_or_canonical`).
    """
    import re  # noqa: PLC0415

    import sqlglot as sg  # noqa: PLC0415
    from xorq_dasher.rules.expr import (  # noqa: PLC0415
        normalize_memory_databasetable,
    )

    name = sg.table(dt.name, quoted=dt.source.compiler.quoted).sql(
        dialect=dt.source.name
    )
    ((_, plan),) = dt.source.raw_sql(f"EXPLAIN SELECT * FROM {name}").fetchall()
    lines = plan.split("\n")
    if len(lines) < 2:
        raise ValueError(f"unexpected EXPLAIN output for {dt.name!r}: {plan!r}")
    scan_line = lines[1]
    m = re.match(r"\s*│\s*(\w+)\s*│\s*", scan_line)
    if m is None:
        raise ValueError(f"unrecognized EXPLAIN scan line for {dt.name!r}: {scan_line!r}")
    scan_kind = m.group(1)
    if scan_kind in ("ARROW_SCAN", "PANDAS_SCAN"):
        return normalize_memory_databasetable(dt)
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
                dt.schema.to_pandas(),
                file_metadata,
            )
        # Fallback to the raw-DDL form when we can't parse paths (preserves
        # dasher 0.1.0 behavior).
        return ("ibis.DatabaseTable.duckdb.file", dt.schema.to_pandas(), sql_ddl)
    raise NotImplementedError(scan_line)


def _normalize_datafusion_databasetable_xorq(dt):
    """Datafusion DT normalizer that stats Parquet/CSV files for content sensitivity.

    Dasher 0.1.0's rule returns just ``(schema, ep_str)`` for parquet/csv-backed
    tables; ep_str captures the path but no mtime/size, so file edits don't
    invalidate ``ModificationTimeStrategy`` cache keys (the test in
    ``test_parquet_cache_storage``). Mirror the legacy xorq behavior: extract
    file paths from the plan and stat them.
    """
    import re  # noqa: PLC0415

    from xorq_dasher.rules.expr import (  # noqa: PLC0415
        normalize_memory_databasetable,
    )

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
                dt.schema.to_pandas(),
                file_metadata,
            )
        raise ValueError(
            f"no parquet/csv paths extractable from execution plan: {ep_str!r}"
        )
    if ep_str.startswith(("MemoryExec:", "DataSourceExec:")):
        return normalize_memory_databasetable(dt)
    if "PyRecordBatchProviderExec" in ep_str:
        return (
            "ibis.DatabaseTable.datafusion.recordbatch",
            dt.schema.to_pandas(),
            dt.name,
        )
    if ep_str.startswith("EmptyExec"):
        raise ValueError("No data to cache")
    raise ValueError(f"unrecognized DataFusion execution plan: {ep_str!r}")


def _databasetable_dispatcher(dt):
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
    is_outer = memo is None
    if is_outer:
        memo = {}
        reset_token = _dt_normalize_memo.set(memo)
    try:
        if dt in memo:
            return memo[dt]
        result = _dispatch_databasetable(dt)
        memo[dt] = result
        return result
    finally:
        if is_outer:
            _dt_normalize_memo.reset(reset_token)


def _dispatch_databasetable(dt):
    from xorq_dasher.rules.expr import (  # noqa: PLC0415
        normalize_cached_node,
        normalize_databasetable,
        normalize_remote_table,
    )

    from xorq.expr.relations import (  # noqa: PLC0415
        CachedNode,
        FlightExpr,
        FlightUDXF,
        Read,
        RemoteTable,
    )

    match dt:
        case Read():
            return _normalize_read_xorq(dt)
        case CachedNode():
            return normalize_cached_node(dt)
        case RemoteTable():
            return normalize_remote_table(dt)
        case FlightExpr():
            # FlightExpr/FlightUDXF carry input_expr / make_connection that the
            # plain datafusion path would silently flatten away. Inline the
            # dasher 0.1.0 logic but use ``_rename_unbound_xorq`` (whose
            # op.replace callback signs ``(node, _kwargs)`` correctly — dasher
            # 0.1.0's ``_rename_unbound`` uses ``**kwargs`` and crashes
            # recreating ops with required positional fields like ``Field``).
            return (
                "xorq.FlightExpr",
                dt.input_expr,
                _rename_unbound_xorq(dt.unbound_expr.op()).to_expr(),
                dt.make_connection,
            )
        case FlightUDXF():
            # ``type(dt.udxf).__qualname__`` distinguishes UDXF classes even
            # when ``exchange_f`` is absent or shared — bare
            # ``getattr(..., None)`` would otherwise collapse two distinct
            # UDXFs (both missing ``exchange_f``) onto the same token.
            return (
                "xorq.FlightUDXF",
                dt.input_expr,
                type(dt.udxf).__qualname__,
                getattr(dt.udxf, "exchange_f", None),
                dt.make_connection,
            )
    # For datafusion-backed file tables, dasher's normalize_datafusion_
    # databasetable stops at ep_str — which captures the path but no stat —
    # so file edits don't invalidate the cache key. _normalize_datafusion_
    # databasetable_xorq stats the underlying files to restore mtime sensitivity.
    if dt.source.name in ("datafusion", "xorq_datafusion"):
        return _normalize_datafusion_databasetable_xorq(dt)
    if dt.source.name == "duckdb":
        return _normalize_duckdb_databasetable_xorq(dt)
    # All other backends fall through to ``xorq_dasher`` ``normalize_databasetable``,
    # which is itself a per-backend dispatch table postgres calls
    # ``get_postgres_n_reltuples``, snowflake calls
    # ``get_snowflake_last_modification_time``, bigquery queries
    # ``__TABLES__.last_modified_time``, pyiceberg calls
    # ``get_iceberg_snapshots_ids``, sqlite calls ``get_sqlite_stats``,
    # trino/gizmosql fall back to ``normalize_remote_databasetable``.
    # Data-sensitivity is preserved upstream, not blindly flattened to
    # schema+name, see xorq_dasher/rules/expr.py::normalize_databasetable.
    return normalize_databasetable(dt)


__all__ = [
    "_databasetable_dispatcher",
    "_normalize_datafusion_databasetable_xorq",
    "_normalize_duckdb_databasetable_xorq",
    "_normalize_read_xorq",
]
