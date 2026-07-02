"""Publish — reconcile a staged changeset into a target. See ADR-0017.

The reconciliation layer, with no knowledge of WAP/tee/audit. ``resolve_strategy``
routes on the backend type (matching the con's own MRO, importing nothing) to one
of five publish mechanisms — native ``MERGE INTO``, iceberg upsert+delete,
statement DML, full rewrite, parquet file merge — and ``_validate`` checks the
changeset contract. ``publish(con, staging, final, *, key, mode)`` and
``publish_parquet(...)`` are the standalone entry points; ``xorq.writes.wap``
wraps this same reconciliation behind the tee + audit gate (the ``make_*_wap_expr``
builders). Dependency flows one way: wap imports publish, never the reverse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Collection, Sequence

from xorq.writes.enums import PublishMode, PublishStrategy


if TYPE_CHECKING:
    from xorq.vendor.ibis.backends import BaseBackend


def _backend_strategy(con: "BaseBackend", mode: PublishMode) -> PublishStrategy:
    """The backend's declared publish mechanism (``REWRITE`` if it declares none).

    Capability is a **backend fact**, so each backend declares it: one that can do a
    keyed merge implements ``publish_strategy(self, mode) -> PublishStrategy`` (see
    e.g. ``xorq/backends/duckdb/__init__.py``). The method is looked up on
    ``type(con)`` — not the instance — so it never trips a backend's table-access
    ``__getattr__``, and it is called on the con we already hold, so routing one
    backend imports none of the others (a single-backend install stays lean). A
    backend with no ``publish_strategy`` (datafusion, xorq_datafusion, pandas — the
    immutable tier) falls back to full ``REWRITE``.
    """
    declare = getattr(type(con), "publish_strategy", None)
    if declare is None:
        from xorq.common.utils.logging_utils import get_logger  # noqa: PLC0415

        # Insurance against a capable engine silently doing O(final) rewrites
        # (e.g. trino has native MERGE but declares no strategy yet — XOR-445).
        get_logger(__name__).debug(
            "no publish_strategy declared; falling back to REWRITE",
            backend=type(con).__name__,
            mode=mode.value,
        )
        return PublishStrategy.REWRITE
    return declare(con, mode)


def resolve_strategy(con: "BaseBackend", mode: PublishMode) -> PublishStrategy:
    """Pick the publish mechanism for ``con`` given the user's ``mode``.

    ``APPEND`` is a mode-level fact (every backend appends) and short-circuits;
    otherwise the backend declares its mechanism (see :func:`_backend_strategy`).
    """
    if mode is PublishMode.APPEND:
        return PublishStrategy.APPEND
    return _backend_strategy(con, mode)


def _validate(mode: PublishMode, key: Sequence[str], columns: Collection[str]) -> None:
    """Validate the changeset contract at build time.

    ``key`` is the identity column(s); ``columns`` is the changeset's schema.
    ``_op`` carries delete intent for ``MERGE`` ('D' deletes, anything else
    upserts). Raises ``ValueError`` on any mismatch.
    """
    has_op = "_op" in columns
    if mode is PublishMode.APPEND:
        if key:
            raise ValueError("APPEND takes no key")
        # `_op` is allowed for APPEND: with no merge it is ordinary data and is
        # appended as-is (the legacy append-only builders never reserved it).
        return
    if not key:
        raise ValueError(f"{mode} requires a non-empty key")
    missing = sorted(set(key) - set(columns))
    if missing:
        raise ValueError(
            f"key columns {missing} not in changeset columns {sorted(columns)}"
        )
    if mode is PublishMode.MERGE and not has_op:
        raise ValueError("MERGE requires an '_op' column ('D' deletes, else upsert)")
    if mode is PublishMode.UPSERT and has_op:
        raise ValueError("UPSERT forbids an '_op' column; use MERGE to honor deletes")


# --- Phase 2: publish mechanisms (NATIVE_MERGE, REWRITE) --------------------
#
# A publish function reconciles the staged changeset into ``final`` on ``con``.
# ``columns`` is the staging schema discovered at publish time; ``_op`` ('D'
# deletes, else upsert) is present only for MERGE. Every function consumes
# staging and is responsible for dropping it (post-commit cleanup).


def _staging_columns(con: "BaseBackend", staging: str) -> list[str]:
    return list(con.table(staging).columns)


def _drop_staging(con: "BaseBackend", staging: str) -> None:
    """Consume staging after publish.

    The tee's bulk-path sink registers staging as a *view* on some backends
    (e.g. duckdb), while a direct create or the per-batch path makes a *table*.
    Drop whichever it is.
    """
    try:
        con.drop_table(staging, force=True)
    except Exception as drop_err:  # noqa: BLE001 — duckdb bulk path leaves a view
        drop_view = getattr(con, "drop_view", None)
        if drop_view is None:
            raise
        try:
            drop_view(staging, force=True)
        except Exception:
            # surface the real drop_table failure, not the view-type noise
            raise drop_err from None


def _staging_missing(name) -> RuntimeError:
    """Fail-fast error when staging is absent at publish (empty input / async sink)."""
    return RuntimeError(
        f"staging {name!r} missing at publish. The sink opens its writer on the "
        "first batch, so either the audited input was empty (no batch, no "
        "artifact) or the staging write has not committed yet (async sink?)."
    )


def _data_cols(columns: Sequence[str], mode: PublishMode) -> list[str]:
    """Columns written to ``final``.

    ``_op`` is a control column only for MERGE (stripped); for APPEND/UPSERT it is
    ordinary data and is kept.
    """
    if mode is PublishMode.MERGE:
        return [c for c in columns if c != "_op"]
    return list(columns)


def _ensure_final(
    con: "BaseBackend",
    final: str,
    staging: str,
    columns: Sequence[str],
    mode: PublishMode,
) -> None:
    """Create an empty ``final`` with the data-column schema if absent."""
    if final in con.list_tables():
        return
    final_cols = _data_cols(columns, mode)
    con.create_table(final, schema=con.table(staging).select(final_cols).schema())


def _q(name: str) -> str:
    """Quote a SQL identifier, escaping any embedded double-quote.

    String authoring (not the sqlglot builder API) is deliberate: ``exp.merge``
    exists only on sqlglot >= 26 and ``Update``'s FROM kwarg is keyed ``"from"``
    (silently dropped as ``from_``) before 28, while co-installation with
    ibis 9.5 (CI compat matrix) caps sqlglot < 25.21. Strings + ``parse_one``
    render identically across 24.0–28.6 (verified). See ADR-0017 Alternatives.
    """
    return '"' + name.replace('"', '""') + '"'


# ``(s._op <> 'D' OR s._op IS NULL)`` — 'D' deletes, anything else upserts. The
# parens are load-bearing: the predicate lands after the implicit AND in MERGE
# ``WHEN MATCHED AND <cond>``, where an unparenthesized OR would bind looser.
_NOT_DEL = """(s."_op" <> 'D' OR s."_op" IS NULL)"""


def _merge_query(
    final: str,
    staging: str,
    key: Sequence[str],
    columns: Sequence[str],
    mode: PublishMode,
):
    """A dialect-agnostic ``MERGE`` statement; rendered per backend via
    ``.sql(dialect=con.dialect)``."""
    import sqlglot  # noqa: PLC0415

    data = [c for c in columns if c not in key and c != "_op"]
    on = " AND ".join(f"f.{_q(k)} = s.{_q(k)}" for k in key)
    sets = ", ".join(f"{_q(c)} = s.{_q(c)}" for c in data)
    cols = ", ".join(_q(c) for c in [*key, *data])
    vals = ", ".join(f"s.{_q(c)}" for c in [*key, *data])
    parts = [f"MERGE INTO {_q(final)} AS f USING {_q(staging)} AS s ON {on}"]
    if mode is PublishMode.MERGE:
        # 'D' deletes; anything else (incl. NULL) upserts.
        parts.append("""WHEN MATCHED AND s."_op" = 'D' THEN DELETE""")
        if data:  # key-only table: nothing to UPDATE, omit the clause (else empty SET)
            parts.append(f"WHEN MATCHED AND {_NOT_DEL} THEN UPDATE SET {sets}")
        parts.append(
            f"WHEN NOT MATCHED AND {_NOT_DEL} THEN INSERT ({cols}) VALUES ({vals})"
        )
    else:
        if data:  # key-only table: nothing to UPDATE
            parts.append(f"WHEN MATCHED THEN UPDATE SET {sets}")
        parts.append(f"WHEN NOT MATCHED THEN INSERT ({cols}) VALUES ({vals})")
    return sqlglot.parse_one("\n".join(parts))


def _publish_native_merge(con, staging, final, key, columns, mode) -> None:
    """One ``MERGE INTO`` statement (duckdb/postgres15+/snowflake/databricks/gizmosql).

    APPEND skips the merge for a plain ``INSERT … SELECT``. Both the INSERT and the
    MERGE are authored once with ``"``-quoted identifiers, then rendered to the
    backend's dialect via ``.sql(dialect=con.dialect)`` — so identifiers transpile
    correctly (e.g. backticks on databricks) and ``raw_sql`` gets a plain string.
    """
    import sqlglot  # noqa: PLC0415

    _ensure_final(con, final, staging, columns, mode)
    if mode is PublishMode.APPEND:
        cols = ", ".join(_q(c) for c in _data_cols(columns, mode))
        query = sqlglot.parse_one(
            f"INSERT INTO {_q(final)} ({cols}) SELECT {cols} FROM {_q(staging)}"
        )
    else:
        query = _merge_query(final, staging, key, columns, mode)
    con.raw_sql(query.sql(dialect=con.dialect))
    _drop_staging(con, staging)


def _publish_rewrite(con, staging, final, key, columns, mode) -> None:
    """anti-join + union-all, materialize, replace (datafusion/xorq/pandas).

    APPEND keeps every existing row (no anti-join); UPSERT/MERGE drop the keys
    the changeset supersedes.
    """
    final_cols = _data_cols(columns, mode)
    s = con.table(staging)
    # MERGE: 'D' deletes, anything else (incl. NULL) upserts.
    applied = (
        s.filter((s["_op"] != "D") | s["_op"].isnull())
        if mode is PublishMode.MERGE
        else s
    )
    applied = applied.select(final_cols)
    if final in con.list_tables():
        existing = con.table(final).select(final_cols)
        survivors = (
            existing if mode is PublishMode.APPEND else existing.anti_join(s, key)
        )
        merged = survivors.union(applied, distinct=False)
    else:
        merged = applied
    # Materialize before replacing final, since `merged` reads final.
    materialized = con.to_pyarrow(merged)
    con.create_table(final, materialized, overwrite=True)
    _drop_staging(con, staging)


def _key_filter(rows, key: Sequence[str]):
    """A pyiceberg predicate matching the (possibly composite) keys in ``rows``.

    Composite keys build an ``Or``-of-``And`` with one term per delete row —
    fine for changeset-sized deltas, pathological for bulk deletes (a 100k-row
    composite-key delete makes a 100k-term predicate tree). Single-column keys
    use ``In`` and don't have this problem.
    """
    from pyiceberg.expressions import And, EqualTo, In, Or  # noqa: PLC0415

    if len(key) == 1:
        return In(key[0], rows.column(key[0]).to_pylist())
    cols = {c: rows.column(c).to_pylist() for c in key}
    terms = []
    for i in range(rows.num_rows):
        term = EqualTo(key[0], cols[key[0]][i])
        for c in key[1:]:
            term = And(term, EqualTo(c, cols[c][i]))
        terms.append(term)
    expr = terms[0]
    for t in terms[1:]:
        expr = Or(expr, t)
    return expr


def _publish_upsert_delete(con, staging, final, key, columns, mode) -> None:
    """pyiceberg publish.

    APPEND uses ``publish_staging_table`` (``add_files`` — metadata-only, references
    staging's parquet, no row rewrite). UPSERT/MERGE drive a single ``Transaction``
    (upsert + delete) so they commit as one snapshot.
    """
    if mode is PublishMode.APPEND:
        con.publish_staging_table(staging, final)  # add_files + drops staging
        return

    import pyarrow.compute as pc  # noqa: PLC0415

    final_cols = _data_cols(columns, mode)
    full_staging = f"{con.namespace}.{staging}"
    full_final = f"{con.namespace}.{final}"
    staged = con.catalog.load_table(full_staging).scan().to_arrow()
    if not con.catalog.table_exists(full_final):
        con.catalog.create_table(full_final, schema=staged.select(final_cols).schema)
    tgt = con.catalog.load_table(full_final)
    with tgt.transaction() as tx:
        if mode is PublishMode.MERGE:
            # 'D' deletes; anything else (incl. NULL) upserts.
            is_delete = pc.fill_null(pc.equal(staged.column("_op"), "D"), False)
            deletes = staged.filter(is_delete)
            if deletes.num_rows:
                tx.delete(_key_filter(deletes, key))
            upserts = staged.filter(pc.invert(is_delete)).select(final_cols)
            if upserts.num_rows:
                tx.upsert(upserts, join_cols=list(key))
        else:
            tx.upsert(staged.select(final_cols), join_cols=list(key))
    _drop_staging(con, staging)


def _publish_statement_dml(con, staging, final, key, columns, mode) -> None:
    """In-place UPDATE + INSERT (+ DELETE) in one transaction (sqlite; pg < 15).

    Genuine in-place update (not delete+reinsert), so triggers/identity survive.
    ``UPDATE ... FROM`` is postgres-native and sqlite >= 3.33. Statements are
    authored once like the merge tier and rendered per dialect via
    ``parse_one(...).sql(dialect=con.dialect)``.
    """
    import sqlglot  # noqa: PLC0415

    _ensure_final(con, final, staging, columns, mode)
    if mode is PublishMode.APPEND:
        cols = ", ".join(_q(c) for c in _data_cols(columns, mode))
        stmts = [f"INSERT INTO {_q(final)} ({cols}) SELECT {cols} FROM {_q(staging)}"]
    else:
        data = [c for c in columns if c not in key and c != "_op"]
        # the UPDATE target can't be aliased on this tier: qualify by table name
        onf = " AND ".join(f"{_q(final)}.{_q(k)} = s.{_q(k)}" for k in key)
        # MERGE: rows with NULL _op upsert (only literal 'D' deletes).
        nd = f" AND {_NOT_DEL}" if mode is PublishMode.MERGE else ""
        cols = ", ".join(_q(c) for c in [*key, *data])
        stmts = []
        if data:  # nothing to UPDATE for a key-only table
            sets = ", ".join(f"{_q(c)} = s.{_q(c)}" for c in data)
            stmts.append(
                f"UPDATE {_q(final)} SET {sets} FROM {_q(staging)} s WHERE {onf}{nd}"
            )
        stmts.append(
            f"INSERT INTO {_q(final)} ({cols}) "
            f"SELECT {cols} FROM {_q(staging)} s "
            f"WHERE NOT EXISTS (SELECT 1 FROM {_q(final)} WHERE {onf}){nd}"
        )
        if mode is PublishMode.MERGE:
            stmts.append(
                f"DELETE FROM {_q(final)} WHERE EXISTS "
                f"""(SELECT 1 FROM {_q(staging)} s WHERE {onf} AND s."_op" = 'D')"""
            )
    with con.begin() as cur:
        for stmt in stmts:
            cur.execute(sqlglot.parse_one(stmt).sql(dialect=con.dialect))
    _drop_staging(con, staging)


def _publish_append(con, staging, final, key, columns, mode) -> None:
    """APPEND: add rows via the backend's own mechanism in its no-key form.

    ``resolve_strategy`` short-circuits APPEND to this entry; it re-dispatches to
    the backend's real strategy, whose publish handles ``mode is APPEND``.
    """
    _PUBLISH[_backend_strategy(con, mode)](con, staging, final, key, columns, mode)


_PUBLISH = {
    PublishStrategy.NATIVE_MERGE: _publish_native_merge,
    PublishStrategy.REWRITE: _publish_rewrite,
    PublishStrategy.UPSERT_DELETE: _publish_upsert_delete,
    PublishStrategy.STATEMENT_DML: _publish_statement_dml,
    PublishStrategy.APPEND: _publish_append,
}


def publish(con, staging, final, *, key=(), mode) -> None:
    """Reconcile an existing ``staging`` table into ``final`` on ``con`` — no WAP.

    The non-WAP entry point. Use it when you already have a staging table — a
    plain write, a dbt model, an external load — and want to merge it into
    ``final`` with ``key`` / ``mode`` without the tee + audit gate.
    :func:`xorq.writes.wap.make_backend_wap_expr` is the audited path over this
    same reconciliation. ``final`` is created if absent; ``staging`` is consumed.

    For ``UPSERT``/``MERGE`` the caller owns the changeset contract — in
    particular **unique keys per changeset** (ADR-0017): the mechanisms diverge
    on duplicate keys, and unlike the WAP builders this path has no audit to
    enforce it.
    """
    key = list(key)
    try:
        columns = _staging_columns(con, staging)
    except Exception:  # noqa: BLE001 — staging never materialized
        raise _staging_missing(staging) from None
    _validate(mode, key, columns)
    _PUBLISH[resolve_strategy(con, mode)](con, staging, final, key, columns, mode)


# --- parquet target ---------------------------------------------------------


def _parquet_concat(staging_path, final_path) -> None:
    """APPEND for a parquet target: stream-concat staging into final (pyarrow only).

    Mirrors the append-only ``make_publish_with_parquet`` — no engine, O(batch)
    memory — so wrapping it keeps the cheap append rather than a full rewrite.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    final_path.parent.mkdir(parents=True, exist_ok=True)
    if not final_path.exists():
        staging_path.replace(final_path)  # first run: staging becomes final
        return
    final_schema = pq.ParquetFile(final_path).schema_arrow
    staging_schema = pq.ParquetFile(staging_path).schema_arrow
    if not staging_schema.equals(final_schema, check_metadata=False):
        raise ValueError("cannot append: staging schema does not match final schema")
    tmp = final_path.with_name(final_path.name + ".merge.tmp")
    try:
        with pq.ParquetWriter(tmp, final_schema) as writer:
            for src in (final_path, staging_path):
                for batch in pq.ParquetFile(src).iter_batches():
                    writer.write_batch(batch)
        tmp.replace(final_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    staging_path.unlink(missing_ok=True)


def _publish_parquet_merge(staging_path, final_path, key, mode) -> None:
    """Reconcile a staging parquet into final.

    APPEND streams a pyarrow concat (cheap); UPSERT/MERGE do the anti-join +
    union-all in pandas — final rows whose key the changeset supersedes are
    dropped, then the applied (non-delete) staging rows are appended — writing a
    temp parquet and atomically renaming it over final. Consumes the staging file.

    Pandas, not a query engine: this runs inside the publish UDF, i.e. *within*
    the engine driving the WAP execution, so spinning up xorq's own datafusion
    here would nest one runtime in another ("cannot start a runtime from within a
    runtime"). Pandas is a core dep with no runtime, so it needs no backend and
    makes no observable difference to the result (the file is rewritten whole
    either way).
    """
    from pathlib import Path  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    staging_path, final_path = Path(staging_path), Path(final_path)
    if not staging_path.exists():
        raise _staging_missing(str(staging_path))
    if mode is PublishMode.APPEND:
        _validate(mode, key, pq.ParquetFile(staging_path).schema_arrow.names)
        _parquet_concat(staging_path, final_path)
        return

    staged = pd.read_parquet(staging_path)
    _validate(mode, key, list(staged.columns))
    final_cols = _data_cols(list(staged.columns), mode)
    # MERGE: 'D' deletes; anything else (incl. NULL, since NaN != 'D') upserts.
    applied = (staged[staged["_op"] != "D"] if mode is PublishMode.MERGE else staged)[
        final_cols
    ]
    if final_path.exists():
        existing = pd.read_parquet(final_path)[final_cols]
        # Anti-join: keep existing rows whose key the changeset does not supersede.
        # Staging keys include the 'D' rows, so deletes drop from survivors too.
        superseded = staged[key].drop_duplicates()
        marked = existing.merge(superseded, on=key, how="left", indicator=True)
        survivors = marked[marked["_merge"] == "left_only"][final_cols]
        result = pd.concat([survivors, applied], ignore_index=True)
    else:
        result = applied
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_path.with_name(final_path.name + ".merge.tmp")
    try:
        result.to_parquet(tmp, index=False)
        tmp.replace(final_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    staging_path.unlink(missing_ok=True)


def publish_parquet(staging_path, final_path, *, key=(), mode) -> None:
    """Reconcile a staging parquet file into ``final_path`` — no WAP.

    The parquet analogue of :func:`publish`; consumes the staging file.
    :func:`xorq.writes.wap.make_parquet_wap_expr` is the audited path over the
    same reconciliation.
    """
    _publish_parquet_merge(staging_path, final_path, list(key), mode)
