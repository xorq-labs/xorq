"""Redshift cache-freshness helpers.

Separate from ``postgres_utils`` on purpose. Redshift speaks the PostgreSQL wire
protocol, which is exactly what made it inherit the PostgreSQL *probe*, and that
probe is wrong here in two different ways:

* ``CHECKPOINT`` is not Redshift syntax at all, so the default cache strategy
  failed outright against a Redshift-backed table.
* ``ANALYZE "<table>"`` is worse than a syntax error, because Redshift *does*
  accept it. It is a real, expensive, write-privileged operation, and the
  postgres probe runs it as a side effect of computing a cache key. A read-only
  warehouse user cannot run it; a read-write one should not have it run unasked.

So the probe here issues no DDL and holds no write privilege: it is a single
read of ``svv_table_info``, the catalog view Redshift documents for table
metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xorq.common.exceptions import RedshiftFreshnessUnavailable


if TYPE_CHECKING:
    import xorq.vendor.ibis.expr.operations as ops
    from xorq.backends.redshift import Backend as RedshiftBackend


__all__ = [
    "N_ROWS_SQL",
    "get_redshift_row_counts",
    "normalize_redshift_backend",
    "normalize_redshift_databasetable",
    "resolve_redshift_schema",
]


# SQLSTATE for insufficient_privilege, which is what a user without SELECT on
# ``svv_table_info`` gets. Matched on the code rather than on the exception
# class because this module is imported on the cache-key path for every
# expression hash, and psycopg is an optional extra (the ``postgres`` one), so
# importing it here would break installs that do not have it.
INSUFFICIENT_PRIVILEGE = "42501"


# Redshift's own metadata view.
#
# Both counts are read because neither alone is a sufficient freshness signal:
#
# * ``tbl_rows`` is the total row count *including rows marked for deletion but
#   not yet vacuumed*, so it moves on INSERT but not on DELETE.
# * ``estimated_visible_rows`` excludes those rows, so it moves on DELETE.
#
# Keying on the pair means any movement in either invalidates. The failure mode
# that costs is a spurious invalidation (a recompute — correct, just slower),
# never a missed one (a stale answer served as fresh).
#
# Deliberately not ``pg_class``: Redshift exposes a pg_class, but its reltuples
# is not maintained the way PostgreSQL's is, so reading it would look like it
# worked and would silently never invalidate a cache.
N_ROWS_SQL = """
SELECT tbl_rows, estimated_visible_rows
FROM svv_table_info
WHERE "table" = %(name)s
  AND "schema" = %(schema)s
"""


def resolve_redshift_schema(dt: ops.DatabaseTable) -> str:
    """The schema the probe must read for ``dt``.

    An unqualified ``con.table("offers")`` produces ``Namespace(catalog=None,
    database=None)`` — the backend's ``table()`` passes through whatever the
    caller gave it and does not resolve anything — while the connection's
    ``search_path`` has already been set from the ``schema=`` connect kwarg. So
    defaulting to ``"public"`` here probes a schema the caller never named: the
    row matches nothing, the probe returns ``None`` forever, and a
    ``SourceStorage`` cache never invalidates. That is the silent staleness this
    whole module exists to avoid, arrived at from the other direction.

    ``current_database`` on this backend is ``SELECT current_schema()`` (the
    Redshift override), which is a read and issues no DDL.

    Known limitation, shared with the postgres probe's own ``FIXME``: a
    ``search_path`` with several entries resolves here to the first one, and a
    session-temporary table is not in ``svv_table_info`` under any schema. Both
    fall back to the ``None`` return below, which is honest about the count but
    cannot distinguish "empty" from "not visible to this probe".
    """
    if (database := dt.namespace.database) is not None:
        return database
    return dt.source.current_database


def get_redshift_row_counts(
    dt: ops.DatabaseTable, schema: str | None = None
) -> tuple[int | None, int | None] | None:
    """``(tbl_rows, estimated_visible_rows)`` for a Redshift table, issuing no DDL.

    Returns ``None`` when the table is simply absent from ``svv_table_info``.
    Verified against a live Redshift: a table that exists but has had no data
    written to it does not appear, so ``None`` is the honest answer for an empty
    table and a cache key is the wrong place to turn that into a hard failure.

    Raises ``RedshiftFreshnessUnavailable`` when the view cannot be *read*,
    which is a different thing entirely and must not be conflated with absence.
    A read-only warehouse user gets ``InsufficientPrivilege: permission denied
    for relation svv_table_info`` — verified live against a user with ``USAGE``
    on the schema and ``SELECT`` on its tables, which is the least-privilege
    shape reported from the field.

    Why this raises instead of falling back:

    * ``pg_class.reltuples`` is readable by that user, and is the trap. Measured
      live: inserting 5 rows without an ``ANALYZE`` moved the real count 12 ->
      17 and ``svv_table_info`` 12 -> 17, while ``reltuples`` stayed at **12**.
      Keying on it would produce a cache that silently never invalidates.
    * ``SELECT count(*)`` is readable and correct, but it is a full scan on
      every cache-key computation. On a multi-million-row fact table that is not
      a freshness probe, it is the query.
    * Returning ``None`` would make the key stable-but-meaningless, i.e. the
      same silent staleness as ``reltuples``, with no error to notice.

    So the honest outcome is a loud, actionable failure. ``ParquetSnapshotCache``
    needs no freshness probe at all and is the supported path for a user who
    cannot be granted this — which the field report found only by trial, because
    nothing said so.
    """
    raw = dt.source.con
    schema = resolve_redshift_schema(dt) if schema is None else schema
    try:
        # ``transaction()`` alongside the cursor is the idiom every other
        # cursor use in the postgres/redshift family follows. Without it, under
        # ``autocommit=False`` this read opens a snapshot that is never
        # committed, and the next statement on the connection inherits it.
        with raw.cursor() as cursor, raw.transaction():
            rows = cursor.execute(
                N_ROWS_SQL, {"name": dt.name, "schema": schema}
            ).fetchall()
    except Exception as e:
        if not _is_permission_error(e):
            raise
        raise RedshiftFreshnessUnavailable(
            f"cannot read svv_table_info to compute a cache key for "
            f"{schema}.{dt.name!r}: {e}. The default cache strategy needs a "
            f"row count to detect upstream changes, and this connection's user "
            f"cannot read that view. Either grant it "
            f"(`GRANT SELECT ON svv_table_info TO <user>`) or use a cache that "
            f"needs no freshness probe, e.g. "
            f"`.cache(ParquetSnapshotCache.from_kwargs())`. Do not work around "
            f"this with pg_class.reltuples: on Redshift it does not track "
            f"writes without an ANALYZE, so the cache would go stale silently. "
            f"To handle this in code, catch "
            f"`xorq.common.exceptions.RedshiftFreshnessUnavailable`."
        ) from e
    if not rows:
        return None
    if len(rows) > 1:
        raise RedshiftFreshnessUnavailable(
            f"svv_table_info returned {len(rows)} rows for {schema}.{dt.name!r}, "
            f"so the row count to key on is ambiguous. This happens when more "
            f"than one database on the cluster exposes that schema and table "
            f"name, e.g. across a datashare. Qualify the table explicitly "
            f"(`con.table(name, database=...)`) so exactly one row matches."
        )
    ((tbl_rows, estimated_visible_rows),) = rows
    return (_as_int(tbl_rows), _as_int(estimated_visible_rows))


def _as_int(value: Any) -> int | None:
    """Coerce a catalog count to ``int``, passing ``None`` through.

    ``tbl_rows`` and ``estimated_visible_rows`` are ``numeric(38,0)``, which
    psycopg returns as ``decimal.Decimal``. dasher's encoder accepts only
    str/int/float/bool/bytes/None, so handing it a ``Decimal`` raises
    ``ValueError: No normalizer registered for <class 'decimal.Decimal'>`` at
    tokenize time — after the probe has already succeeded, which is exactly the
    kind of failure an offline fake serving Python ``int`` cannot show you.
    """
    return None if value is None else int(value)


def _is_permission_error(exc: BaseException | None) -> bool:
    """True when ``exc``'s cause chain carries SQLSTATE 42501.

    Matched on ``sqlstate`` rather than on the message: ``"permission denied"``
    appears in ``PermissionError(13, 'Permission denied')`` from an unreadable
    ssl key or pgpass file too, and relabelling one of those with advice to
    ``GRANT SELECT ON svv_table_info`` sends the reader somewhere there is
    nothing to find. psycopg sets ``sqlstate`` on the exception instance, so
    reading it needs no import and no class reference.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if getattr(exc, "sqlstate", None) == INSUFFICIENT_PRIVILEGE:
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def normalize_redshift_backend(con: RedshiftBackend) -> tuple:
    """Connection identity for a Redshift backend.

    Mirrors ``xorq_dasher``'s postgres rule, which this backend cannot reuse:
    that rule is a ``match con.name`` with no ``redshift`` case and a
    ``raise ValueError`` default, so without this every Redshift cache key —
    and ``SnapshotStrategy.normalize_backend``, which is the no-probe path the
    privilege error above recommends — dies with ``no normalization rule for
    backend 'redshift'`` *after* the probe has run.
    """
    info = con.con.info
    params = info.get_parameters()
    return (
        "backend.redshift",
        params.get("host"),
        info.port,
        params.get("dbname"),
    )


def normalize_redshift_databasetable(dt: ops.DatabaseTable) -> tuple:
    """Same shape as dasher's per-backend normalizers, with a DDL-free probe.

    The trailing row counts are what make this a *freshness* key rather than an
    identity key. Routing Redshift to dasher's identity-only
    ``normalize_remote_databasetable`` (as trino and gizmosql are routed) would
    also avoid the DDL, and would let a SourceStorage cache serve stale results
    forever -- trading a loud failure for a quiet wrong answer.

    The resolved schema is in the key as well as ``dt.namespace``, because the
    namespace is ``None`` for an unqualified table: without it, the same table
    name in two schemas on two differently-scoped connections produces two keys
    that differ only by row count.
    """
    schema = resolve_redshift_schema(dt)
    return (
        "ibis.DatabaseTable.redshift",
        dt.name,
        dt.schema,
        dt.source,
        dt.namespace,
        schema,
        get_redshift_row_counts(dt, schema),
    )
