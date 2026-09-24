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
# Keying on the pair means any movement in EITHER invalidates, so the common
# failure is a spurious invalidation (a recompute -- correct, just slower).
# It is not a guarantee against missed ones: a cardinality-preserving mutation
# (an UPDATE in place, or a delete-and-reinsert followed by the automatic
# vacuum) returns both counts to a value some retained entry is already keyed
# on. Row counts are a cheap freshness signal, not a content hash.
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


# Read only when svv_table_info returns nothing, to tell the two causes apart.
# That view lists user tables and materialized views WITH AT LEAST ONE ROW, so
# an empty answer means either "an ordinary table with no rows yet" -- honest,
# and ``None`` is the right key component -- or "a relation this view never
# tracks": a plain view, a late-binding view, a Spectrum external table, or a
# session-temp table. Without this second read those are one case, and the
# second silently yields a key that can never change.
#
# Measured on a live Redshift: relkind is 'r' for both a populated and an empty
# ordinary table, and no row comes back for a name that does not resolve in the
# schema. pg_class is readable by a least-privilege user (unlike
# svv_table_info), so this runs on the path where the probe is most constrained.
RELKIND_SQL = """
SELECT c.relkind
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relname = %(name)s
  AND n.nspname = %(schema)s
"""

# pg_class.relkind for an ordinary table. Every other value is a relation
# svv_table_info does not track row counts for.
RELKIND_ORDINARY_TABLE = "r"


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

    A multi-entry ``search_path`` is NOT a hazard here, though an earlier
    version of this docstring claimed it was. ``get_schema``
    (``vendor/ibis/backends/postgres/__init__.py``) resolves an unqualified name
    through ``database or self.current_database`` -- the same expression -- so
    any table ``table()`` accepted lives in the schema this reads. A table
    reachable only via a later ``search_path`` entry raises ``TableNotFound`` at
    ``table()`` time; it never becomes a stale key.
    """
    if (database := dt.namespace.database) is not None:
        return database
    return dt.source.current_database


def get_redshift_row_counts(
    dt: ops.DatabaseTable, schema: str | None = None
) -> tuple[int | None, int | None] | None:
    """``(tbl_rows, estimated_visible_rows)`` for a Redshift table, issuing no DDL.

    Returns ``None`` for an ordinary table that is absent from
    ``svv_table_info``. Verified against a live Redshift: a table that exists but
    has had no data written to it does not appear, so ``None`` is the honest
    answer for an empty table and a cache key is the wrong place to turn that
    into a hard failure. A relation the view does not track at all -- a view, an
    external table, a temp table -- is NOT that case and raises; see
    ``_absent_row_counts``.

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
    # svv_table_info covers the CONNECTED database only, and the probe's WHERE
    # cannot reach past it. A catalog-qualified table would therefore be scored
    # against a same-named table in this database, or against nothing at all --
    # a wrong freshness signal or a frozen key, both silent. Refuse instead.
    if (catalog := dt.namespace.catalog) is not None:
        raise RedshiftFreshnessUnavailable(
            f"cannot compute a freshness key for catalog {catalog!r}, table "
            f"{dt.name!r}: svv_table_info describes only the connected database, "
            f"so a table qualified with another catalog would be scored against "
            f"a same-named table here, or against nothing at all -- a wrong "
            f"signal or a frozen key, and both are silent. Connect to that "
            f"database directly, or use a cache that needs no freshness probe, "
            f"e.g. `.cache(ParquetSnapshotCache.from_kwargs())`."
        )
    # Resolution is INSIDE the try: for an unqualified table it is itself a
    # round trip, and a privilege failure there deserves the same actionable
    # error a qualified table gets rather than a raw driver traceback.
    try:
        schema = resolve_redshift_schema(dt) if schema is None else schema
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
        where = f"{schema}.{dt.name}" if schema is not None else dt.name
        raise RedshiftFreshnessUnavailable(
            f"cannot read the Redshift catalog to compute a cache key for "
            f"{where}: {e}. The default cache strategy needs a row count to "
            f"detect upstream changes, and this connection's user cannot read "
            f"it. Either grant it (`GRANT SELECT ON svv_table_info TO <user>`) "
            f"or use a cache that needs no freshness probe, e.g. "
            f"`.cache(ParquetSnapshotCache.from_kwargs())`. Do not work around "
            f"this with pg_class.reltuples: on Redshift it does not track "
            f"writes without an ANALYZE, so the cache would go stale silently. "
            f"To handle this in code, catch "
            f"`xorq.common.exceptions.RedshiftFreshnessUnavailable`."
        ) from e
    if not rows:
        return _absent_row_counts(raw, dt, schema)
    if len(rows) > 1:
        raise RedshiftFreshnessUnavailable(
            f"svv_table_info returned {len(rows)} rows for {schema}.{dt.name}, "
            f"so the row count to key on would be a guess. The view describes "
            f"one database and one row per relation, so this should not be "
            f"reachable; treat it as a bug in this probe rather than something "
            f"to work around, and use `.cache(ParquetSnapshotCache.from_kwargs())` "
            f"meanwhile."
        )
    ((tbl_rows, estimated_visible_rows),) = rows
    return (_as_int(tbl_rows), _as_int(estimated_visible_rows))


def _absent_row_counts(
    raw: Any, dt: ops.DatabaseTable, schema: str
) -> tuple[int | None, int | None] | None:
    """Decide what an empty ``svv_table_info`` answer means.

    Two very different conditions produce it, and conflating them is how a
    cache goes quietly stale:

    * an ordinary table with no rows written yet -- the view tracks only
      relations with at least one row, so absence is the honest answer and
      ``None`` is a sound key component: it changes as soon as data lands;
    * a relation the view never tracks at all -- a view, a late-binding view, a
      Spectrum external table, or a session-temp table (which this backend's own
      psycopg ingest creates). For those, ``None`` is not "empty", it is
      "unknowable", and returning it yields a key that can never change no
      matter what the underlying data does.

    Only the first may return ``None``. ``relkind`` separates them for one
    extra read, taken only on this path, against a catalog a least-privilege
    user can read.
    """
    with raw.cursor() as cursor, raw.transaction():
        found = cursor.execute(
            RELKIND_SQL, {"name": dt.name, "schema": schema}
        ).fetchall()
    if found and found[0][0] == RELKIND_ORDINARY_TABLE:
        return None
    kind = f"relkind {found[0][0]!r}" if found else "not present in pg_class"
    raise RedshiftFreshnessUnavailable(
        f"{schema}.{dt.name} is absent from svv_table_info and is {kind}, so it "
        f"is not an ordinary table this probe can measure -- a view, an external "
        f"table or a session-temporary table, depending. Returning no row count "
        f"for it would produce a cache key that never changes, which is worse "
        f"than this error. Cache the query that defines it instead, or use "
        f"`.cache(ParquetSnapshotCache.from_kwargs())`, which needs no probe."
    )


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
