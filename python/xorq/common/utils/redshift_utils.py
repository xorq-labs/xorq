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
read of ``pg_catalog.pg_statistic_indicator``, the catalog that
``svv_table_info.estimated_visible_rows`` is itself computed from.

Reading the indicator rather than the view is not a micro-optimisation.
``svv_table_info`` has no ACL at all and is superuser-only, so the
least-privilege warehouse user -- the shape reported twice from the field --
cannot read it without a grant a superuser must issue. The indicator carries a
PUBLIC SELECT grant, so the same number is readable one join earlier by every
user, with no grant, and costs ~24ms against the view's ~417ms: the view reaches
that number through stv_tbl_perm, stv_slices, stv_blocklist and a node-capacity
subquery, none of which a least-privilege user can read either.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xorq.common.exceptions import RedshiftFreshnessUnavailable


if TYPE_CHECKING:
    import xorq.vendor.ibis.expr.operations as ops
    from xorq.backends.redshift import Backend as RedshiftBackend


__all__ = [
    "ROW_COUNTERS_SQL",
    "get_redshift_row_counts",
    "normalize_redshift_backend",
    "normalize_redshift_databasetable",
    "resolve_redshift_schema",
]


# SQLSTATE for insufficient_privilege, which is what a user without SELECT on
# the catalog being read gets. Matched on the code rather than on the exception
# class because this module is imported on the cache-key path for every
# expression hash, and psycopg is an optional extra (the ``postgres`` one), so
# importing it here would break installs that do not have it.
INSUFFICIENT_PRIVILEGE = "42501"


# The three counters ``svv_table_info`` is built from, read at the source.
#
# ``pg_get_viewdef('svv_table_info'::regclass, true)`` shows the view computing
# ``estimated_visible_rows`` as ``sum(stairows)`` over exactly this grouping,
# and ``stats_off`` as ``LEAST((staidels + staiins) * 100 / stairows, 100)``.
# The shape below -- including the ``HAVING`` -- is copied from that definition
# rather than invented, so this reads the same number the view would report.
#
# All three are keyed on because they answer different questions, measured live
# on 2026-09-24 with no ANALYZE at any point:
#
#     step                 actual   stairows  staiins  staidels
#     created, unwritten        0          0        0         0
#     +3 rows                   3          3        3         0
#     +2 rows                   5          5        5         0
#     -1 row                    4          4        5         1
#
# * ``stairows`` is the visible row count and tracked the truth exactly.
# * ``staiins`` and ``staidels`` are inserts and deletes since the last ANALYZE,
#   and move INDEPENDENTLY of ``stairows``. That is what lets this key catch a
#   cardinality-preserving mutation: Redshift implements UPDATE as
#   delete-plus-insert, so an in-place UPDATE moves the pair while leaving the
#   row count where it was. A key on row counts alone cannot see that.
#
# An ANALYZE (including Serverless background auto-analyze) resets ``staiins``
# and ``staidels`` to zero while moving ``stairows``, so the key can change with
# no data change. That is a spurious invalidation -- a recompute, the safe
# direction -- and the price of the extra sensitivity above.
#
# Deliberately not ``pg_class.reltuples``: Redshift exposes it and a
# least-privilege user can read it, which makes it the trap on this path rather
# than the answer. Measured, it sat at 0.001 while this very table reached 5
# rows, and in an earlier run stayed at 12 while the truth moved 12 -> 17.
ROW_COUNTERS_SQL = """
SELECT sum(i.stairows), sum(i.staiins), sum(i.staidels)
FROM pg_catalog.pg_statistic_indicator i
JOIN pg_catalog.pg_class c ON c.oid = i.stairelid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
WHERE c.relname = %(name)s
  AND n.nspname = %(schema)s
GROUP BY i.stairelid
HAVING count(i.stairelid) = 1
"""


# Read only when the indicator returns nothing, and only to say WHY in the
# error. It no longer decides anything: an ordinary table always has an
# indicator row, including one created and never written, which returns
# (0, 0, 0) -- measured, where ``svv_table_info`` returns no row at all for the
# same table. So absence here is never "empty"; it means the relation is one the
# statistics catalog does not track (a view, a late-binding view, a Spectrum
# external table, a session-temp table), or it does not resolve at all, or it
# has several indicator rows and the ``HAVING`` above excluded it as the view
# itself would. relkind tells those apart for the reader.
#
# Measured on a live Redshift: relkind is 'r' for both a populated and an empty
# ordinary table, and no row comes back for a name that does not resolve in the
# schema. pg_class carries a PUBLIC SELECT grant, as the indicator does, so this
# read is available wherever the probe itself is.
RELKIND_SQL = """
SELECT c.relkind
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relname = %(name)s
  AND n.nspname = %(schema)s
"""

# pg_class.relkind for an ordinary table. Every other value is a relation the
# statistics catalog does not track row counters for.
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
) -> tuple[int, int, int]:
    """``(stairows, staiins, staidels)`` for a Redshift table, issuing no DDL.

    Always a triple of ints, or an exception. It never returns ``None``, and an
    empty table is not a special case: a table created and never written returns
    ``(0, 0, 0)``, measured live, where ``svv_table_info`` returns no row at all
    for the same table. The absent-versus-unreadable ambiguity that shaped an
    earlier version of this probe simply does not arise on this signal.

    Readable by a least-privilege user with no grant: verified on a directly
    authenticated connection holding only ``USAGE`` on the schema and ``SELECT``
    on its tables (``usesuper = f``), reading counters for a table it does not
    own, while ``svv_table_info`` in the same session still failed with
    ``permission denied``. ``pg_statistic_indicator``, ``pg_class`` and
    ``pg_namespace`` each carry a PUBLIC SELECT grant; ``svv_table_info`` has no
    ACL at all.

    Raises ``RedshiftFreshnessUnavailable`` in three cases, all loud on purpose:

    * the catalog cannot be *read*. With a PUBLIC grant this should be
      unreachable, so it means the grant has been revoked on this cluster --
      not the ordinary least-privilege case, which now works.
    * the relation has no indicator row: a view, a late-binding view, a Spectrum
      external table, a session-temp table, or a name that does not resolve.
      See ``_no_counters``.
    * more than one row comes back, which the ``HAVING`` should already have
      excluded.

    Why it raises rather than degrading to something readable:

    * ``pg_class.reltuples`` is readable by that user and is the trap. Measured,
      it sat at **0.001** while the table reached 5 rows; in an earlier run it
      stayed at 12 while the truth and ``svv_table_info`` moved 12 -> 17. Keying
      on it produces a cache that silently never invalidates.
    * ``SELECT count(*)`` is readable and correct, but is a full scan on every
      cache-key computation. On a multi-million-row fact that is not a probe,
      it is the query.
    * a key component that cannot change is the same silent staleness as
      ``reltuples``, with no error to notice.

    ``ParquetSnapshotCache`` needs no freshness probe at all and remains the
    supported path for anyone this cannot serve.
    """
    raw = dt.source.con
    # pg_catalog describes the CONNECTED database only, and the probe's WHERE
    # cannot reach past it. A catalog-qualified table would therefore be scored
    # against a same-named table in this database, or against nothing at all --
    # a wrong freshness signal or a frozen key, both silent. Refuse instead.
    if (catalog := dt.namespace.catalog) is not None:
        raise RedshiftFreshnessUnavailable(
            f"cannot compute a freshness key for catalog {catalog!r}, table "
            f"{dt.name!r}: the statistics catalog describes only the connected "
            f"database, "
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
                ROW_COUNTERS_SQL, {"name": dt.name, "schema": schema}
            ).fetchall()
    except Exception as e:
        if not _is_permission_error(e):
            raise
        where = f"{schema}.{dt.name}" if schema is not None else dt.name
        raise RedshiftFreshnessUnavailable(
            f"cannot read the Redshift statistics catalog to compute a cache "
            f"key for {where}: {e}. This is unusual rather than the ordinary "
            f"least-privilege case: pg_statistic_indicator normally carries a "
            f"PUBLIC SELECT grant, so a denial here means it has been revoked "
            f"on this cluster. Restore it (`GRANT SELECT ON "
            f"pg_catalog.pg_statistic_indicator TO <user>`), or use a cache "
            f"that needs no freshness probe, e.g. "
            f"`.cache(ParquetSnapshotCache.from_kwargs())`. Do not work around "
            f"this with pg_class.reltuples: on Redshift it does not track "
            f"writes without an ANALYZE, so the cache would go stale silently. "
            f"To handle this in code, catch "
            f"`xorq.common.exceptions.RedshiftFreshnessUnavailable`."
        ) from e
    if not rows:
        _no_counters(raw, dt, schema)
    if len(rows) > 1:
        raise RedshiftFreshnessUnavailable(
            f"the statistics catalog returned {len(rows)} rows for "
            f"{schema}.{dt.name}, so the counters to key on would be a guess. "
            f"The GROUP BY is per-relation and the HAVING admits only "
            f"single-row relations, so this should not be reachable; treat it "
            f"as a bug in this probe rather than something to work around, and "
            f"use `.cache(ParquetSnapshotCache.from_kwargs())` meanwhile."
        )
    ((stairows, staiins, staidels),) = rows
    return (_as_int(stairows), _as_int(staiins), _as_int(staidels))


def _no_counters(raw: Any, dt: ops.DatabaseTable, schema: str) -> None:
    """Always raises. Reads ``relkind`` only to say what the relation is.

    An ordinary table always has an indicator row -- including one created and
    never written, which returns ``(0, 0, 0)``. So unlike the ``svv_table_info``
    probe this replaced, absence here never means "empty" and there is nothing
    to disambiguate: every path out of this function is an error. ``relkind``
    buys the reader a useful one instead of "no rows".

    An ordinary table reaching here is a real anomaly rather than a user error,
    and the message says so, because the remaining explanation is the ``HAVING``
    in the probe -- copied from the view's own definition -- excluding a relation
    that carries several indicator rows.
    """
    with raw.cursor() as cursor, raw.transaction():
        found = cursor.execute(
            RELKIND_SQL, {"name": dt.name, "schema": schema}
        ).fetchall()
    if found and found[0][0] == RELKIND_ORDINARY_TABLE:
        raise RedshiftFreshnessUnavailable(
            f"{schema}.{dt.name} is an ordinary table but has no row in the "
            f"statistics catalog, which should not happen -- an empty table "
            f"reports (0, 0, 0). The likely cause is that it carries several "
            f"indicator rows and the probe's HAVING excluded it, exactly as "
            f"svv_table_info itself would. Treat it as a bug in this probe, and "
            f"use `.cache(ParquetSnapshotCache.from_kwargs())` meanwhile."
        )
    kind = f"relkind {found[0][0]!r}" if found else "not present in pg_class"
    raise RedshiftFreshnessUnavailable(
        f"{schema}.{dt.name} has no row in the statistics catalog and is "
        f"{kind}, so it "
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
    a missing catalog grant sends the reader somewhere there is
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
