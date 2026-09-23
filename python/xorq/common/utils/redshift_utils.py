"""Redshift cache-freshness helpers.

Separate from ``postgres_utils`` on purpose. Redshift speaks the PostgreSQL wire
protocol, which is exactly what made it inherit the PostgreSQL *probe*, and that
probe is wrong here in two different ways:

* ``CHECKPOINT`` is not Redshift syntax at all, so the default cache strategy
  failed outright against a Redshift-backed table (rc16 transcript :5101).
* ``ANALYZE "<table>"`` is worse than a syntax error, because Redshift *does*
  accept it. It is a real, expensive, write-privileged operation, and the
  postgres probe runs it as a side effect of computing a cache key. A read-only
  warehouse user cannot run it; a read-write one should not have it run unasked.

So the probe here issues no DDL and holds no write privilege: it is a single
read of ``svv_table_info``, the catalog view Redshift documents for table
metadata.
"""

from __future__ import annotations


# Redshift's own metadata view. ``tbl_rows`` is the total row count including
# rows pending vacuum, which is the property we want: it moves when data is
# written, which is the whole point of a freshness probe, and unlike
# PostgreSQL's ``pg_class.reltuples`` it does not need an ANALYZE first to
# become meaningful.
#
# Deliberately not ``pg_class``: Redshift exposes a pg_class, but its reltuples
# is not maintained the way PostgreSQL's is, so reading it would look like it
# worked and would silently never invalidate a cache.
N_ROWS_SQL = """
SELECT tbl_rows
FROM svv_table_info
WHERE "table" = %(name)s
  AND "schema" = %(schema)s
"""


class RedshiftFreshnessUnavailable(Exception):
    """``svv_table_info`` could not be read, so no freshness key can be built.

    Deliberately its own type, and deliberately raised rather than swallowed.
    See ``get_redshift_n_rows`` for why silently degrading is the one option
    that is worse than failing.
    """


def get_redshift_n_rows(dt):
    """Row count for a Redshift table, issuing no DDL.

    Returns ``None`` when the table is simply absent from ``svv_table_info``.
    Verified against a live Redshift 2026-09-23: a table that exists but has
    had no data written to it does not appear, so ``None`` is the honest answer
    for an empty table and a cache key is the wrong place to turn that into a
    hard failure.

    Raises ``RedshiftFreshnessUnavailable`` when the view cannot be *read*,
    which is a different thing entirely and must not be conflated with absence.
    A read-only warehouse user gets ``InsufficientPrivilege: permission denied
    for relation svv_table_info`` -- verified live against a user with
    ``USAGE`` on the schema and ``SELECT`` on its tables, which is exactly the
    shape of the rc16 reporter's user.

    Why this raises instead of falling back:

    * ``pg_class.reltuples`` is readable by that user, and is the trap. Measured
      live: inserting 5 rows without an ``ANALYZE`` moved the real count 12 ->
      17 and ``svv_table_info`` 12 -> 17, while ``reltuples`` stayed at **12**.
      Keying on it would produce a cache that silently never invalidates.
    * ``SELECT count(*)`` is readable and correct, but it is a full scan on
      every cache-key computation. On the reporter's 2.1M-row fact that is not
      a freshness probe, it is the query.
    * Returning ``None`` would make the key stable-but-meaningless, i.e. the
      same silent staleness as ``reltuples``, with no error to notice.

    So the honest outcome is a loud, actionable failure. ``ParquetSnapshotCache``
    needs no freshness probe at all and is the supported path for a user who
    cannot be granted this -- which the rc16 reporter found by trial (:5222)
    because nothing said so.
    """
    con = dt.source
    schema = dt.namespace.database or "public"
    try:
        with con.con.cursor() as cursor:
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
            f"writes without an ANALYZE, so the cache would go stale silently."
        ) from e
    if not rows:
        return None
    ((n_rows, *_),) = rows
    return n_rows


def _is_permission_error(exc) -> bool:
    """Match a privilege failure without importing psycopg at module scope.

    Checked on the class name rather than by catching
    ``psycopg.errors.InsufficientPrivilege`` directly: this module is imported
    on the cache-key path for every expression hash, and psycopg is an optional
    extra (the ``postgres`` one), so importing it here would break installs that
    do not have it -- the same trap that caught the introspection work.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if type(exc).__name__ == "InsufficientPrivilege":
            return True
        if "permission denied" in str(exc).lower():
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def normalize_redshift_databasetable(dt):
    """Same shape as dasher's per-backend normalizers, with a DDL-free probe.

    The trailing row count is what makes this a *freshness* key rather than an
    identity key. Routing Redshift to dasher's identity-only
    ``normalize_remote_databasetable`` (as trino and gizmosql are routed) would
    also avoid the DDL, and would let a SourceStorage cache serve stale results
    forever -- trading a loud failure for a quiet wrong answer.
    """
    return (
        "ibis.DatabaseTable.redshift",
        dt.name,
        dt.schema,
        dt.source,
        dt.namespace,
        get_redshift_n_rows(dt),
    )
