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


def get_redshift_n_rows(dt):
    """Row count for a Redshift table, issuing no DDL.

    Returns ``None`` when the table is absent from ``svv_table_info`` rather
    than raising. That is not defensive padding -- it is the documented
    behaviour of the view, which only lists tables that are visible to the
    current user AND have had data written to them. An empty or
    freshly-created table legitimately does not appear, and a cache key is the
    wrong place to turn that into a hard failure.

    The cost of ``None`` is a key that cannot distinguish "empty" from
    "invisible", so a table that stays empty keeps a stable key. That is
    correct for the empty case and conservative for the invisible one.
    """
    con = dt.source
    schema = dt.namespace.database or "public"
    with con.con.cursor() as cursor:
        rows = cursor.execute(
            N_ROWS_SQL, {"name": dt.name, "schema": schema}
        ).fetchall()
    if not rows:
        return None
    ((n_rows, *_),) = rows
    return n_rows


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
