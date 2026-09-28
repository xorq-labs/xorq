"""Redshift cache-key helpers.

Separate from ``postgres_utils`` on purpose. Redshift speaks the PostgreSQL wire
protocol, which is exactly what made it inherit the PostgreSQL freshness
*probe*, and that probe is wrong here in two different ways:

* ``CHECKPOINT`` is not Redshift syntax at all, so the default cache strategy
  failed outright against a Redshift-backed table.
* ``ANALYZE "<table>"`` is worse than a syntax error, because Redshift *does*
  accept it. It is a real, expensive, write-privileged operation, and the
  postgres probe runs it as a side effect of computing a cache key. A read-only
  warehouse user cannot run it; a read-write one should not have it run unasked.

Nor is there a Redshift probe to put in its place. A freshness key needs a
per-table signal that moves whenever the data does and never returns to an
earlier value, readable by a user holding only ``USAGE`` and ``SELECT``:

* ``pg_statistic_indicator`` is readable (a PUBLIC grant, measured), but its
  insert and delete counters count "since the last ANALYZE" and reset on every
  one, background auto-analyze included. An ``UPDATE`` followed by an analyze
  returns the counters to the values they had before the ``UPDATE``, so the key
  returns to one whose cached result predates the change, and that result is
  served. A dropped and recreated table reloaded to the same row count
  reproduces its predecessor's counters the same way.
* ``pg_class.reltuples`` is readable and is not maintained on write: measured,
  it sat at 0.001 while a table reached 5 rows.
* ``svv_table_info`` is superuser-only (measured). ``stv_tbl_perm``,
  ``stl_analyze`` and ``sys_analyze_history`` are documented as
  superuser-only, and the ``stl_*`` logs show a regular user only its own
  rows.

So a freshness key over a Redshift table is refused, before any statement is
sent, and the snapshot key is the supported one. That key is identity only and
must therefore identify the relation completely, which is what
``resolve_redshift_schema`` is for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xorq.common.exceptions import RedshiftFreshnessUnavailable


if TYPE_CHECKING:
    import xorq.vendor.ibis.expr.operations as ops
    from xorq.backends.redshift import Backend as RedshiftBackend


__all__ = [
    "normalize_redshift_backend",
    "normalize_redshift_databasetable",
    "normalize_redshift_snapshot_databasetable",
    "resolve_redshift_schema",
]


# The session temp schema holding a table of this name. ``^`` is the LIKE
# escape because a backslash would have to survive psycopg's quoting as well as
# SQL's; ``%%`` is a literal ``%`` to psycopg.
SESSION_TEMP_RELATION_SQL = """
SELECT table_schema
FROM svv_columns
WHERE table_schema LIKE 'pg^_temp^_%%' ESCAPE '^'
  AND table_name = %(name)s
LIMIT 1
"""


def resolve_redshift_schema(dt: ops.DatabaseTable) -> str:
    """The schema an unqualified ``dt`` actually names.

    An unqualified ``con.table("offers")`` produces ``Namespace(catalog=None,
    database=None)`` -- the backend's ``table()`` passes through whatever the
    caller gave it and does not resolve anything -- while the connection's
    ``search_path`` has already been set from the ``schema=`` connect kwarg. So
    the namespace alone cannot tell ``a.offers`` from ``b.offers`` read through
    two differently-scoped connections to one cluster, and an identity key
    built from it serves one schema's snapshot for the other.

    ``current_database`` on this backend is ``SELECT current_schema()`` (the
    Redshift override), which is a read and issues no DDL.

    A multi-entry ``search_path`` is NOT a hazard here. ``get_schema``
    (``vendor/ibis/backends/postgres/__init__.py``) resolves an unqualified name
    through ``database or self.current_database`` -- the same expression -- so a
    table reachable only via a later ``search_path`` entry raises
    ``TableNotFound`` at ``table()`` time; it never reaches a key.

    The session-temporary schema IS a hazard, and is why this can raise.
    ``table()`` also accepts a name that exists only in the session's
    ``pg_temp_<N>`` schema, and this backend mints such tables itself, through
    ``create_table(..., temporary=True)``. Resolving such a name to
    ``current_schema()`` would key a *different*, permanent relation that merely
    shares the name. Nor can a temp table be keyed on its own terms: it is
    invisible to every other session and gone when this one ends, so a key
    naming it would describe something no later reader can see. Refuse instead.

    Redshift has no ``pg_my_temp_schema()``, so the inherited postgres
    ``_session_temp_db`` cannot answer which schema that is: it raises
    ``UndefinedFunction``. See ``_session_temp_schema_of``.
    """
    if (database := dt.namespace.database) is not None:
        return database
    con = dt.source
    schema = con.current_database
    temp_schema = _session_temp_schema_of(con.con, dt.name)
    if temp_schema is None:
        return schema
    raise RedshiftFreshnessUnavailable(
        f"{dt.name!r} resolves to the session-temporary schema {temp_schema}, "
        f"which no cache key can describe: the table is invisible to every "
        f"other session and is dropped when this one ends. Keying "
        f"{schema}.{dt.name} instead would be worse -- it would describe a "
        f"different, permanent relation that happens to share the name. "
        f"Qualify the table with a permanent schema if that is what you meant, "
        f"or cache the query that populates the temp table instead."
    )


def _session_temp_schema_of(raw: Any, name: str) -> str | None:
    """The ``pg_temp_<N>`` schema holding a temp table ``name``, if any.

    Matched by pattern because the schema cannot be looked up first: Redshift
    has no ``pg_my_temp_schema()``. ``svv_columns`` because it lists temporary
    tables where ``svv_all_columns`` does not. Binding a temp table by name
    needs the same view and pattern, for the same two reasons, so the key and
    ``table()`` can agree on which names are temporary.
    """
    with raw.cursor() as cursor, raw.transaction():
        row = cursor.execute(SESSION_TEMP_RELATION_SQL, {"name": name}).fetchone()
    return row[0] if row else None


def normalize_redshift_backend(con: RedshiftBackend) -> tuple:
    """Connection identity for a Redshift backend.

    Mirrors ``xorq_dasher``'s postgres rule, which this backend cannot reuse:
    that rule is a ``match con.name`` with no ``redshift`` case and a
    ``raise ValueError`` default, so without this every Redshift snapshot key
    dies with ``no normalization rule for backend 'redshift'``.
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
    """Always raises: Redshift has no signal a freshness key can rely on.

    This is the global (data-sensitive) rule, reached by ``ParquetCache`` and
    every other ``ModificationTimeStrategy`` cache. Refusing here, before any
    statement is sent, is the alternative to routing Redshift to dasher's
    identity-only ``normalize_remote_databasetable`` (as trino and gizmosql are
    routed), which would let those caches serve stale results forever -- a
    loud failure traded for a quiet wrong answer. See the module docstring for
    why no Redshift catalog read can do better.
    """
    raise RedshiftFreshnessUnavailable(
        f"cannot compute a freshness cache key for Redshift table {dt.name!r}: "
        f"Redshift exposes no per-table change signal that a cache key can "
        f"rely on. The counters a least-privilege user can read "
        f"(pg_statistic_indicator) reset on every ANALYZE, background "
        f"auto-analyze included, so after a change that leaves the row count "
        f"unchanged the key can return to its value from before the change, "
        f"and the result cached then would be served. Use "
        f"`.cache(ParquetSnapshotCache.from_kwargs())` instead -- from the "
        f"command line, `xorq run-cached --cache-type snapshot` -- and drop the "
        f"cached entry when the table's data changes. To handle this in code, "
        f"catch `xorq.common.exceptions.RedshiftFreshnessUnavailable`."
    )


def normalize_redshift_snapshot_databasetable(dt: ops.DatabaseTable) -> tuple:
    """Snapshot identity for a Redshift table: the fallback's, plus the schema.

    ``SnapshotStrategy``'s fallback keys on ``name``, ``schema``, ``source`` and
    ``namespace``, and this backend's ``source`` identity is host, port and
    database only. For an unqualified table the namespace is empty, so two
    connections scoped to different schemas gave ``a.offers`` and ``b.offers``
    one key, and the snapshot of one was served for the other. The resolved
    schema closes that, and resolving it also refuses a session temp table.
    """
    keys = ("name", "schema", "source", "namespace")
    return (
        *((k, getattr(dt, k)) for k in keys),
        ("resolved_schema", resolve_redshift_schema(dt)),
    )
