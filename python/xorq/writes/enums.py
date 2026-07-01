from __future__ import annotations

from xorq.common.compat import StrEnum


class WriteMode(StrEnum):
    CREATE = "create"
    APPEND = "append"


class PublishMode(StrEnum):
    """How a staged changeset combines into ``final`` — the one knob a caller sets.

    See ADR-0017.
    """

    APPEND = "append"  # add all staging rows; no key; duplicates allowed
    UPSERT = "upsert"  # insert-or-update by key; no _op column
    MERGE = (
        "merge"  # upsert + delete; requires an _op column ('D' deletes, else upsert)
    )


class PublishStrategy(StrEnum):
    """Mechanism for a publish, auto-resolved from backend capability.

    Internal plumbing — callers pick a :class:`PublishMode`, not a strategy.
    See ADR-0017.
    """

    APPEND = "append"  # add rows: concat / add_files / INSERT…SELECT
    NATIVE_MERGE = "native_merge"  # one MERGE INTO statement
    UPSERT_DELETE = (
        "upsert_delete"  # pyiceberg Transaction.upsert + .delete (one snapshot)
    )
    STATEMENT_DML = "statement_dml"  # UPDATE + INSERT + DELETE in one transaction
    REWRITE = "rewrite"  # anti-join + union-all -> materialize -> atomic swap


class BackendType(StrEnum):
    """Backend ``Backend``-class type-paths used to route a :class:`PublishStrategy`.

    Each value is the fully-qualified ``module.qualname`` of a backend's
    ``Backend`` class. Routing matches a live connection's own MRO against these
    strings and imports nothing (a ``StrEnum`` member *is* its string), so picking
    the strategy for one backend never imports another. See ``writes/publish.py``
    and ADR-0017.
    """

    DUCKDB = "xorq.backends.duckdb.Backend"
    SNOWFLAKE = "xorq.backends.snowflake.Backend"
    DATABRICKS = "xorq.backends.databricks.Backend"
    GIZMOSQL = "xorq.backends.gizmosql.Backend"
    SQLITE = "xorq.backends.sqlite.Backend"
    PYICEBERG = "xorq.backends.pyiceberg.Backend"
    POSTGRES = "xorq.backends.postgres.Backend"
