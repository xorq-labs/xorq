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


class StagingStrategy(StrEnum):
    """Where the audited changeset is staged before publish.

    ``TABLE`` (default) stages into a separate table on the target backend and
    publishes by reconciling it into ``final`` (any :class:`PublishMode`).
    ``BRANCH`` stages on a branch *of the final table itself* and publishes by
    fast-forwarding main — metadata-only, all-or-nothing snapshot promotion, so
    it is ``APPEND``-only (no keyed merge is meaningful) and requires a backend
    whose type declares ``publish_branch`` (pyiceberg). See ADR-0017.
    """

    TABLE = "table"
    BRANCH = "branch"


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
