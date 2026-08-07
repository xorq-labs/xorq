"""Incremental Write-Audit-Publish into a DuckDB table (upsert / merge / delete).

Incremental publish applies a *changeset* — insert-or-update by key, or delete —
rather than appending. Like every WAP flow it is a **deferred expression**:
``source.pipe(make_backend_wap_expr(con, key=…, mode=…), staging, final, audit_fn)``
builds a lazy expr; nothing is written until ``.execute()``. The audit aggregate
is a pipeline breaker, so staging fully commits before the publish decision runs.
The mechanism is the backend's own capability — duckdb declares ``NATIVE_MERGE``,
so publish is a single ``MERGE INTO`` (ADR-0017).

``PublishMode``: ``UPSERT`` (insert-or-update by key) and ``MERGE`` (upsert +
delete via an ``_op`` column — ``'D'`` deletes, anything else upserts).

``audit_fn`` is supplied at the pipe (the "A" in WAP): pass fails -> publish,
audit fails -> nothing published, staging retained. Omit it and ``UPSERT``/
``MERGE`` fall back to a default no-duplicate-keys gate.
"""

from __future__ import annotations

import pandas as pd

import xorq.api as xo
from xorq.writes import PublishMode, make_backend_wap_expr


STAGING, FINAL = "staging", "final"
con = xo.duckdb.connect()


def audit_no_nulls(df: pd.DataFrame) -> bool:
    """DQ gate: reject the whole changeset if any value in ``v`` is null."""
    return bool(df["v"].notna().all())


def src(data: dict, name: str):
    return xo.connect().register(xo.memtable(data), table_name=name)


# Deferred WAP exprs; the audit_fn rides at the pipe. Nothing runs until execute().
upsert = src({"id": [2, 3], "v": ["B", "c"]}, "upsert").pipe(  # update 2, insert 3
    make_backend_wap_expr(con, key=["id"], mode=PublishMode.UPSERT),
    STAGING,
    FINAL,
    audit_no_nulls,
)
merge = src(  # update 2, delete 3, insert 4 (via the _op column)
    {"id": [2, 3, 4], "v": ["BB", "x", "d"], "_op": ["U", "D", "I"]}, "merge"
).pipe(
    make_backend_wap_expr(con, key=["id"], mode=PublishMode.MERGE),
    STAGING,
    FINAL,
    audit_no_nulls,
)
bad = src({"id": [9], "v": [None]}, "bad").pipe(  # a null -> fails the audit
    make_backend_wap_expr(con, key=["id"], mode=PublishMode.UPSERT),
    STAGING,
    FINAL,
    audit_no_nulls,
)


def read_final() -> pd.DataFrame:
    return con.table(FINAL).execute().sort_values("id").reset_index(drop=True)


if __name__ in ("__main__", "__pytest_main__"):
    con.create_table(
        FINAL, pd.DataFrame({"id": [1, 2], "v": ["a", "b"]})
    )  # seed target

    assert upsert.execute()["published"].iloc[0]  # -> {1:a, 2:B, 3:c}
    assert merge.execute()["published"].iloc[0]  # -> {1:a, 2:BB, 4:d}  (id 3 deleted)
    print(read_final().to_string(index=False))
    assert read_final()["id"].tolist() == [1, 2, 4]
    assert read_final()["v"].tolist() == ["a", "BB", "d"]

    # audit rejects the null -> not published, final untouched, staging retained
    out = bad.execute()
    assert not out["passed"].iloc[0] and not out["published"].iloc[0]
    assert read_final()["id"].tolist() == [1, 2, 4]
    print("audit rejected the bad changeset; final unchanged")

    # For the same reconciliation *without* the tee + audit gate (an already-
    # staged changeset), see publish_incremental_duckdb.py.
    pytest_examples_passed = True
