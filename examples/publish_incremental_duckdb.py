"""Incremental publish into a DuckDB table — the standalone (non-WAP) primitive.

``publish(con, staging, final, *, key, mode)`` reconciles a staging table you
already have into ``final`` — upsert / merge / delete — with **no tee and no audit
gate**. Use it when the changeset is already staged (a plain write, a dbt model,
an external load); reach for the WAP builders (``wap_incremental_duckdb.py``) when
you want the write + audit + publish pipeline instead.

The mechanism is the backend's own capability: duckdb declares ``NATIVE_MERGE``, so
publish is a single ``MERGE INTO`` (ADR-0017). ``publish`` creates ``final`` if
absent and consumes ``staging``.

``PublishMode``: ``UPSERT`` (insert-or-update by key) and ``MERGE`` (upsert +
delete via an ``_op`` column — ``'D'`` deletes, anything else upserts). For
``UPSERT``/``MERGE`` the caller owns the changeset contract (unique keys per
delta); unlike the WAP builders this path has no audit to enforce it.
"""

from __future__ import annotations

import pandas as pd

import xorq.api as xo
from xorq.writes import PublishMode, publish


con = xo.duckdb.connect()


def read_final() -> pd.DataFrame:
    return con.table("final").execute().sort_values("id").reset_index(drop=True)


if __name__ in ("__main__", "__pytest_main__"):
    con.create_table("final", pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}))

    # UPSERT: reconcile an already-staged changeset — update id=2, insert id=3.
    con.create_table("staging", pd.DataFrame({"id": [2, 3], "v": ["B", "c"]}))
    publish(con, "staging", "final", key=["id"], mode=PublishMode.UPSERT)
    assert read_final()["id"].tolist() == [1, 2, 3]
    assert read_final()["v"].tolist() == ["a", "B", "c"]

    # MERGE: update id=2, delete id=3, insert id=4 (via the _op column).
    con.create_table(
        "staging",
        pd.DataFrame({"id": [2, 3, 4], "v": ["BB", "x", "d"], "_op": ["U", "D", "I"]}),
    )
    publish(con, "staging", "final", key=["id"], mode=PublishMode.MERGE)
    print(read_final().to_string(index=False))
    assert read_final()["id"].tolist() == [1, 2, 4]
    assert read_final()["v"].tolist() == ["a", "BB", "d"]

    pytest_examples_passed = True
