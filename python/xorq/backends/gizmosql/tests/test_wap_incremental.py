"""Incremental WAP publish over gizmosql (NATIVE_MERGE, duckdb dialect). ADR-0017 / XOR-445.

gizmosql is a DuckDB-backed Flight SQL server, so it routes to ``NATIVE_MERGE`` and
renders the duckdb dialect (which supports ``MERGE INTO``). Gated behind
``@pytest.mark.gizmosql``; the conftest ``con`` fixture spins up a local server
(skips if the ``gizmosql`` package is absent).
"""

from __future__ import annotations

import pandas as pd
import pytest

import xorq.api as xo
from xorq.vendor.ibis import util
from xorq.writes.enums import PublishMode
from xorq.writes.wap import make_backend_wap_expr


def _run(con, data: pd.DataFrame, key, mode, staging: str, final: str) -> pd.DataFrame:
    src = xo.connect().register(xo.memtable(data), table_name=util.gen_name("src"))
    return src.pipe(
        make_backend_wap_expr(con, key=key, mode=mode), staging, final
    ).execute()


@pytest.mark.gizmosql
def test_gizmosql_incremental_upsert(con) -> None:
    staging, final = util.gen_name("staging"), util.gen_name("final")
    try:
        _run(
            con,
            pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}),
            ["id"],
            PublishMode.UPSERT,
            staging,
            final,
        )
        out = _run(
            con,
            pd.DataFrame({"id": [2, 3], "v": ["B", "c"]}),  # update 2, insert 3
            ["id"],
            PublishMode.UPSERT,
            staging,
            final,
        )
        assert out["published"].iloc[0]
        got = con.table(final).execute().sort_values("id").reset_index(drop=True)
        assert got["id"].tolist() == [1, 2, 3]
        assert got["v"].tolist() == ["a", "B", "c"]
    finally:
        con.drop_table(final, force=True)


@pytest.mark.gizmosql
def test_gizmosql_incremental_merge_with_deletes(con) -> None:
    staging, final = util.gen_name("staging"), util.gen_name("final")
    try:
        _run(
            con,
            pd.DataFrame({"id": [1, 2, 3], "v": ["a", "b", "c"]}),
            ["id"],
            PublishMode.UPSERT,
            staging,
            final,
        )
        # update 2, delete 3, insert 4
        delta = pd.DataFrame(
            {"id": [2, 3, 4], "v": ["B", "x", "d"], "_op": ["U", "D", "I"]}
        )
        out = _run(con, delta, ["id"], PublishMode.MERGE, staging, final)
        assert out["published"].iloc[0]
        got = con.table(final).execute().sort_values("id").reset_index(drop=True)
        assert got["id"].tolist() == [1, 2, 4]
        assert got["v"].tolist() == ["a", "B", "d"]
    finally:
        con.drop_table(final, force=True)
