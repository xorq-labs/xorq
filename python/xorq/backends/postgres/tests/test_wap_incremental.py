"""Incremental WAP publish over postgres (NATIVE_MERGE on pg 15+). ADR-0017 / XOR-445.

postgres 15+ routes to native ``MERGE INTO``; older servers fall back to
``STATEMENT_DML`` (UPDATE+INSERT+DELETE in one transaction). Gated behind
``@pytest.mark.postgres`` and the CI postgres service (``connect_env`` reads the
``POSTGRES_*`` env vars).
"""

from __future__ import annotations

import pandas as pd
import pytest

import xorq.api as xo
from xorq.vendor.ibis import util
from xorq.writes.enums import PublishMode
from xorq.writes.wap import make_backend_wap_expr


@pytest.fixture
def pg_con():
    return xo.postgres.connect_env()


def _run(con, data: pd.DataFrame, key, mode, staging: str, final: str) -> pd.DataFrame:
    src = xo.connect().register(xo.memtable(data), table_name=util.gen_name("src"))
    return src.pipe(
        make_backend_wap_expr(con, key=key, mode=mode), staging, final
    ).execute()


@pytest.mark.postgres
def test_postgres_incremental_upsert(pg_con) -> None:
    staging, final = util.gen_name("staging"), util.gen_name("final")
    try:
        _run(
            pg_con,
            pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}),
            ["id"],
            PublishMode.UPSERT,
            staging,
            final,
        )
        out = _run(
            pg_con,
            pd.DataFrame({"id": [2, 3], "v": ["B", "c"]}),  # update 2, insert 3
            ["id"],
            PublishMode.UPSERT,
            staging,
            final,
        )
        assert out["published"].iloc[0]
        got = pg_con.table(final).execute().sort_values("id").reset_index(drop=True)
        assert got["id"].tolist() == [1, 2, 3]
        assert got["v"].tolist() == ["a", "B", "c"]
    finally:
        pg_con.drop_table(final, force=True)


@pytest.mark.postgres
def test_postgres_incremental_merge_with_deletes(pg_con) -> None:
    staging, final = util.gen_name("staging"), util.gen_name("final")
    try:
        _run(
            pg_con,
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
        out = _run(pg_con, delta, ["id"], PublishMode.MERGE, staging, final)
        assert out["published"].iloc[0]
        got = pg_con.table(final).execute().sort_values("id").reset_index(drop=True)
        assert got["id"].tolist() == [1, 2, 4]
        assert got["v"].tolist() == ["a", "B", "d"]
    finally:
        pg_con.drop_table(final, force=True)
