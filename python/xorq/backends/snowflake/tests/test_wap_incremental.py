"""Incremental WAP publish over Snowflake (NATIVE_MERGE strategy). ADR-0017.

Snowflake routes to the native ``MERGE INTO`` path (verified to render via
sqlglot). These round-trips require live Snowflake credentials, so they are
gated behind ``@pytest.mark.snowflake`` and the env-keypair fixtures in
``conftest.py``; they do not run without an account.
"""

from __future__ import annotations

import pandas as pd
import pytest

import xorq.api as xo
from xorq.backends.snowflake.tests.conftest import inside_temp_schema
from xorq.writes.enums import PublishMode
from xorq.writes.wap import make_backend_wap_expr


def _run(con, data: pd.DataFrame, mode: PublishMode) -> pd.DataFrame:
    src = xo.connect().register(xo.memtable(data), table_name="src")
    return src.pipe(
        make_backend_wap_expr(con, key=["id"], mode=mode),
        "staging",
        "final",
    ).execute()


@pytest.mark.snowflake
def test_snowflake_incremental_upsert(sf_con, temp_catalog, temp_db) -> None:
    with inside_temp_schema(sf_con, temp_catalog, temp_db):
        _run(sf_con, pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}), PublishMode.UPSERT)
        out = _run(
            sf_con,
            pd.DataFrame({"id": [2, 3], "v": ["B", "c"]}),  # update 2, insert 3
            PublishMode.UPSERT,
        )
        assert out["published"].iloc[0]
        final = sf_con.table("final").execute().sort_values("id").reset_index(drop=True)
        assert final["id"].tolist() == [1, 2, 3]
        assert final["v"].tolist() == ["a", "B", "c"]


@pytest.mark.snowflake
def test_snowflake_incremental_merge_with_deletes(
    sf_con, temp_catalog, temp_db
) -> None:
    with inside_temp_schema(sf_con, temp_catalog, temp_db):
        _run(
            sf_con,
            pd.DataFrame({"id": [1, 2, 3], "v": ["a", "b", "c"]}),
            PublishMode.UPSERT,
        )
        # update 2, delete 3, insert 4
        delta = pd.DataFrame(
            {"id": [2, 3, 4], "v": ["B", "x", "d"], "_op": ["U", "D", "I"]}
        )
        out = _run(sf_con, delta, PublishMode.MERGE)
        assert out["published"].iloc[0]
        final = sf_con.table("final").execute().sort_values("id").reset_index(drop=True)
        assert final["id"].tolist() == [1, 2, 4]
        assert final["v"].tolist() == ["a", "B", "d"]
