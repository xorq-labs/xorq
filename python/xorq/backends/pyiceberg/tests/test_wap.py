"""WAP (Write-Audit-Publish) over the pyiceberg backend.

Both strategies go through ``make_iceberg_wap_expr``:

  * branch-based (``make_iceberg_wap_expr(con, table_name)``): staging branch
    is fast-forwarded into main on pass, retained on fail.
  * table-based (``make_iceberg_wap_expr(con)``): staging table is copied into
    final and dropped on pass, retained on fail.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import pytest

import xorq.api as xo
import xorq.vendor.ibis.expr.types as ir
from xorq.backends.pyiceberg import Backend
from xorq.writes import make_iceberg_wap_expr
from xorq.writes.wap import FINAL as FINAL_COL
from xorq.writes.wap import PASSED as PASSED_COL
from xorq.writes.wap import STAGING as STAGING_COL
from xorq.writes.wap import make_publish_with_iceberg


FINAL = "final"
STAGING = "staging"

good = {"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]}
bad = {"a": [1, None, 3, 4], "b": ["w", "x", "y", "z"]}


def no_nulls(df: pd.DataFrame) -> bool:
    return bool(df["a"].notna().all())


def _wap_expr(
    wap: Callable[..., ir.Table], data: dict[str, list], table_name: str
) -> ir.Table:
    return (
        xo.connect()
        .register(xo.memtable(data), table_name=table_name)
        .pipe(wap, STAGING, FINAL, no_nulls)
    )


def _refs(con: Backend) -> dict:
    return con.catalog.load_table(f"{con.namespace}.{FINAL}").refs()


def test_branch_wap_pass(fresh_con: Backend) -> None:
    wap = make_iceberg_wap_expr(fresh_con, FINAL)
    out = _wap_expr(wap, good, "src_good").execute()

    assert out["passed"].iloc[0]
    assert out["published"].iloc[0]
    assert STAGING not in _refs(fresh_con), "staging branch should be removed"
    assert len(fresh_con.table(FINAL).execute()) == 4


def test_branch_wap_fail(fresh_con: Backend) -> None:
    wap = make_iceberg_wap_expr(fresh_con, FINAL)
    out = _wap_expr(wap, bad, "src_bad").execute()

    assert not out["passed"].iloc[0]
    assert not out["published"].iloc[0]
    assert STAGING in _refs(fresh_con), "rejected staging branch is retained"
    ice = fresh_con.catalog.load_table(f"{fresh_con.namespace}.{FINAL}")
    main_snap = ice.current_snapshot()
    assert main_snap is None or len(ice.scan().to_arrow()) == 0, (
        "failing audit must not publish data to main branch"
    )


def test_branch_wap_append(fresh_con: Backend) -> None:
    wap = make_iceberg_wap_expr(fresh_con, FINAL)
    _wap_expr(wap, good, "src1").execute()
    _wap_expr(wap, good, "src2").execute()

    assert len(fresh_con.table(FINAL).execute()) == 8


def test_table_wap_pass(fresh_con: Backend) -> None:
    wap = make_iceberg_wap_expr(fresh_con)
    out = _wap_expr(wap, good, "src_good").execute()

    assert out["passed"].iloc[0]
    assert out["published"].iloc[0]
    assert FINAL in fresh_con.list_tables()
    assert STAGING not in fresh_con.list_tables(), "staging table dropped after publish"
    assert len(fresh_con.table(FINAL).execute()) == 4


def test_table_wap_fail(fresh_con: Backend) -> None:
    wap = make_iceberg_wap_expr(fresh_con)
    out = _wap_expr(wap, bad, "src_bad").execute()

    assert not out["passed"].iloc[0]
    assert not out["published"].iloc[0]
    assert FINAL not in fresh_con.list_tables(), "failing audit must not publish"
    assert STAGING in fresh_con.list_tables(), "rejected staging table is retained"


def test_table_wap_append(fresh_con: Backend) -> None:
    wap = make_iceberg_wap_expr(fresh_con)
    _wap_expr(wap, good, "src1").execute()
    _wap_expr(wap, good, "src2").execute()

    assert len(fresh_con.table(FINAL).execute()) == 8


def test_table_wap_publish_does_not_purge_staging_files(fresh_con: Backend) -> None:
    # Regression: dropping staging must leave final's just-registered files intact.
    wap = make_iceberg_wap_expr(fresh_con)
    _wap_expr(wap, good, "src_good").execute()

    assert STAGING not in fresh_con.list_tables()
    published = fresh_con.table(FINAL).execute().sort_values("a")
    assert published["a"].tolist() == good["a"]
    assert published["b"].tolist() == good["b"]


def test_branch_wap_rerun_after_fail_raises(fresh_con: Backend) -> None:
    # A retained staging branch blocks an identical retry rather than appending.
    wap = make_iceberg_wap_expr(fresh_con, FINAL)
    _wap_expr(wap, bad, "src_bad").execute()
    assert STAGING in _refs(fresh_con)

    with pytest.raises(ValueError, match="already exists"):
        _wap_expr(wap, good, "src_retry").execute()


def test_iceberg_publish_guard_rejects_multi_row(fresh_con: Backend) -> None:
    # Publish expects the audit aggregate to collapse to exactly one row.
    publish = make_publish_with_iceberg(fresh_con)
    df = pd.DataFrame(
        {
            STAGING_COL: [STAGING, STAGING],
            FINAL_COL: [FINAL, FINAL],
            PASSED_COL: [True, True],
        }
    )
    with pytest.raises(ValueError, match="expected 1 row"):
        publish.fn(df)
