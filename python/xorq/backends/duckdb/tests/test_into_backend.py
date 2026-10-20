"""into_backend fan-out edge cases on duckdb.

The shared shapes live in ``xorq.backends.tests.into_backend_fanout``; the
tests at the end pin shapes where duckdb's exact scan-count contract is what
matters.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

import xorq.api as xo
from xorq.backends.tests.into_backend_fanout import *  # noqa: F403
from xorq.backends.tests.into_backend_fanout import reader_counts


@pytest.fixture
def target() -> xo.Backend:
    return xo.duckdb.connect()


def test_into_backend_asof_tolerance_single_scan(
    df: pd.DataFrame, target: xo.Backend
) -> None:
    # #983/#2086: the tolerance lowering is a null-out projection over a single
    # ASOF LEFT JOIN, so each input is scanned once. The old filter+re-join
    # lowering scanned the left twice, consuming one-shot readers and producing
    # empty results -- assert both the single scan and a non-empty result.
    left = pd.DataFrame(
        {"site": ["a", "b"], "ts": [datetime(2024, 1, 1), datetime(2024, 1, 2)]}
    )
    right = pd.DataFrame(
        {
            "site": ["a", "b"],
            "ev": ["x", "y"],
            "ts": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        }
    )
    lt = xo.memtable(left).into_backend(target, "l")
    rt = xo.memtable(right).into_backend(target, "r")
    expr = lt.asof_join(
        rt, on="ts", predicates="site", tolerance=timedelta(seconds=1)
    ).drop("ts_right")
    assert reader_counts(expr) == [1, 1]
    assert not expr.execute().empty


def test_into_backend_partition_window_unbounded(
    df: pd.DataFrame, target: xo.Backend
) -> None:
    # duckdb lowers ``sum(v) OVER (PARTITION BY k)`` to a GROUP BY self-join
    # that scans its input twice while the sqlglot AST shows one Table node,
    # so no count is derived and the cache is unbounded (no eviction, never a
    # cap); before, max_readers=1 and the second reader raised duckdb.Error:
    # Maximum number of readers reached. GH #2108.
    rt = xo.memtable(df).into_backend(target, "t")
    expr = rt.mutate(s=rt.v.sum().over(xo.window(group_by="k")))
    assert reader_counts(expr) == []
    assert expr.execute().sort_values("id").s.tolist() == [30, 30, 70, 70]
    # the same window with an ORDER BY is scanned once and counted: [] above
    # comes from the window rule, not from a compile that failed
    ordered = rt.mutate(s=rt.v.sum().over(xo.window(group_by="k", order_by="id")))
    assert reader_counts(ordered) == [1]


def test_into_backend_partition_window_unbounds_only_its_table(
    df: pd.DataFrame, target: xo.Backend
) -> None:
    # the window reads the left table through a CTE; the right table is
    # scanned once, as the AST says, and keeps its count (and its eviction)
    left = xo.memtable(df).into_backend(target, "l")
    right = xo.memtable(df[["id", "k"]]).into_backend(target, "r")
    windowed = left.mutate(s=left.v.sum().over(xo.window(group_by="k")))
    expr = windowed.join(right, "id").select("id", "s")
    assert reader_counts(expr) == [1]
    assert expr.execute().sort_values("id").s.tolist() == [30, 30, 70, 70]
