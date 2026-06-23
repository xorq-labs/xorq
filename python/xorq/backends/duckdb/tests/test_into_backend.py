"""into_backend fan-out edge cases for the duckdb backend.

The duckdb backend scans the StreamCache built by
``register_and_transform_remote_tables`` directly, once per physical table
reference in the compiled SQL. ``max_readers`` must equal that scan count
exactly: an under-count crashes with "Maximum number of readers reached", an
over-count silently disables eviction. These tests pin the count for each
fan-out shape and assert the query still produces correct results.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

import xorq.api as xo
import xorq.vendor.ibis.expr.types as ir
from xorq.backends.duckdb import Backend
from xorq.expr.remote_table_exec import count_remote_table_readers
from xorq.tests.util import assert_frame_equal


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "k": ["a", "a", "b", "b"],
            "v": [10, 20, 30, 40],
        }
    )


@pytest.fixture
def target() -> Backend:
    return xo.duckdb.connect()


def reader_counts(expr: ir.Table) -> list[int]:
    return sorted(count_remote_table_readers(expr).values())


def test_into_backend_bare_single_reader(df: pd.DataFrame, target: Backend) -> None:
    expr = xo.memtable(df).into_backend(target, "t")
    assert reader_counts(expr) == [1]
    result = expr.execute().sort_values("id").reset_index(drop=True)
    assert_frame_equal(result, df, check_like=True)


def test_into_backend_filter_single_scan(df: pd.DataFrame, target: Backend) -> None:
    rt = xo.memtable(df).into_backend(target, "t")
    expr = rt.filter(rt.v > 15)
    assert reader_counts(expr) == [1]
    assert sorted(expr.execute()["v"]) == [20, 30, 40]


def test_into_backend_many_columns_one_scan(df: pd.DataFrame, target: Backend) -> None:
    # several column refs over one table resolve within a single scan
    rt = xo.memtable(df).into_backend(target, "t")
    expr = rt.select(s=rt.id + rt.v, d=rt.v - rt.id)
    assert reader_counts(expr) == [1]
    assert not expr.execute().empty


def test_into_backend_self_join_two_readers(df: pd.DataFrame, target: Backend) -> None:
    rt = xo.memtable(df).into_backend(target, "t")
    expr = rt.join(rt.view(), "k")
    assert reader_counts(expr) == [2]
    # k='a' -> 2x2, k='b' -> 2x2
    assert len(expr.execute()) == 8


def test_into_backend_two_scalar_subqueries_two_readers(
    df: pd.DataFrame, target: Backend
) -> None:
    # the StreamCache deadlock shape: two concurrent scans of one cache
    rt = xo.memtable(df).into_backend(target, "t")
    expr = rt.v.sum().as_scalar().as_table().mutate(cnt=rt.v.count().as_scalar())
    assert reader_counts(expr) == [2]
    result = expr.execute()
    assert int(result.iloc[0, 0]) == 100
    assert int(result["cnt"].iloc[0]) == 4


def test_into_backend_threeway_fanout_three_readers(
    df: pd.DataFrame, target: Backend
) -> None:
    rt = xo.memtable(df).into_backend(target, "t")
    expr = (
        rt.filter(rt.v < 15).union(rt.filter(rt.v > 35)).union(rt.filter(rt.k == "b"))
    )
    assert reader_counts(expr) == [3]
    assert sorted(expr.execute()["v"]) == [10, 30, 40, 40]


def test_into_backend_asof_tolerance_single_scan(
    df: pd.DataFrame, target: Backend
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


def test_into_backend_self_join_limit_early_termination(
    df: pd.DataFrame, target: Backend
) -> None:
    # fan-out (2 readers) + limit: the scan stops before exhausting the cache
    rt = xo.memtable(df).into_backend(target, "t")
    expr = rt.join(rt.view(), "k").limit(2)
    assert reader_counts(expr) == [2]
    assert len(expr.execute()) == 2


def test_into_backend_two_distinct_tables(df: pd.DataFrame, target: Backend) -> None:
    left = xo.memtable(df).into_backend(target, "l")
    right = xo.memtable(df.assign(v=df.v * 2)).into_backend(target, "r")
    expr = left.join(right, "k")
    counts = count_remote_table_readers(expr)
    assert len(counts) == 2
    assert sorted(counts.values()) == [1, 1]
    assert len(expr.execute()) == 8


def test_into_backend_empty_table_fanout(target: Backend) -> None:
    empty = pd.DataFrame(
        {"id": pd.Series([], dtype="int64"), "v": pd.Series([], dtype="int64")}
    )
    rt = xo.memtable(empty).into_backend(target, "e")
    expr = rt.join(rt.view(), "id")
    assert reader_counts(expr) == [2]
    assert expr.execute().empty


def test_into_backend_nested_chain(df: pd.DataFrame, target: Backend) -> None:
    # data flows memtable -> datafusion -> duckdb, then fans out on duckdb
    inner = xo.memtable(df).into_backend(xo.connect(), "inner")
    rt = inner.into_backend(target, "outer")
    expr = rt.join(rt.view(), "id")
    # only the outer RemoteTable is resolved in this pass (inner is recursive)
    assert reader_counts(expr) == [2]
    assert len(expr.execute()) == 4
