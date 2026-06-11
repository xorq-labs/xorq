"""into_backend fan-out edge cases for the xorq_datafusion backend.

Unlike duckdb, the datafusion backend reads the StreamCache built by
``register_and_transform_remote_tables`` exactly once (``from_stream``) and
re-wraps it into an inner cache that DataFusion scans per reference. So the
outer cache's ``max_readers`` (the value computed here) only ever sees one
reader — an over-count merely disables eviction, never crashes. These tests
confirm the reader count is computed correctly and, more importantly, that
fan-out queries execute correctly and without the GIL/mutex deadlock that
concurrent StreamCache scans previously triggered.
"""

import pandas as pd
import pytest

import xorq.api as xo
from xorq.expr.relations import count_remote_table_readers
from xorq.tests.util import assert_frame_equal


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "k": ["a", "a", "b", "b"],
            "v": [10, 20, 30, 40],
        }
    )


@pytest.fixture
def target():
    return xo.connect()


def reader_counts(expr):
    return sorted(count_remote_table_readers(expr).values())


def test_into_backend_bare_single_reader(df, target):
    expr = xo.memtable(df).into_backend(target, "t")
    assert reader_counts(expr) == [1]
    result = expr.execute().sort_values("id").reset_index(drop=True)
    assert_frame_equal(result, df, check_like=True)


def test_into_backend_filter_single_scan(df, target):
    rt = xo.memtable(df).into_backend(target, "t")
    expr = rt.filter(rt.v > 15)
    assert reader_counts(expr) == [1]
    assert sorted(expr.execute()["v"]) == [20, 30, 40]


def test_into_backend_many_columns_one_scan(df, target):
    rt = xo.memtable(df).into_backend(target, "t")
    expr = rt.select(s=rt.id + rt.v, d=rt.v - rt.id)
    assert reader_counts(expr) == [1]
    assert not expr.execute().empty


def test_into_backend_self_join_two_readers(df, target):
    rt = xo.memtable(df).into_backend(target, "t")
    expr = rt.join(rt.view(), "k")
    assert reader_counts(expr) == [2]
    assert len(expr.execute()) == 8


def test_into_backend_two_scalar_subqueries_no_deadlock(df, target):
    # two concurrent scans of one cache: the GIL/mutex deadlock regression
    rt = xo.memtable(df).into_backend(target, "t")
    expr = rt.v.sum().as_scalar().as_table().mutate(cnt=rt.v.count().as_scalar())
    assert reader_counts(expr) == [2]
    result = expr.execute()
    assert int(result.iloc[0, 0]) == 100
    assert int(result["cnt"].iloc[0]) == 4


def test_into_backend_threeway_fanout_three_readers(df, target):
    rt = xo.memtable(df).into_backend(target, "t")
    expr = (
        rt.filter(rt.v < 15).union(rt.filter(rt.v > 35)).union(rt.filter(rt.k == "b"))
    )
    assert reader_counts(expr) == [3]
    assert sorted(expr.execute()["v"]) == [10, 30, 40, 40]


def test_into_backend_self_join_limit_early_termination(df, target):
    # fan-out (2 readers) + limit: the scan stops before exhausting the cache
    rt = xo.memtable(df).into_backend(target, "t")
    expr = rt.join(rt.view(), "k").limit(2)
    assert reader_counts(expr) == [2]
    assert len(expr.execute()) == 2


def test_into_backend_two_distinct_tables(df, target):
    left = xo.memtable(df).into_backend(target, "l")
    right = xo.memtable(df.assign(v=df.v * 2)).into_backend(target, "r")
    expr = left.join(right, "k")
    counts = count_remote_table_readers(expr)
    assert len(counts) == 2
    assert sorted(counts.values()) == [1, 1]
    assert len(expr.execute()) == 8


def test_into_backend_empty_table_fanout(target):
    empty = pd.DataFrame(
        {"id": pd.Series([], dtype="int64"), "v": pd.Series([], dtype="int64")}
    )
    rt = xo.memtable(empty).into_backend(target, "e")
    expr = rt.join(rt.view(), "id")
    assert reader_counts(expr) == [2]
    assert expr.execute().empty


def test_into_backend_nested_chain(df, target):
    # data flows memtable -> duckdb -> datafusion, then fans out on datafusion
    inner = xo.memtable(df).into_backend(xo.duckdb.connect(), "inner")
    rt = inner.into_backend(target, "outer")
    expr = rt.join(rt.view(), "id")
    assert reader_counts(expr) == [2]
    assert len(expr.execute()) == 4
