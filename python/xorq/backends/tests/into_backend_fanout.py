"""into_backend fan-out shapes shared by the duckdb and xorq_datafusion suites.

Each shape pins the reader count the remote pass computes from the compiled
SQL and checks the query still executes correctly. The importing test module
supplies the ``target`` fixture. This file has no ``test_`` prefix, so pytest
collects the suite only through the per-backend modules that import it.
"""

from __future__ import annotations

import pandas as pd
import pytest

import xorq.api as xo
from xorq.expr.remote_table_exec import count_remote_table_readers
from xorq.tests.util import assert_frame_equal


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {"id": [1, 2, 3, 4], "k": ["a", "a", "b", "b"], "v": [10, 20, 30, 40]}
    )


def reader_counts(expr) -> list[int]:
    return sorted(count_remote_table_readers(expr).values())


# the count a PARTITION BY-only window yields per backend: none on duckdb,
# whose optimiser re-scans the window's input, and none on pandas, which
# compiles no SQL; one scan everywhere else
PARTITION_WINDOW_COUNTS = {"duckdb": [], "pandas": []}


def test_into_backend_partition_window_count(df: pd.DataFrame, target) -> None:
    rt = xo.memtable(df).into_backend(target, "t")
    expr = rt.mutate(s=rt.v.sum().over(xo.window(group_by="k")))
    assert reader_counts(expr) == PARTITION_WINDOW_COUNTS.get(target.name, [1])
    assert expr.execute().sort_values("id").s.tolist() == [30, 30, 70, 70]


def _same_rows(got: pd.DataFrame, df: pd.DataFrame) -> bool:
    assert_frame_equal(
        got.sort_values("id").reset_index(drop=True), df, check_like=True
    )
    return True


# (shape over one remote table `rt`, expected reader counts, check on the result)
SHAPES = [
    pytest.param(lambda rt: rt, [1], _same_rows, id="bare"),
    pytest.param(
        lambda rt: rt.filter(rt.v > 15),
        [1],
        lambda got, df: sorted(got["v"]) == [20, 30, 40],
        id="filter",
    ),
    pytest.param(
        lambda rt: rt.select(s=rt.id + rt.v, d=rt.v - rt.id),
        [1],
        lambda got, df: not got.empty,
        id="many-columns-one-scan",
    ),
    pytest.param(
        lambda rt: rt.join(rt.view(), "k"),
        [2],
        lambda got, df: len(got) == 8,  # k='a' -> 2x2, k='b' -> 2x2
        id="self-join",
    ),
    pytest.param(
        # two concurrent scans of one cache: the deadlock shape on datafusion
        lambda rt: (
            rt.v.sum().as_scalar().as_table().mutate(cnt=rt.v.count().as_scalar())
        ),
        [2],
        lambda got, df: (int(got.iloc[0, 0]), int(got["cnt"].iloc[0])) == (100, 4),
        id="two-scalar-subqueries",
    ),
    pytest.param(
        lambda rt: (
            rt.filter(rt.v < 15)
            .union(rt.filter(rt.v > 35))
            .union(rt.filter(rt.k == "b"))
        ),
        [3],
        lambda got, df: sorted(got["v"]) == [10, 30, 40, 40],
        id="threeway-union",
    ),
    pytest.param(
        # fan-out plus limit: the scan stops before exhausting the cache
        lambda rt: rt.join(rt.view(), "k").limit(2),
        [2],
        lambda got, df: len(got) == 2,
        id="self-join-limit",
    ),
]


@pytest.mark.parametrize(("shape", "counts", "check"), SHAPES)
def test_into_backend_fanout_shape(df, target, shape, counts, check) -> None:
    expr = shape(xo.memtable(df).into_backend(target, "t"))
    assert reader_counts(expr) == counts
    assert check(expr.execute(), df)


def test_into_backend_two_distinct_tables(df, target) -> None:
    left = xo.memtable(df).into_backend(target, "l")
    right = xo.memtable(df.assign(v=df.v * 2)).into_backend(target, "r")
    expr = left.join(right, "k")
    assert reader_counts(expr) == [1, 1]
    assert len(expr.execute()) == 8


def test_into_backend_empty_table_fanout(target) -> None:
    empty = pd.DataFrame(
        {"id": pd.Series([], dtype="int64"), "v": pd.Series([], dtype="int64")}
    )
    rt = xo.memtable(empty).into_backend(target, "e")
    expr = rt.join(rt.view(), "id")
    assert reader_counts(expr) == [2]
    assert expr.execute().empty


def test_into_backend_nested_chain(df, target) -> None:
    # memtable -> the other backend -> target, then fans out on target; only the
    # outer RemoteTable is resolved in this pass
    other = xo.connect() if target.name == "duckdb" else xo.duckdb.connect()
    rt = xo.memtable(df).into_backend(other, "inner").into_backend(target, "outer")
    expr = rt.join(rt.view(), "id")
    assert reader_counts(expr) == [2]
    assert len(expr.execute()) == 4


__all__ = ["df", "reader_counts", *(name for name in dir() if name.startswith("test_"))]
