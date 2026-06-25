from __future__ import annotations

import pyarrow as pa

import xorq.api as xo
from xorq.expr.remote_table_exec import count_remote_table_readers


def test_into_backend_pandas_executes() -> None:
    pandas_con = xo.pandas.connect()
    t = xo.memtable(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}))
    expr = t.into_backend(pandas_con, "t")
    result = expr.execute()
    assert len(result) == 3
    assert sorted(result.columns) == ["a", "b"]
    assert list(result.sort_values("a")["a"]) == [1, 2, 3]


def test_into_backend_pandas_filter() -> None:
    pandas_con = xo.pandas.connect()
    t = xo.memtable(pa.table({"x": [10, 20, 30]}))
    expr = t.into_backend(pandas_con, "t").filter(xo._.x > 15)
    result = expr.execute()
    assert sorted(result["x"]) == [20, 30]


def test_into_backend_pandas_fanout_unbounded_cache() -> None:
    # pandas is a non-SQL backend: count_remote_table_readers can produce no
    # AST, so it returns an empty mapping and the StreamCache is built
    # unbounded (max_readers=None). This exercises that path and confirms a
    # fan-out shape (union over the same RemoteTable) still executes. A
    # self-join is not used: the pandas executor has no SelfReference rule.
    pandas_con = xo.pandas.connect()
    rt = xo.memtable(pa.table({"k": ["a", "b", "b"], "v": [10, 30, 40]})).into_backend(
        pandas_con, "t"
    )
    assert count_remote_table_readers(rt) == {}
    expr = rt.filter(rt.v < 20).union(rt.filter(rt.v > 35))
    assert sorted(expr.execute()["v"]) == [10, 40]
