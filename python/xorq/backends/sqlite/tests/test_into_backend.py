from __future__ import annotations

import pyarrow as pa

import xorq.api as xo


def test_into_backend_sqlite_executes() -> None:
    sqlite_con = xo.sqlite.connect(":memory:")
    t = xo.memtable(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}))
    expr = t.into_backend(sqlite_con, "t")
    result = expr.execute()
    assert len(result) == 3
    assert sorted(result.columns) == ["a", "b"]
    assert list(result.sort_values("a")["a"]) == [1, 2, 3]


def test_into_backend_sqlite_filter() -> None:
    sqlite_con = xo.sqlite.connect(":memory:")
    t = xo.memtable(pa.table({"x": [10, 20, 30]}))
    expr = t.into_backend(sqlite_con, "t").filter(xo._.x > 15)
    result = expr.execute()
    assert sorted(result["x"]) == [20, 30]
