from __future__ import annotations

import re
from pathlib import Path

import pytest

import xorq.api as xo
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.vendor import ibis


readers = pytest.mark.parametrize(
    "method,filename",
    [
        pytest.param("read_csv", "t.csv", id="csv"),
        pytest.param("read_json", "t.json", id="json"),
    ],
)


def read(con, tmp_path: Path, method: str, filename: str, typ):
    (path := tmp_path / filename).write_text("a\n" if filename.endswith("csv") else "")
    return getattr(con, method)(path, columns={"a": typ})


def duckdb_types(con, t) -> dict[str, str]:
    return {
        name: typ
        for name, typ, *_ in con.con.sql(f'DESCRIBE "{t.get_name()}"').fetchall()
    }


# one case per rule: ibis string, ibis DataType, DuckDB name, a DuckDB type ibis
# maps lossily (kept verbatim), a parametrized type (must reach DuckDB quoted),
# `T[]` vs `array<T>`, and a name valid in both that keeps its ibis meaning
@readers
@pytest.mark.parametrize(
    "typ,expected_duckdb,expected_ibis",
    [
        pytest.param("int64", "BIGINT", "int64", id="ibis"),
        pytest.param(dt.int32, "INTEGER", "int32", id="datatype"),
        pytest.param("BIGINT", "BIGINT", "int64", id="duckdb"),
        pytest.param("TIMETZ", "TIME WITH TIME ZONE", "time", id="TIMETZ-verbatim"),
        pytest.param("DECIMAL(10,2)", "DECIMAL(10,2)", "decimal(10,2)", id="DECIMAL"),
        pytest.param("int[]", "INTEGER[]", "array<int32>", id="int[]-duckdb"),
        pytest.param("array<int64>", "BIGINT[]", "array<int64>", id="array<int64>"),
        pytest.param("INT8", "TINYINT", "int8", id="INT8-ibis"),
    ],
)
def test_read_type_reaches_duckdb(
    tmp_path, method, filename, typ, expected_duckdb, expected_ibis
):
    con = xo.duckdb.connect()
    t = read(con, tmp_path, method, filename, typ)
    assert duckdb_types(con, t)["a"] == expected_duckdb
    assert t.schema()["a"] == ibis.dtype(expected_ibis)


# one case per rejection path: parse error, parser TypeError, empty string,
# unrepresentable nested, a subtype the mapper raises on, an `unknown` DataType
@readers
@pytest.mark.parametrize(
    "typ",
    [
        "NOT_A_TYPE",
        "timestamp('\\x')",
        "",
        "BIT[]",
        "GEOMETRY(FOO)",
        dt.unknown,
    ],
    ids=repr,
)
def test_read_rejected_type(tmp_path, method, filename, typ):
    with pytest.raises(
        ValueError,
        match=rf"'a'.*{re.escape(repr(typ))}.*DuckDB type ibis can represent",
    ):
        read(xo.duckdb.connect(), tmp_path, method, filename, typ)


def test_read_csv_columns_and_types(tmp_path):
    (path := tmp_path / "t.csv").write_text("a,b\n1,x\n2,y\n")
    con = xo.duckdb.connect()
    t = con.read_csv(path, columns={"a": "BIGINT", "b": "TEXT"})
    assert t.schema() == ibis.schema({"a": "int64", "b": "string"})
    assert t.execute().a.tolist() == [1, 2]
    t = con.read_csv(path, types={"a": "DECIMAL(10,2)"})
    assert duckdb_types(con, t) == {"a": "DECIMAL(10,2)", "b": "VARCHAR"}
