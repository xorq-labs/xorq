from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest

import xorq.api as xo
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.backends.duckdb import Backend
from xorq.vendor import ibis
from xorq.vendor.ibis.expr import types as ir


Reader = Callable[[Backend, Path, "str | dt.DataType"], ir.Table]


def read_csv_columns(con: Backend, tmp_path: Path, typ: str | dt.DataType) -> ir.Table:
    path = tmp_path / "t.csv"
    path.write_text("a\n")
    return con.read_csv(path, columns={"a": typ})


def read_csv_types(con: Backend, tmp_path: Path, typ: str | dt.DataType) -> ir.Table:
    path = tmp_path / "t.csv"
    path.write_text("a\n")
    return con.read_csv(path, types={"a": typ})


def read_json_columns(con: Backend, tmp_path: Path, typ: str | dt.DataType) -> ir.Table:
    path = tmp_path / "t.json"
    path.write_text("")
    return con.read_json(path, columns={"a": typ})


readers = pytest.mark.parametrize(
    "reader",
    [
        pytest.param(read_csv_columns, id="csv-columns"),
        pytest.param(read_csv_types, id="csv-types"),
        pytest.param(read_json_columns, id="json-columns"),
    ],
)


def duckdb_type(con: Backend, t: ir.Table, column: str = "a") -> str:
    rows = con.con.sql(f'DESCRIBE "{t.get_name()}"').fetchall()
    ((actual,),) = ((typ,) for name, typ, *_ in rows if name == column)
    return actual


@readers
@pytest.mark.parametrize(
    "typ,expected_duckdb,expected_ibis",
    [
        pytest.param("int64", "BIGINT", "int64", id="ibis"),
        pytest.param(dt.int32, "INTEGER", "int32", id="datatype"),
        pytest.param("BIGINT", "BIGINT", "int64", id="duckdb"),
        pytest.param("TEXT", "VARCHAR", "string", id="duckdb-alias"),
        pytest.param("TIMETZ", "TIME WITH TIME ZONE", "time", id="TIMETZ"),
        pytest.param("DECIMAL(10,2)", "DECIMAL(10,2)", "decimal(10,2)", id="DECIMAL"),
        pytest.param("decimal(10,2)", "DECIMAL(10,2)", "decimal(10,2)", id="decimal"),
        pytest.param("VARCHAR[]", "VARCHAR[]", "array<string>", id="VARCHAR[]"),
        pytest.param("array<int64>", "BIGINT[]", "array<int64>", id="array<int64>"),
        pytest.param("TIMESTAMP_NS", "TIMESTAMP_NS", "timestamp(9)", id="TIMESTAMP_NS"),
        # a name valid in both keeps its ibis meaning
        pytest.param("INT", "BIGINT", "int64", id="INT-ibis"),
        pytest.param("FLOAT", "DOUBLE", "float64", id="FLOAT-ibis"),
        pytest.param("INT8", "TINYINT", "int8", id="INT8-ibis"),
        # `T[]` never parses as ibis, so its element keeps DuckDB's meaning
        pytest.param("int[]", "INTEGER[]", "array<int32>", id="int[]-duckdb"),
    ],
)
def test_read_type_reaches_duckdb(
    tmp_path: Path,
    reader: Reader,
    typ: str | dt.DataType,
    expected_duckdb: str,
    expected_ibis: str,
) -> None:
    con = xo.duckdb.connect()
    t = reader(con, tmp_path, typ)
    assert duckdb_type(con, t) == expected_duckdb
    assert t.schema()["a"] == ibis.dtype(expected_ibis)


@readers
@pytest.mark.parametrize(
    "typ",
    [
        pytest.param("NOT_A_TYPE", id="unknown"),
        pytest.param("decimal(a,b)", id="decimal(a,b)"),
        pytest.param("MAP(VARCHAR)", id="MAP(VARCHAR)"),
        pytest.param('a"b', id="unbalanced-quote"),
        pytest.param("BIT", id="BIT-unrepresentable"),
        pytest.param("BIT[]", id="BIT[]-nested-unknown"),
        pytest.param("STRUCT(x BIT)", id="STRUCT-nested-unknown"),
        pytest.param("MAP(VARCHAR, BIT)", id="MAP-nested-unknown"),
        pytest.param("timestamp(12)", id="timestamp(12)-invalid-scale"),
        pytest.param("GEOMETRY(FOO)", id="GEOMETRY(FOO)-unknown-subtype"),
        pytest.param("unknown", id="ibis-unknown-string"),
        pytest.param(dt.unknown, id="unknown-datatype"),
        pytest.param(dt.Array(dt.unknown), id="nested-unknown-datatype"),
    ],
)
def test_read_rejected_type(
    tmp_path: Path, reader: Reader, typ: str | dt.DataType
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"'a'.*{re.escape(repr(typ))}.*DuckDB type ibis can represent",
    ):
        reader(xo.duckdb.connect(), tmp_path, typ)


def test_read_csv_columns_executes(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    path.write_text("a,b\n1,x\n2,y\n")
    t = xo.duckdb.connect().read_csv(path, columns={"a": "BIGINT", "b": "TEXT"})
    assert t.schema() == ibis.schema({"a": "int64", "b": "string"})
    assert t.execute().a.tolist() == [1, 2]
