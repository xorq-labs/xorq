from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
from pytest import param

import xorq.api as xo
import xorq.vendor.ibis.expr.operations as ops
from xorq.vendor import ibis


def test_attach_file(tmp_path):
    dbpath = str(tmp_path / "attached.db")
    path_client = xo.sqlite.connect(dbpath)
    path_client.create_table("test", schema=ibis.schema(dict(a="int")))

    client = xo.sqlite.connect()

    assert not client.list_tables()

    client.attach("baz", Path(dbpath))
    client.attach("bar", dbpath)

    foo_tables = client.list_tables(database="baz")
    bar_tables = client.list_tables(database="bar")

    assert foo_tables == ["test"]
    assert foo_tables == bar_tables


@pytest.mark.parametrize(
    "url, ext",
    [
        param(lambda p: p, "sqlite", id="no-scheme-sqlite-ext"),
        param(lambda p: p, "db", id="no-scheme-db-ext"),
        param(lambda p: p, "db", id="absolute-path"),
        param(
            lambda p: f"{os.path.relpath(p)}",
            "db",
            id="relative-path",
        ),
        param(lambda _: "", "db", id="in-memory-empty"),
        param(lambda _: ":memory:", "db", id="in-memory-explicit"),
    ],
)
def test_connect(url, ext, tmp_path):
    path = url(os.path.abspath(tmp_path / f"test.{ext}"))
    with sqlite3.connect(path):
        pass
    con = xo.sqlite.connect(path)
    one = xo.literal(1)
    assert con.execute(one) == 1


@pytest.mark.parametrize(
    "op",
    (
        ops.Project,  # Core operations handled in non-standard ways
        ops.Filter,  # Core operations handled in non-standard ways
        ops.Sort,  # Core operations handled in non-standard ways
        ops.Aggregate,  # Core operations handled in non-standard ways
        ops.Capitalize,  # Handled by base class rewrite
        ops.Sample,  # Handled by compiler-specific rewrite
        ops.Cast,  # Handled by visit_* method
    ),
)
def test_has_operation(sqlite_con, op):
    assert sqlite_con.has_operation(op)


def test_list_temp_tables_by_default(sqlite_con):
    name = ibis.util.gen_name("sqlite_temp_table")
    sqlite_con.create_table(name, schema={"a": "int"}, temp=True)
    assert name in sqlite_con.list_tables(database="temp")
    assert name in sqlite_con.list_tables()
