from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import xorq.vendor.ibis as ibis
from xorq.vendor.ibis.util import gen_name


if TYPE_CHECKING:
    from xorq.backends.bigquery import Backend
    from xorq.vendor.ibis.expr import types as ir


# the google client libraries are an optional (`--extra bigquery`) dependency
pytest.importorskip("google.cloud.bigquery")


@pytest.mark.bigquery
def test_can_connect(con: Backend) -> None:
    assert con.name == "bigquery"
    assert con.list_tables() is not None


@pytest.mark.bigquery
def test_raw_sql(con: Backend) -> None:
    result = list(con.raw_sql("SELECT 1 AS x"))
    assert result[0].x == 1


@pytest.mark.bigquery
def test_dataset_listed(con: Backend, dataset_id: str) -> None:
    assert dataset_id in con.list_databases()


@pytest.mark.bigquery
def test_create_and_drop_table(con: Backend) -> None:
    name = gen_name("xorq_gbq_table")
    schema = ibis.schema({"a": "int64", "b": "string"})
    con.create_table(name, schema=schema)
    try:
        assert name in con.list_tables()
        assert con.table(name).schema() == schema
    finally:
        con.drop_table(name, force=True)
    assert name not in con.list_tables()


@pytest.mark.bigquery
def test_create_table_from_expr(con: Backend, batting: ir.Table) -> None:
    name = gen_name("xorq_gbq_from_expr")
    expr = batting.filter(batting.yearID == 2015).select("playerID", "yearID")
    con.create_table(name, obj=expr)
    try:
        assert name in con.list_tables()
        assert not con.table(name).execute().empty
    finally:
        con.drop_table(name, force=True)


@pytest.mark.bigquery
def test_get_schema(con: Backend) -> None:
    # read_parquet lands the table in an anonymous session dataset, so use a
    # table created directly in the connection's default dataset
    name = gen_name("xorq_gbq_schema")
    schema = ibis.schema({"a": "int64", "b": "string"})
    con.create_table(name, schema=schema)
    try:
        assert con.get_schema(name) == schema
    finally:
        con.drop_table(name, force=True)
