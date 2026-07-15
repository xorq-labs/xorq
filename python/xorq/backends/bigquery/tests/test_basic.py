from __future__ import annotations

from operator import methodcaller
from typing import TYPE_CHECKING

import pytest

import xorq.api as xo
from xorq.common.utils.dasher import tokenize
from xorq.tests.util import assert_frame_equal, check_eq


if TYPE_CHECKING:
    from xorq.backends.bigquery import Backend
    from xorq.vendor.ibis.expr import types as ir


# the google client libraries are an optional (`--extra bigquery`) dependency
pytest.importorskip("google.cloud.bigquery")


@pytest.mark.bigquery
def test_read_record_batches(con: Backend, batting: ir.Table) -> None:
    reader = (
        batting.filter(batting.yearID == 2015)
        .select("playerID", "yearID", "G")
        .to_pyarrow_batches()
    )
    name = "record_batches_batting"
    t = con.read_record_batches(reader, table_name=name)
    try:
        assert name in con.list_tables()
        result = t.execute()
        assert not result.empty
        assert list(result.columns) == ["playerID", "yearID", "G"]
    finally:
        con.drop_table(name, force=True)


@pytest.mark.bigquery
def test_filter_select(batting: ir.Table) -> None:
    result = batting.filter(batting.yearID == 2015).select("playerID").execute()
    assert not result.empty
    assert list(result.columns) == ["playerID"]


@pytest.mark.bigquery
def test_aggregate(batting: ir.Table) -> None:
    expr = batting.group_by("yearID").agg(total_g=batting.G.sum())
    result = expr.execute()
    assert not result.empty
    assert set(result.columns) == {"yearID", "total_g"}


@pytest.mark.bigquery
@pytest.mark.parametrize(
    "collect",
    (
        pytest.param("to_pyarrow", id="to_pyarrow"),
        pytest.param("to_pyarrow_batches", id="to_pyarrow_batches"),
        pytest.param("execute", id="execute"),
    ),
)
def test_can_collect(batting: ir.Table, collect: str) -> None:
    expr = (
        batting.filter(batting.yearID == 2015)
        .select("playerID", "yearID", "G")
        .mutate(add_1=batting.G + 1)
    )
    assert methodcaller(collect)(expr) is not None


@pytest.mark.bigquery
def test_join(batting: ir.Table, awards_players: ir.Table) -> None:
    def make_right(t):
        return t[t.lgID == "NL"].drop("yearID", "lgID")

    left = batting[batting.yearID == 2015]
    predicate = ["playerID"]
    result_order = ["playerID", "yearID", "lgID", "stint"]

    expr = left.join(make_right(awards_players), predicate, how="inner")
    result = (
        expr.execute()[left.columns].sort_values(result_order).reset_index(drop=True)
    )

    expected = (
        check_eq(
            left.execute(),
            make_right(awards_players).execute(),
            how="inner",
            on=predicate,
            suffixes=("_x", "_y"),
        )[left.columns]
        .sort_values(result_order)
        .reset_index(drop=True)
    )

    assert_frame_equal(result, expected, check_like=True)


@pytest.mark.bigquery
def test_sql(con: Backend, batting: ir.Table) -> None:
    expr = batting.filter(batting.yearID == 2015).select("playerID")
    query = xo.to_sql(expr)
    result = con.sql(query).execute()
    assert not result.empty


@pytest.mark.bigquery
def test_train_test_split(batting: ir.Table) -> None:
    (train, test) = xo.train_test_splits(batting, 0.2)
    assert train.execute() is not None
    assert test.execute() is not None


@pytest.mark.bigquery
def test_into_backend_duckdb(batting: ir.Table) -> None:
    ddb = xo.duckdb.connect()
    expr = (
        batting.filter(batting.yearID == 2015)
        .select("playerID", "yearID", "G")
        .into_backend(ddb, name="ddb_batting")
    )
    assert tokenize(expr) is not None
    assert not expr.execute().empty
