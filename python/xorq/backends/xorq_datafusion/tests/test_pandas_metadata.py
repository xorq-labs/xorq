"""Cross joins of pandas-sourced tables (xorq #2266).

``pa.Table.from_pandas`` attaches a ``{"pandas": ...}`` schema-level metadata
blob. DataFusion's schema equality is metadata-sensitive in the
``join_selection`` physical optimizer rule and in its logical/physical schema
verifier, so two tables carrying *different* blobs used to fail to join with an
internal "Schema mismatch" error. Registration drops the blob now.
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

import xorq.api as xo
from xorq.backends.xorq_datafusion import Backend


@pytest.fixture
def left_df() -> pd.DataFrame:
    return pd.DataFrame({"k": ["x"], "v": [1]})


@pytest.fixture
def right_df() -> pd.DataFrame:
    return pd.DataFrame({"g": ["y"]})


def cross_join_agg(con: Backend) -> pd.DataFrame:
    return (
        con.table("a")
        .cross_join(con.table("b"))
        .group_by("g")
        .aggregate(n=xo._.v.sum())
        .execute()
    )


@pytest.mark.parametrize(
    "to_source",
    [
        pytest.param(lambda df: df, id="pandas"),
        pytest.param(pa.Table.from_pandas, id="pyarrow-table"),
        pytest.param(
            lambda df: pa.Table.from_pandas(df).to_batches()[0], id="record-batch"
        ),
        pytest.param(
            lambda df: pa.Table.from_pandas(df).to_reader(), id="record-batch-reader"
        ),
    ],
)
def test_cross_join_of_registered_pandas_sourced_tables(
    to_source: callable, left_df: pd.DataFrame, right_df: pd.DataFrame
) -> None:
    con = xo.connect()
    con.register(to_source(left_df), "a")
    con.register(to_source(right_df), "b")

    assert cross_join_agg(con).to_dict("records") == [{"g": "y", "n": 1}]


@pytest.mark.parametrize(
    "to_source",
    [
        pytest.param(lambda df: df, id="pandas"),
        pytest.param(pa.Table.from_pandas, id="pyarrow-table"),
    ],
)
def test_cross_join_of_created_pandas_sourced_tables(
    to_source: callable, left_df: pd.DataFrame, right_df: pd.DataFrame
) -> None:
    con = xo.connect()
    con.create_table("a", to_source(left_df))
    con.create_table("b", to_source(right_df))

    assert cross_join_agg(con).to_dict("records") == [{"g": "y", "n": 1}]


def test_cross_join_raw_sql(left_df: pd.DataFrame, right_df: pd.DataFrame) -> None:
    con = xo.connect()
    con.create_table("a", left_df)
    con.create_table("b", right_df)

    expected = pd.DataFrame({"k": ["x"], "v": [1], "g": ["y"]})
    actual = con.sql('SELECT * FROM "a" CROSS JOIN "b"').execute()
    assert actual.to_dict("records") == expected.to_dict("records")


def test_cross_join_of_memtables() -> None:
    a = xo.memtable({"k": ["x", "y"], "v": [1, 2]})
    b = xo.memtable({"g": ["a", "b"]})

    actual = a.cross_join(b).group_by("g").aggregate(n=xo._.v.sum()).execute()
    assert sorted(actual.to_dict("records"), key=lambda d: d["g"]) == [
        {"g": "a", "n": 3},
        {"g": "b", "n": 3},
    ]


def test_registered_table_has_no_pandas_metadata(left_df: pd.DataFrame) -> None:
    con = xo.connect()
    table = con.create_table("a", left_df)

    assert table.to_pyarrow().schema.metadata is None
