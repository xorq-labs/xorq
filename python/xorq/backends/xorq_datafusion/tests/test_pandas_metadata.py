"""Cross joins of pandas-sourced tables (xorq #2266).

``pa.Table.from_pandas`` attaches a ``{"pandas": ...}`` schema-level metadata
blob. DataFusion's schema equality is metadata-sensitive in the
``join_selection`` physical optimizer rule and in its logical/physical schema
verifier, so two tables *both* carrying the blob fail to join with an internal
"Schema mismatch" error. Registration drops the blob now.

``read_parquet`` needs no such handling even for a file pandas wrote: the blob
lives in the parquet key-value metadata and DataFusion infers the table schema
without it (see ``test_read_parquet_has_no_pandas_metadata``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

import xorq.api as xo
from xorq.backends.tests.pandas_metadata_util import PANDAS_SOURCES, engine_schema
from xorq.backends.xorq_datafusion import Backend


# create_table routes through a pandas conversion a one-shot reader does not
# survive -- unrelated to #2266.
CREATABLE_SOURCES = [p for p in PANDAS_SOURCES if "record-batch-reader" not in p.id]
# read_record_batches takes batch sources only, and any iterable of batches.
BATCH_SOURCES = [
    pytest.param(pa.Table.from_pandas, id="pyarrow-table"),
    pytest.param(lambda df: pa.Table.from_pandas(df).to_batches(), id="batch-list"),
    pytest.param(
        lambda df: pa.Table.from_pandas(df).to_reader(), id="record-batch-reader"
    ),
]


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


@pytest.mark.parametrize("to_source", PANDAS_SOURCES)
def test_cross_join_of_registered_pandas_sourced_tables(
    to_source: Callable[[pd.DataFrame], Any],
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
) -> None:
    con = xo.connect()
    con.register(to_source(left_df), "a")
    con.register(to_source(right_df), "b")

    assert cross_join_agg(con).to_dict("records") == [{"g": "y", "n": 1}]


@pytest.mark.parametrize("to_source", CREATABLE_SOURCES)
def test_cross_join_of_created_pandas_sourced_tables(
    to_source: Callable[[pd.DataFrame], Any],
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
) -> None:
    con = xo.connect()
    con.create_table("a", to_source(left_df))
    con.create_table("b", to_source(right_df))

    assert cross_join_agg(con).to_dict("records") == [{"g": "y", "n": 1}]


@pytest.mark.parametrize("to_source", BATCH_SOURCES)
def test_cross_join_of_read_record_batches(
    to_source: Callable[[pd.DataFrame], Any],
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
) -> None:
    con = xo.connect()
    con.read_record_batches(to_source(left_df), table_name="a")
    con.read_record_batches(to_source(right_df), table_name="b")

    # the registered readers are one-shot, so this is the only scan
    assert cross_join_agg(con).to_dict("records") == [{"g": "y", "n": 1}]


def test_cross_join_read_record_batches_against_registered_table(
    left_df: pd.DataFrame, right_df: pd.DataFrame
) -> None:
    """The registration paths must agree: one strips, the other must too."""
    con = xo.connect()
    con.read_record_batches(pa.Table.from_pandas(left_df), table_name="a")
    con.register(pa.Table.from_pandas(right_df), "b")

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


@pytest.mark.parametrize("to_source", PANDAS_SOURCES)
def test_registered_table_has_no_pandas_metadata(
    to_source: Callable[[pd.DataFrame], Any], left_df: pd.DataFrame
) -> None:
    con = xo.connect()
    con.register(to_source(left_df), "a")

    assert engine_schema(con, "a").metadata is None


@pytest.mark.parametrize("to_source", CREATABLE_SOURCES)
def test_created_table_has_no_pandas_metadata(
    to_source: Callable[[pd.DataFrame], Any], left_df: pd.DataFrame
) -> None:
    con = xo.connect()
    con.create_table("a", to_source(left_df))

    assert engine_schema(con, "a").metadata is None


@pytest.mark.parametrize("to_source", BATCH_SOURCES)
def test_read_record_batches_table_has_no_pandas_metadata(
    to_source: Callable[[pd.DataFrame], Any], left_df: pd.DataFrame
) -> None:
    con = xo.connect()
    con.read_record_batches(to_source(left_df), table_name="a")

    assert engine_schema(con, "a").metadata is None


def test_cross_join_of_registered_filesystem_datasets(
    left_df: pd.DataFrame, right_df: pd.DataFrame, tmp_path: Path
) -> None:
    """A parquet file pandas wrote carries the blob in the dataset schema."""

    def to_dataset(df: pd.DataFrame, name: str) -> ds.Dataset:
        directory = tmp_path / name
        directory.mkdir()
        pq.write_table(pa.Table.from_pandas(df), directory / "t.parquet")
        return ds.dataset(directory, format="parquet")

    con = xo.connect()
    con.register(to_dataset(left_df, "a"), "a")
    con.register(to_dataset(right_df, "b"), "b")

    assert engine_schema(con, "a").metadata is None
    assert engine_schema(con, "b").metadata is None
    assert cross_join_agg(con).to_dict("records") == [{"g": "y", "n": 1}]


def test_read_parquet_has_no_pandas_metadata(
    left_df: pd.DataFrame, tmp_path: Path
) -> None:
    path = tmp_path / "left.parquet"
    pq.write_table(pa.Table.from_pandas(left_df), path)

    con = xo.connect()
    con.read_parquet(path, table_name="a")

    assert engine_schema(con, "a").metadata is None
