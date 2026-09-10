"""Cross joins of pandas-sourced tables on the datafusion backend (xorq #2266).

See ``xorq/backends/xorq_datafusion/tests/test_pandas_metadata.py`` for the
description of the metadata-sensitive schema equality this guards against.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pytest

import xorq.api as xo
from xorq.backends.datafusion import Backend


PANDAS_SOURCES = [
    pytest.param(lambda df: df, id="pandas"),
    pytest.param(pa.Table.from_pandas, id="pyarrow-table"),
    pytest.param(
        lambda df: pa.Table.from_pandas(df).to_batches()[0], id="record-batch"
    ),
    pytest.param(
        lambda df: pa.Table.from_pandas(df).to_reader(), id="record-batch-reader"
    ),
    pytest.param(lambda df: ds.dataset(pa.Table.from_pandas(df)), id="dataset"),
]


def engine_schema(con: Backend, name: str) -> pa.Schema:
    """The schema DataFusion itself holds for ``name``.

    Not ``con.table(name).to_pyarrow().schema``: that rebuilds the result from
    the ibis schema, which never carries metadata, so it reports none whether
    or not registration dropped the blob.
    """
    return con.con.sql(f'SELECT * FROM "{name}"').schema()


@pytest.mark.parametrize("to_source", PANDAS_SOURCES)
def test_cross_join_of_pandas_sourced_tables(
    to_source: Callable[[pd.DataFrame], Any],
) -> None:
    con = xo.datafusion.connect()
    con.create_table("a", to_source(pd.DataFrame({"k": ["x"], "v": [1]})))
    con.create_table("b", to_source(pd.DataFrame({"g": ["y"]})))

    actual = (
        con.table("a")
        .cross_join(con.table("b"))
        .group_by("g")
        .aggregate(n=xo._.v.sum())
        .execute()
    )
    assert actual.to_dict("records") == [{"g": "y", "n": 1}]


@pytest.mark.parametrize("to_source", PANDAS_SOURCES)
def test_registered_table_has_no_pandas_metadata(
    to_source: Callable[[pd.DataFrame], Any],
) -> None:
    con = xo.datafusion.connect()
    con.create_table("a", to_source(pd.DataFrame({"k": ["x"], "v": [1]})))

    assert engine_schema(con, "a").metadata is None
