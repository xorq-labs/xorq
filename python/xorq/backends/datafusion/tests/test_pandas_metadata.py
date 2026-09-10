"""Cross joins of pandas-sourced tables on the datafusion backend (xorq #2266).

See ``xorq/backends/xorq_datafusion/tests/test_pandas_metadata.py`` for the
description of the metadata-sensitive schema equality this guards against.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import pyarrow as pa
import pytest

import xorq.api as xo


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


def test_registered_table_has_no_pandas_metadata() -> None:
    con = xo.datafusion.connect()
    table = con.create_table("a", pd.DataFrame({"k": ["x"], "v": [1]}))

    assert table.to_pyarrow().schema.metadata is None
