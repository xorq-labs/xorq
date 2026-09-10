"""Shared fixtures for the per-backend pandas-metadata tests (xorq #2266)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pytest


def _categorize(df: pd.DataFrame) -> pd.DataFrame:
    """Retype the string columns as ``pd.Categorical``.

    ``from_pandas`` renders those as Arrow dictionary columns, so the source
    carries the blob *and* a type whose handling is less trivial than a plain
    metadata swap (dictionary has no self-cast kernel in every Arrow release).
    """
    return df.astype(dict.fromkeys(df.select_dtypes("object"), "category"))


PANDAS_SOURCES = [
    pytest.param(lambda df: df, id="pandas"),
    pytest.param(pa.Table.from_pandas, id="pyarrow-table"),
    pytest.param(
        lambda df: pa.Table.from_pandas(df).to_batches()[0], id="record-batch"
    ),
    pytest.param(
        lambda df: pa.Table.from_pandas(df).to_reader(), id="record-batch-reader"
    ),
    pytest.param(
        lambda df: pa.Table.from_pandas(_categorize(df)).to_reader(),
        id="categorical-record-batch-reader",
    ),
    pytest.param(lambda df: ds.dataset(pa.Table.from_pandas(df)), id="dataset"),
]


def engine_schema(con: Any, name: str) -> pa.Schema:
    """The schema DataFusion itself holds for ``name``.

    Not ``con.table(name).to_pyarrow().schema``: that rebuilds the result from
    the ibis schema, which never carries metadata, so it reports none whether
    or not registration dropped the blob.
    """
    return con.con.sql(f'SELECT * FROM "{name}"').schema()
