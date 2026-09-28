"""Shared fixtures for the per-backend pandas-metadata tests (xorq #2266)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pytest


def _categorical_reader(df: pd.DataFrame) -> pa.RecordBatchReader:
    """Retype the string columns as ``pd.Categorical``, then read as Arrow.

    ``from_pandas`` renders those as Arrow dictionary columns, so the source
    carries the blob *and* a type whose handling is less trivial than a plain
    metadata swap (dictionary may lack a self-cast kernel in some Arrow
    releases). ``string`` joins ``object`` because pandas renders string
    columns as either, depending on ``future.infer_string``.
    """
    columns = df.select_dtypes(include=["object", "string"]).columns
    table = pa.Table.from_pandas(df.astype(dict.fromkeys(columns, "category")))
    assert any(pa.types.is_dictionary(field.type) for field in table.schema), (
        f"no dictionary column in {table.schema}"
    )
    return table.to_reader()


PANDAS_SOURCES = [
    pytest.param(lambda df: df, id="pandas"),
    pytest.param(pa.Table.from_pandas, id="pyarrow-table"),
    pytest.param(
        lambda df: pa.Table.from_pandas(df).to_batches()[0], id="record-batch"
    ),
    pytest.param(
        lambda df: pa.Table.from_pandas(df).to_reader(), id="record-batch-reader"
    ),
    pytest.param(_categorical_reader, id="categorical-record-batch-reader"),
    pytest.param(lambda df: ds.dataset(pa.Table.from_pandas(df)), id="dataset"),
]


def engine_schema(con: Any, name: str) -> pa.Schema:
    """The schema DataFusion itself holds for ``name``.

    Not ``con.table(name).to_pyarrow().schema``: that rebuilds the result from
    the ibis schema, which never carries metadata, so it reports none whether
    or not registration dropped the blob.
    """
    return con.con.sql(f'SELECT * FROM "{name}"').schema()
