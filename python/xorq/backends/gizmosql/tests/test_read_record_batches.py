"""GizmoSQL read_record_batches coverage, including StreamCache inputs.

The remote pass (REMOTE_PASS) feeds a batchcorder StreamCache straight
into the target backend's read_record_batches, so these pin that gizmosql
accepts a StreamCache (it exposes __iter__/.schema) and, in particular, that an
empty stream still materializes the declared columns rather than raising
"Must pass schema, or at least one RecordBatch".
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest
from batchcorder import StreamCache

import xorq.api as xo
from xorq.vendor.ibis import _


pytestmark = pytest.mark.gizmosql


def _stream_cache(schema: pa.Schema, batches: list[pa.RecordBatch]) -> StreamCache:
    reader = pa.RecordBatchReader.from_batches(schema, iter(batches))
    return StreamCache(reader)


def test_read_record_batches_stream_cache(con: Any, temp_table: str) -> None:
    """A non-empty StreamCache ingests every row."""
    schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
    batches = [pa.record_batch({"a": [i], "b": [str(i)]}) for i in range(3)]
    cache = _stream_cache(schema, batches)

    t = con.read_record_batches(cache, table_name=temp_table)
    result = t.execute()

    assert len(result) == 3
    assert sorted(result.columns) == ["a", "b"]
    assert sorted(result["a"]) == [0, 1, 2]


def test_read_record_batches_empty_stream_cache(con: Any, temp_table: str) -> None:
    """Regression: an empty StreamCache creates the table with zero rows.

    Previously read_record_batches called pa.Table.from_batches(source) with no
    schema, so a zero-batch stream raised ValueError before the table could be
    created. Passing schema=source.schema keeps the declared columns.
    """
    schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
    cache = _stream_cache(schema, [])

    t = con.read_record_batches(cache, table_name=temp_table)
    result = t.execute()

    assert len(result) == 0
    assert sorted(result.columns) == ["a", "b"]


def test_into_backend_empty_result_to_gizmosql(con: Any) -> None:
    """End-to-end: an upstream that filters to zero rows still lands as an empty
    table on gizmosql, exercising the empty-StreamCache path through the real
    RemoteTable pipeline."""
    ddb_con = xo.duckdb.connect()
    local = ddb_con.create_table(
        "empty_src", pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    )
    expr = local.filter(_.a > 100).into_backend(con, "empty_gz")
    result = expr.execute()

    assert len(result) == 0
    assert sorted(result.columns) == ["a", "b"]
