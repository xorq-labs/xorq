from __future__ import annotations

from typing import Callable

import pyarrow as pa
import pytest
from batchcorder import StreamCache

import xorq.api as xo


SCHEMA = pa.schema([("a", pa.int64())])


def make_reader(n: int = 3) -> pa.RecordBatchReader:
    batches = (pa.record_batch([[i]], schema=SCHEMA) for i in range(n))
    return pa.RecordBatchReader.from_batches(SCHEMA, batches)


def drained(n: int = 3) -> StreamCache:
    cache = StreamCache(make_reader(n))
    cache.ingest_all()
    return cache


@pytest.mark.parametrize(
    "make_source",
    (
        pytest.param(lambda: pa.table({"a": [0, 1, 2]}), id="table"),
        pytest.param(make_reader, id="reader"),
        pytest.param(drained, id="stream-cache"),
    ),
)
def test_read_record_batches_scans_repeatedly(
    make_source: Callable[[], pa.Table | pa.RecordBatchReader | StreamCache],
) -> None:
    con = xo.datafusion.connect()

    t = con.read_record_batches(make_source(), "t")

    assert [t.count().execute() for _ in range(2)] == [3, 3]
    assert con.table("t").a.sum().execute() == 3


def test_read_record_batches_replaces_name_and_generates_one() -> None:
    con = xo.datafusion.connect()
    con.read_record_batches(make_reader(3), "t")

    con.read_record_batches(make_reader(5), "t")
    generated = con.read_record_batches(make_reader(2))

    assert con.table("t").count().execute() == 5
    assert generated.count().execute() == 2
    assert generated.get_name() != "t"


def test_tee_into_the_parent_connection() -> None:
    con = xo.datafusion.connect()
    t = con.create_table("src", pa.table({"a": [1, 2, 3]}))

    rows = t.tee(con, table_name="copy").execute()

    assert len(rows) == 3
    assert con.table("copy").count().execute() == 3
