from __future__ import annotations

from typing import Callable

import datafusion
import pandas as pd
import pyarrow as pa
import pytest
from batchcorder import StreamCache
from batchcorder_datafusion import TeeTableProvider

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


def test_read_record_batches_tee_provider_reads_upstream_once() -> None:
    con = xo.datafusion.connect()
    pulled = []
    batches = (
        pulled.append(i) or pa.record_batch([[i]], schema=SCHEMA) for i in range(3)
    )
    reader = pa.RecordBatchReader.from_batches(SCHEMA, batches)

    t = con.read_record_batches(TeeTableProvider(reader), "t")

    assert len(t.join(t.view(), "a").execute()) == 3
    assert pulled == [0, 1, 2]
    con.drop_table("t")
    assert con.list_tables() == []


def test_drop_table_keeps_a_table_differing_only_in_case() -> None:
    con = xo.datafusion.connect()
    con.create_table("mytable", pa.table({"a": [1]}))
    con.create_table("MyTable", pa.table({"a": [2]}))

    con.drop_table("MyTable")

    assert sorted(con.list_tables()) == ["mytable"]


def test_read_record_batches_tee_provider_keeps_a_mixed_case_name() -> None:
    con = xo.datafusion.connect()

    t = con.read_record_batches(TeeTableProvider(make_reader(3)), "MyTee")

    assert t.count().execute() == 3
    con.drop_table("MyTee")
    assert con.list_tables() == []


def test_drop_table_refuses_a_view() -> None:
    con = xo.datafusion.connect()
    con.con.sql("CREATE VIEW vv AS SELECT 1 AS x").collect()

    with pytest.raises(Exception, match="vv"):
        con.drop_table("vv")

    assert con.list_tables() == ["vv"]


def test_read_record_batches_mixed_case_keeps_a_lowercase_table() -> None:
    con = xo.datafusion.connect()
    con.create_table("mytable", pa.table({"a": [1]}))

    t = con.read_record_batches(pa.table({"a": [2]}), "MyTable")

    assert t.a.sum().execute() == 2
    assert con.table("mytable").a.sum().execute() == 1


def test_drop_table_without_information_schema() -> None:
    ctx = datafusion.SessionContext(
        datafusion.SessionConfig().with_information_schema(False)
    )
    con = xo.datafusion.connect(ctx)
    con.read_record_batches(TeeTableProvider(make_reader(3)), "tee")
    ctx.from_arrow(pa.table({"a": [1]}), "base")

    con.drop_table("tee")
    con.drop_table("base")

    assert not ctx.table_exist("tee")
    assert not ctx.table_exist("base")


def test_read_record_batches_tee_provider_refuses_pandas_metadata() -> None:
    con = xo.datafusion.connect()
    reader = pa.Table.from_pandas(pd.DataFrame({"a": [1]})).to_reader()
    with pytest.raises(ValueError, match="pandas"):
        con.read_record_batches(TeeTableProvider(reader), "t")


def test_drop_table_with_a_custom_default_catalog() -> None:
    config = datafusion.SessionConfig().with_default_catalog_and_schema(
        "mycat", "mysch"
    )
    ctx = datafusion.SessionContext(config)
    con = xo.datafusion.connect(ctx)
    ctx.from_arrow(pa.table({"a": [1]}), "base")

    con.drop_table("base")

    assert not ctx.table_exist("base")


def test_read_record_batches_tee_provider_rolls_back_on_table_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    con = xo.datafusion.connect()

    def boom(self, *args, **kwargs):
        raise RuntimeError("table build failed")

    monkeypatch.setattr(type(con), "table", boom)
    with pytest.raises(RuntimeError, match="table build failed"):
        con.read_record_batches(TeeTableProvider(make_reader(3)), "t")

    assert not con.con.table_exist("t")
