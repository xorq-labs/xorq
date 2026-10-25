from __future__ import annotations

from collections.abc import Callable

import pyarrow as pa
import pytest
from batchcorder import StreamCache

import xorq.api as xo
from xorq.backends.duckdb import Backend
from xorq.vendor.ibis.expr import types as ir


FRAME = pa.table({"k": [0, 1], "v": [1.0, 2.0]})


def single_scan_relation(con: Backend) -> ir.Table:
    # a relation over a stream that allows one scan; referenced twice, the
    # compiler hoists it into a CTE, which must then be scanned once
    reader = pa.RecordBatchReader.from_batches(FRAME.schema, FRAME.to_batches())
    t = con.read_record_batches(StreamCache(reader, max_readers=1))
    return t.filter(t.v > 0)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(lambda m: m.union(m).limit(1), id="self_union_limit"),
        pytest.param(lambda m: m.join(m.view(), "k").limit(1), id="self_join_limit"),
    ],
)
def test_cte_over_registered_stream_scans_it_once(
    shape: Callable[[ir.Table], ir.Table],
) -> None:
    con = xo.duckdb.connect()
    expr = shape(single_scan_relation(con))
    assert len(expr.execute()) == 1


def test_create_table_over_cte_reading_a_stream_scans_it_once() -> None:
    con = xo.duckdb.connect()
    m = single_scan_relation(con)
    assert con.create_table("x", m.union(m).limit(1)).count().execute() == 1


@pytest.mark.parametrize(
    "make_table",
    [
        pytest.param(lambda con: con.create_table("t", FRAME), id="table"),
        pytest.param(lambda con: con.read_record_batches(FRAME), id="arrow_table"),
    ],
)
def test_cte_over_rescannable_table_compiles_as_before(
    make_table: Callable[[Backend], ir.Table],
) -> None:
    con = xo.duckdb.connect()
    t = make_table(con)
    m = t.filter(t.v > 0)
    expr = m.union(m).limit(1)
    assert con.compile(expr) == con.compiler.to_sqlglot(expr).sql(dialect=con.dialect)


def test_compile_leaves_an_open_result_streaming() -> None:
    con = xo.duckdb.connect()
    t = con.create_table("t", pa.table({"v": range(200_000)}))
    m = t.filter(t.v >= 0)
    reader = con.to_pyarrow_batches(t, chunk_size=1_000)
    first = reader.read_next_batch()
    con.compile(m.union(m).limit(1))
    assert first.num_rows + reader.read_all().num_rows == 200_000
