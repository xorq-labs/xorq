from __future__ import annotations

import pyarrow as pa
import pytest

import xorq.api as xo
from xorq.backends.postgres import Backend as PostgresBackend


@pytest.mark.postgres
def test_to_pyarrow_batches_simple(pg: PostgresBackend) -> None:
    expr = pg.table("batting").limit(100)
    reader = pg.to_pyarrow_batches(expr)
    assert isinstance(reader, pa.RecordBatchReader)
    table = reader.read_all()
    assert table.num_rows == 100
    assert table.num_columns > 0


@pytest.mark.postgres
def test_to_pyarrow_batches_with_filter(pg: PostgresBackend) -> None:
    expr = pg.table("batting").filter(pg.table("batting").yearID == 2015)
    reader = pg.to_pyarrow_batches(expr)
    table = reader.read_all()
    assert table.num_rows > 0
    assert all(v == 2015 for v in table.column("yearID").to_pylist())


@pytest.mark.postgres
def test_to_pyarrow_batches_into_backend_roundtrip(pg: PostgresBackend) -> None:
    con = xo.connect()
    expr = pg.table("batting").limit(50).into_backend(con, "pg_bat")
    result = expr.execute()
    assert len(result) == 50
