from __future__ import annotations

import pyarrow as pa
import pytest
from adbc_driver_manager import AdbcStatusCode
from adbc_driver_manager import ProgrammingError as ADBCProgrammingError

import xorq.api as xo
from xorq.backends.postgres import Backend as PostgresBackend
from xorq.common.utils.postgres_utils import PgADBC


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


@pytest.mark.postgres
def test_to_pyarrow_batches_falls_back_when_adbc_unavailable(
    pg: PostgresBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    # When PgADBC.get_conn raises (no ADBC URI / config error), the swallowed
    # exception must route to the psycopg fallback and still return correct data.
    def boom(self, **kwargs):
        raise RuntimeError("no adbc uri")

    monkeypatch.setattr(PgADBC, "get_conn", boom)
    expr = pg.table("batting").limit(10)
    table = pg.to_pyarrow_batches(expr).read_all()
    assert table.num_rows == 10
    assert table.num_columns > 0


@pytest.mark.postgres
def test_to_pyarrow_batches_falls_through_on_adbc_programming_error(
    pg: PostgresBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ADBC connects but execute raises ADBCProgrammingError (the temp-table-
    # invisible case): fall_through routes to psycopg, and the finally block
    # must still close both the cursor and the ADBC connection.
    closed = {"cursor": False, "conn": False}

    class FakeCursor:
        def execute(self, query):
            raise ADBCProgrammingError(
                "relation does not exist", status_code=AdbcStatusCode.NOT_FOUND
            )

        def fetch_record_batch(self):  # pragma: no cover - never reached
            raise AssertionError("should not fetch after a programming error")

        def close(self):
            closed["cursor"] = True

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def close(self):
            closed["conn"] = True

    monkeypatch.setattr(PgADBC, "get_conn", lambda self, **kwargs: FakeConn())
    expr = pg.table("batting").limit(10)
    table = pg.to_pyarrow_batches(expr).read_all()
    assert table.num_rows == 10
    assert closed == {"cursor": True, "conn": True}
