import datetime
import sqlite3
import threading
import time

import pyarrow as pa
import pytest

import xorq.api as xo
from xorq.backends.sqlite import Backend as SQLiteBackend
from xorq.flight import FlightServer
from xorq.flight.client import make_put_descriptor
from xorq.vendor.ibis.util import gen_name


FIRST = pa.table({"a": [1, 2]})
SECOND = pa.table({"a": [3]})


def sqlite_file(tmp_path):
    # served from Flight's threads, so not bound to the creating thread
    return xo.sqlite.connect(str(tmp_path / "x.db"), check_same_thread=False)


INGESTING = (
    pytest.param(sqlite_file, id="sqlite"),
    pytest.param(
        lambda _: xo.postgres.connect_env(), id="postgres", marks=pytest.mark.postgres
    ),
)
BUFFERED = (
    pytest.param(lambda _: xo.duckdb.connect(), id="duckdb"),
    pytest.param(
        lambda _: xo.sqlite.connect(check_same_thread=False), id="sqlite-memory"
    ),
)


@pytest.fixture
def con(request, tmp_path):
    return request.param(tmp_path)


@pytest.fixture
def table(con):
    # postgres is shared across test workers
    name = gen_name("upload_ingest")
    yield name
    con.drop_table(name, force=True)


def count(con, table):
    return con.table(table).count().execute()


@pytest.mark.parametrize("con", INGESTING, indirect=True)
@pytest.mark.parametrize(
    ("mode", "expected"),
    ((None, 3), ("append", 3), ("replace", 1)),
)
def test_streamed_upload_on_existing_table(con, table, mode, expected):
    with FlightServer(make_connection=lambda: con) as server:
        server.client.upload_table(table, FIRST)
        server.client.upload_batches(table, SECOND.to_reader(), mode=mode)
    assert count(con, table) == expected


@pytest.mark.parametrize("con", INGESTING, indirect=True)
@pytest.mark.parametrize("mode", (None, "create", "replace"))
def test_streamed_upload_on_new_table(con, table, mode):
    with FlightServer(make_connection=lambda: con) as server:
        server.client.upload_batches(table, FIRST.to_reader(), mode=mode)
    assert count(con, table) == 2


@pytest.mark.parametrize("con", INGESTING, indirect=True)
def test_streamed_upload_create_refuses_existing_table(con, table):
    with FlightServer(make_connection=lambda: con) as server:
        server.client.upload_table(table, FIRST)
        with pytest.raises(pa.ArrowException, match="already exists"):
            server.client.upload_table(table, SECOND, mode="create")
    assert count(con, table) == 2


@pytest.mark.parametrize("con", INGESTING, indirect=True)
def test_streamed_upload_append_refuses_absent_table(con, table):
    with FlightServer(make_connection=lambda: con) as server:
        with pytest.raises(pa.ArrowException, match="(no such table|does not exist)"):
            server.client.upload_table(table, FIRST, mode="append")
    assert table not in con.list_tables()


@pytest.mark.parametrize("con", INGESTING, indirect=True)
def test_upload_with_kwargs_keeps_create_table_keywords(con, table):
    # an ADBC ingest has no `overwrite`; the buffered write honours it
    with FlightServer(make_connection=lambda: con) as server:
        server.client.upload_table(table, FIRST)
        server.client.upload_table(table, SECOND, overwrite=True)
    assert count(con, table) == 1


@pytest.mark.parametrize("con", (*INGESTING, *BUFFERED), indirect=True)
def test_cancelled_upload_writes_nothing(con, table):
    def stalling():
        yield from FIRST.to_batches()
        time.sleep(2)
        yield from SECOND.to_batches()

    reader = pa.RecordBatchReader.from_batches(FIRST.schema, stalling())
    # leaving the block shuts the server down, which waits for do_put
    with FlightServer(make_connection=lambda: con) as server:
        url = server.flight_url
        client = pa.flight.connect(f"{url.scheme}://{url.host}:{url.port}")
        # the client's deadline passes while the upload stalls
        options = pa.flight.FlightCallOptions(timeout=0.5)
        with pytest.raises(pa.ArrowException):
            writer, _ = client.do_put(
                make_put_descriptor(table, None, {}), reader.schema, options=options
            )
            for batch in reader:
                writer.write_batch(batch)
            writer.done_writing()
            writer.close()
    assert table not in con.list_tables()


@pytest.mark.parametrize("con", INGESTING, indirect=True)
def test_streamed_upload_declares_buffered_types(con, table):
    data = pa.table(
        {
            "flag": [True],
            "ts": [datetime.datetime(2020, 1, 1)],
            "n": pa.array([1], pa.int32()),
        }
    )
    expected = gen_name("buffered")
    con.create_table(expected, data)
    try:
        with FlightServer(make_connection=lambda: con) as server:
            server.client.upload_table(table, data)
        assert con.table(table).schema() == con.table(expected).schema()
        assert con.table(table).to_pyarrow().equals(con.table(expected).to_pyarrow())
    finally:
        con.drop_table(expected, force=True)


@pytest.mark.parametrize("con", INGESTING, indirect=True)
@pytest.mark.parametrize("mode", (None, "append"))
def test_streamed_append_matches_columns_by_name_and_casts(con, table, mode):
    con.create_table(table, schema=xo.schema({"a": "int32", "b": "string"}))
    upload = pa.table({"b": ["x"], "a": pa.array([1], pa.int64())})
    with FlightServer(make_connection=lambda: con) as server:
        server.client.upload_table(table, upload, mode=mode)
    assert con.table(table).to_pyarrow().to_pylist() == [{"a": 1, "b": "x"}]


@pytest.mark.parametrize("con", INGESTING, indirect=True)
def test_streamed_upload_leaves_target_readable_while_it_arrives(con, table):
    stall = threading.Event()

    def stalling():
        yield from FIRST.to_batches()
        stall.wait(timeout=10)
        yield from SECOND.to_batches()

    with FlightServer(make_connection=lambda: con) as server:
        server.client.upload_table(table, FIRST)
        uploading = threading.Thread(
            target=server.client.upload_batches,
            args=(table, pa.RecordBatchReader.from_batches(FIRST.schema, stalling())),
            kwargs={"mode": "replace"},
        )
        uploading.start()
        try:
            # the first batch has reached the server's ingest
            time.sleep(0.5)
            started = time.monotonic()
            # served under the server lock, as do_get would be
            result = server.client.execute(con.table(table).count())
            assert result.column(0).to_pylist() == [2]
            assert time.monotonic() - started < 2
        finally:
            stall.set()
            uploading.join()
    assert count(con, table) == 3


@pytest.mark.postgres
@pytest.mark.parametrize(
    "con",
    (pytest.param(lambda _: xo.postgres.connect_env(), id="postgres"),),
    indirect=True,
)
def test_postgres_without_adbc_connection_keeps_buffered_upload(con, table):
    # authenticated another way (.pgpass, PGPASSWORD): no URI to build
    con._con_kwargs.pop("password")
    assert con.upload_ingest(table) is None
    with FlightServer(make_connection=lambda: con) as server:
        server.client.upload_table(table, FIRST)
    assert count(con, table) == 2


@pytest.mark.parametrize("con", INGESTING, indirect=True)
def test_upload_into_temporary_table_appends_to_it(con, table):
    # only the server's own session sees the temporary table
    with FlightServer(make_connection=lambda: con) as server:
        server.client.upload_table(table, FIRST, temp=True)
        server.client.upload_table(table, SECOND)
    assert count(con, table) == 3
    con.drop_table(table)
    assert table not in con.list_tables()


def test_sqlite_from_connection_keeps_buffered_upload(tmp_path):
    # no path to reopen: the backend was handed a connection
    con = SQLiteBackend.from_connection(
        sqlite3.connect(tmp_path / "x.db", check_same_thread=False)
    )
    table = gen_name("upload_ingest")
    assert con.upload_ingest(table) is None
    with FlightServer(make_connection=lambda: con) as server:
        server.client.upload_table(table, FIRST)
    assert count(con, table) == 2
