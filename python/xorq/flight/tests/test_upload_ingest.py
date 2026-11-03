import time

import pyarrow as pa
import pytest

import xorq.api as xo
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
