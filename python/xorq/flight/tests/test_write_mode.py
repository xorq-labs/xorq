import pyarrow as pa
import pytest

import xorq.api as xo
from xorq.flight import FlightServer
from xorq.flight.enums import UploadMode
from xorq.flight.server import parse_put_command, put_table


CONNECTS = {
    "duckdb": lambda: xo.duckdb.connect(),
    "xorq": lambda: xo.connect(),
}
FIRST = pa.table({"a": [1, 2]})
SECOND = pa.table({"a": [3]})


@pytest.fixture(params=tuple(CONNECTS))
def con(request):
    return CONNECTS[request.param]()


def count(con, name="t"):
    return con.table(name).count().execute()


# do_put runs on a native gRPC thread, so drive its body on this one too
@pytest.mark.parametrize(
    ("mode", "expected"),
    ((None, 3), (UploadMode.APPEND, 3), (UploadMode.REPLACE, 1)),
)
def test_write_mode_on_existing_table(con, mode, expected):
    put_table(con, "t", FIRST, None)
    put_table(con, "t", SECOND, mode)
    assert count(con) == expected


@pytest.mark.parametrize("mode", (None, UploadMode.CREATE, UploadMode.REPLACE))
def test_write_mode_on_new_table(con, mode):
    put_table(con, "t", FIRST, mode)
    assert count(con) == 2


def test_write_mode_create_refuses_existing_table(con):
    put_table(con, "t", FIRST, None)
    with pytest.raises(Exception, match="already exists"):
        put_table(con, "t", SECOND, UploadMode.CREATE)
    assert count(con) == 2


def test_write_mode_append_refuses_absent_table(con):
    with pytest.raises(Exception, match="not found"):
        put_table(con, "t", FIRST, UploadMode.APPEND)


@pytest.mark.parametrize("mode", tuple(UploadMode))
def test_write_mode_refuses_overwrite_kwarg(con, mode):
    put_table(con, "t", FIRST, None)
    with pytest.raises(ValueError, match="instead of overwrite"):
        put_table(con, "t", SECOND, mode, overwrite=True)
    assert count(con) == 2


def test_write_mode_unknown_raises_before_writing(con):
    with pytest.raises(ValueError, match="'merge' is not a valid UploadMode"):
        put_table(con, "t", FIRST, "merge")
    assert "t" not in con.tables


@pytest.mark.parametrize(
    ("mode", "expected"),
    ((None, 3), ("append", 3), ("replace", 1)),
)
def test_write_mode_over_flight(con, mode, expected):
    with FlightServer(make_connection=lambda: con) as server:
        server.client.upload_table("t", FIRST)
        server.client.upload_batches("t", SECOND.to_reader(), mode=mode)
        actual = server.client.execute(server.con.table("t"))
    assert actual.num_rows == expected


def test_write_mode_create_over_flight_refuses_existing_table(con):
    with FlightServer(make_connection=lambda: con) as server:
        server.client.upload_table("t", FIRST, mode="create")
        with pytest.raises(pa.ArrowException, match="already exists"):
            server.client.upload_table("t", SECOND, mode="create")


def test_write_mode_unknown_fails_before_upload():
    sent = []

    def batches():
        for batch in FIRST.to_batches():
            sent.append(batch)
            yield batch

    reader = pa.RecordBatchReader.from_batches(FIRST.schema, batches())
    with FlightServer() as server:
        with pytest.raises(ValueError, match="'merge' is not a valid UploadMode"):
            server.client.upload_batches("t", reader, mode="merge")
        assert "t" not in server.con.list_tables()
    assert not sent


def test_flight_backend_create_table_honours_overwrite(con):
    with FlightServer(make_connection=lambda: con) as server:
        backend = server.con
        backend.create_table("t", FIRST)
        with pytest.raises(pa.ArrowException, match="already exists"):
            backend.create_table("t", SECOND)
        replaced = backend.create_table("t", SECOND.to_reader(), overwrite=True)
        assert server.client.execute(replaced).num_rows == 1


@pytest.mark.parametrize(
    ("command", "expected"),
    (
        (b"t", ("t", {}, None)),
        ({"table_name": "t"}, ("t", {}, None)),
        (
            {"table_name": "t", "kwargs": {"temp": True}, "mode": "replace"},
            ("t", {"temp": True}, "replace"),
        ),
    ),
)
def test_write_mode_parse_put_command(command, expected):
    assert parse_put_command(command) == expected
