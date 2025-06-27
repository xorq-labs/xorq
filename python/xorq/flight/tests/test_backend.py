import functools
import threading

import pandas as pd
import pyarrow as pa
import pytest

import xorq as xo
from xorq import _
from xorq.common.utils.tls_utils import TLSKwargs
from xorq.flight import FlightServer
from xorq.flight.exchanger import make_udxf
from xorq.flight.tests.test_server import make_flight_url


@pytest.fixture(scope="module")
def flight_server():
    flight_url = make_flight_url(None)
    with FlightServer(
        flight_url=flight_url,
        verify_client=False,
        make_connection=xo.duckdb.connect,
    ) as server:
        yield server


def test_create_table_from_pa_table(flight_server):
    # Prepare a simple pyarrow Table
    pa_table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    # Get the flight backend (Ibis) client
    backend = flight_server.con
    # Create table from a pa.Table
    tbl = backend.create_table("pa_tbl", pa_table)
    # Read back via the FlightClient, should equal the original
    result = flight_server.client.execute(tbl)
    assert isinstance(result, pa.Table)
    # Compare data and schema
    assert result.schema.equals(pa_table.schema)
    assert result.to_pydict() == pa_table.to_pydict()


def test_create_table_from_recordbatchreader(flight_server):
    # Prepare a pyarrow Table and make a RecordBatchReader
    pa_table = pa.table({"x": [10, 20], "y": [True, False]})
    reader = pa.RecordBatchReader.from_batches(pa_table.schema, pa_table.to_batches())
    backend = flight_server.con
    # Create table from a RecordBatchReader
    tbl = backend.create_table("rbr_tbl", reader)
    result = flight_server.client.execute(tbl)
    assert isinstance(result, pa.Table)
    assert result.schema.equals(pa_table.schema)
    assert result.to_pydict() == pa_table.to_pydict()


@pytest.mark.parametrize("obj", [[], 42])
def test_create_table_invalid_inputs(flight_server, obj):
    backend = flight_server.con
    # Creating from unsupported type should raise TypeError
    with pytest.raises(TypeError, match="Unsupported type for create_table"):
        backend.create_table("bad_tbl", obj)


def test_backend_get_flight_udxf():
    flight_url = make_flight_url(None, scheme="grpc+tls")
    tls_kwargs = TLSKwargs.from_common_name(verify_client=False)

    def dummy(df: pd.DataFrame):
        return pd.DataFrame({"row_count": [42]})

    dummy_udxf = make_udxf(
        dummy,
        xo.schema({"dummy": "int64"}),
        xo.schema({"row_count": "int64"}),
    )

    server = FlightServer(
        flight_url=flight_url,
        verify_client=False,
        exchangers=[dummy_udxf],
        **tls_kwargs.server_kwargs,
    )

    server_thread = threading.Thread(
        target=functools.partial(server.serve, block=True), daemon=True
    )
    server_thread.start()
    server.client

    con = xo.connect()
    backend = xo.flight.connect(flight_url, tls_kwargs)

    f = backend.get_flight_udxf(dummy_udxf.command)
    dummy_table = con.register(pd.DataFrame({"dummy": [21, 0, 21]}), table_name="dummy")
    expr = dummy_table.filter(dummy_table.dummy >= 0).pipe(f).filter(_.row_count == 42)

    assert not expr.execute().empty
    with pytest.raises(ValueError):
        foo_table = con.register(pd.DataFrame({"foo": ["value"]}), table_name="foo")
        foo_table.pipe(f).execute()

    server.server.shutdown()
    server_thread.join(timeout=2.0)

    assert not server_thread.is_alive()
