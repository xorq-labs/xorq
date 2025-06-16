import pyarrow as pa
import pytest

import xorq as xo
from xorq.flight import FlightServer
from xorq.flight.tests.test_server import make_flight_url


@pytest.fixture(scope="module")
def flight_server():
    flight_url = make_flight_url(None)
    with FlightServer(
        flight_url=flight_url,
        verify_client=False,
        connection=xo.duckdb.connect,
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


def test_create_table_from_iterable_of_batches(flight_server):
    # Prepare a pyarrow Table and get its batches as a list
    pa_table = pa.table({"m": [0, 1, 2], "n": [3.0, 4.0, 5.0]})
    batches = pa_table.to_batches()
    backend = flight_server.con
    # Create table from Iterable[RecordBatch]
    tbl = backend.create_table("iter_tbl", batches)
    result = flight_server.client.execute(tbl)
    assert isinstance(result, pa.Table)
    assert result.schema.equals(pa_table.schema)
    assert result.to_pydict() == pa_table.to_pydict()


@pytest.mark.parametrize("obj", [[], 42])
def test_create_table_invalid_inputs(flight_server, obj):
    backend = flight_server.con
    # Creating a table from an empty iterable should raise ValueError
    # Creating from unsupported type should raise TypeError
    if isinstance(obj, list):
        with pytest.raises(
            ValueError, match="Cannot create table from empty batch list"
        ):
            backend.create_table("empty_tbl", obj)
    else:
        with pytest.raises(TypeError, match="Unsupported type for create_table"):
            backend.create_table("bad_tbl", obj)
