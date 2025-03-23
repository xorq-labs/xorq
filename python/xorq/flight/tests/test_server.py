import datetime

import pandas as pd
import pyarrow as pa
import pytest

import xorq as xo
from xorq.common.utils.rbr_utils import instrument_reader
from xorq.flight import FlightServer, FlightUrl
from xorq.flight.action import AddExchangeAction
from xorq.flight.exchanger import EchoExchanger, PandasUDFExchanger


def make_flight_url(port):
    if port is not None:
        assert not FlightUrl.port_in_use(port), f"Port {port} already in use"
    flight_url = FlightUrl(port=port)
    assert FlightUrl.port_in_use(flight_url.port), (
        f"Port {flight_url.port} should be in use"
    )
    return flight_url


@pytest.mark.parametrize(
    "connection,port",
    [
        pytest.param(xo.duckdb.connect, 5005, id="duckdb"),
        pytest.param(xo.datafusion.connect, 5005, id="datafusion"),
        pytest.param(xo.connect, 5005, id="xorq"),
    ],
)
def test_port_in_use(connection, port):
    assert port is not None
    flight_url = make_flight_url(port)
    with pytest.raises(OSError, match="Address already in use"):
        with FlightServer(
            flight_url=flight_url,
            connection=connection,
        ) as _:
            # entering the above context releases the port
            # so we won't raise until we enter the second context and try to use it
            flight_url2 = FlightUrl(port=port)  # noqa: F841


@pytest.mark.parametrize(
    "connection,port",
    [
        pytest.param(xo.duckdb.connect, None, id="duckdb"),
        pytest.param(xo.datafusion.connect, None, id="datafusion"),
        pytest.param(xo.connect, None, id="xorq"),
    ],
)
def test_register_and_list_tables(connection, port):
    flight_url = make_flight_url(port)

    with FlightServer(
        flight_url=flight_url,
        verify_client=False,
        connection=connection,
    ) as main:
        con = main.con
        assert con.version is not None

        data = pa.table(
            {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}
        ).to_pandas()

        con.register(data, table_name="users")
        t = con.table("users")
        actual = xo.execute(t)

        assert t.schema() is not None
        assert "users" in con.list_tables()
        assert isinstance(actual, pd.DataFrame)


@pytest.mark.parametrize(
    "connection,port",
    [
        pytest.param(xo.duckdb.connect, None, id="duckdb"),
        pytest.param(xo.datafusion.connect, None, id="datafusion"),
        pytest.param(xo.connect, None, id="xorq"),
    ],
)
def test_into_backend_flight_server(connection, port, parquet_dir):
    batting = xo.read_parquet(parquet_dir / "batting.parquet")
    flight_url = make_flight_url(port)

    with FlightServer(
        flight_url=flight_url,
        verify_client=False,
        connection=connection,
    ) as main:
        con = main.con
        t = batting.filter(batting.yearID == 2015).into_backend(con, "xo_batting")
        expr = (
            t.join(t, "playerID")
            .limit(15)
            .select(player_id="playerID", year_id="yearID_right")
        )

        assert not expr.execute().empty


@pytest.mark.parametrize(
    "connection,port",
    [
        pytest.param(xo.duckdb.connect, None, id="duckdb"),
        pytest.param(xo.datafusion.connect, None, id="datafusion"),
        pytest.param(xo.connect, None, id="xorq"),
    ],
)
def test_read_parquet(connection, port, parquet_dir):
    flight_url = make_flight_url(port)
    with FlightServer(
        flight_url=flight_url,
        verify_client=False,
        connection=connection,
    ) as main:
        con = main.con
        batting = con.read_parquet(parquet_dir / "batting.parquet")
        assert xo.execute(batting) is not None


@pytest.mark.parametrize(
    "connection,port",
    [
        pytest.param(xo.duckdb.connect, None, id="duckdb"),
        pytest.param(xo.datafusion.connect, None, id="datafusion"),
        pytest.param(xo.connect, None, id="xorq"),
    ],
)
def test_exchange(connection, port):
    flight_url = make_flight_url(port)

    def my_f(df):
        return df[["a", "b"]].sum(axis=1)

    with FlightServer(
        flight_url=flight_url,
        verify_client=False,
        connection=connection,
    ) as main:
        client = main.client
        udf_exchanger = PandasUDFExchanger(
            my_f,
            schema_in=pa.schema(
                (
                    pa.field("a", pa.int64()),
                    pa.field("b", pa.int64()),
                )
            ),
            name="x",
            typ=pa.int64(),
            append=True,
        )
        client.do_action(AddExchangeAction.name, udf_exchanger, options=client._options)

        # a small example
        df_in = pd.DataFrame({"a": [1], "b": [2], "c": [100]})
        fut, rbr = client.do_exchange(
            udf_exchanger.command,
            pa.RecordBatchReader.from_stream(df_in),
        )
        df_out = rbr.read_pandas()
        writes_reads = fut.result()

        assert writes_reads["n_writes"] == writes_reads["n_reads"]
        assert df_out is not None

        # demonstrate streaming
        df_in = pd.DataFrame(
            {
                "a": range(100_000),
                "b": range(100_000, 200_000),
                "c": range(200_000, 300_000),
            }
        )
        fut, rbr = client.do_exchange_batches(
            udf_exchanger.command,
            instrument_reader(pa.Table.from_pandas(df_in).to_reader(max_chunksize=100)),
        )
        first_batch = next(rbr)
        first_batch_time = datetime.datetime.now()
        assert first_batch is not None, f"must get first batch by {first_batch_time}"

        rest = rbr.read_pandas()
        rest_time = datetime.datetime.now()
        assert rest is not None, f"must get first batch by {rest_time}"
        assert first_batch_time < rest_time

        writes_reads = fut.result()
        assert writes_reads["n_writes"] == 1000  # because
        assert writes_reads["n_reads"] == 1000


@pytest.mark.parametrize(
    "connection",
    (
        xo.duckdb.connect,
        xo.datafusion.connect,
        xo.connect,
    ),
)
def test_reentry(connection):
    df_in = pd.DataFrame({"a": [1], "b": [2], "c": [100]})
    with FlightServer(
        verify_client=False,
        connection=connection,
    ) as server:
        fut, rbr = server.client.do_exchange(
            EchoExchanger.command,
            pa.RecordBatchReader.from_stream(df_in),
        )
        df_out = rbr.read_pandas()
        assert df_in.equals(df_out)
    with server:
        fut, rbr = server.client.do_exchange(
            EchoExchanger.command,
            pa.RecordBatchReader.from_stream(df_in),
        )
        df_out = rbr.read_pandas()
        assert df_in.equals(df_out)


@pytest.mark.parametrize(
    "connection",
    (
        xo.duckdb.connect,
        xo.datafusion.connect,
        xo.connect,
    ),
)
def test_serve_close(connection):
    df_in = pd.DataFrame({"a": [1], "b": [2], "c": [100]})
    server = FlightServer(
        verify_client=False,
        connection=connection,
    )

    server.serve()
    fut, rbr = server.client.do_exchange(
        EchoExchanger.command,
        pa.RecordBatchReader.from_stream(df_in),
    )
    df_out = rbr.read_pandas()
    assert df_in.equals(df_out)
    server.close()

    server.serve()
    fut, rbr = server.client.do_exchange(
        EchoExchanger.command,
        pa.RecordBatchReader.from_stream(df_in),
    )
    df_out = rbr.read_pandas()
    assert df_in.equals(df_out)
    server.close()


def test_ctor_exchanger_registration():
    from xorq.flight.action import ListExchangesAction
    from xorq.flight.exchanger import make_udxf

    def dummy(df: pd.DataFrame):
        return pd.DataFrame({"row_count": [42]})

    schema_in = xo.schema({"dummy": "int64"})
    dummy_udxf = make_udxf(
        dummy,
        schema_in.to_pyarrow(),
        xo.schema({"row_count": "int64"}).to_pyarrow(),
    )
    flight_server = FlightServer(exchangers=[dummy_udxf])
    with flight_server:
        client = flight_server.client
        available = client.do_action_one(ListExchangesAction.name)
        assert dummy_udxf.command in available
        client.do_exchange(
            dummy_udxf.command,
            xo.memtable({"dummy": [0]}, schema=schema_in).to_pyarrow_batches(),
        )


@pytest.mark.parametrize(
    "expr",
    [
        pytest.param(xo.table({"id": int}, name="users").count(), id="scalar"),
        pytest.param(xo.literal(1), id="literal"),
        pytest.param(xo.table({"id": int}, name="users").id, id="column"),
    ],
)
def test_execute_query_non_relation_expr(expr):
    flight_url = make_flight_url(None)
    data = pa.table({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}).to_pandas()

    with FlightServer(
        flight_url=flight_url,
        verify_client=False,
        connection=xo.duckdb.connect,
    ) as main:
        main.con.register(data, table_name="users")
        actual = main.client.execute_query(expr)
        assert isinstance(actual, pa.Table)
        assert len(actual) > 0
