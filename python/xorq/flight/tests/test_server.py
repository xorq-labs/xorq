import datetime
import functools
import operator
import threading
import time

import pandas as pd
import pyarrow as pa
import pytest

import xorq as xo
import xorq.expr.datatypes as dt
import xorq.flight.action as A
import xorq.flight.exchanger as E
from xorq.common.utils import classproperty
from xorq.common.utils.rbr_utils import instrument_reader
from xorq.common.utils.tls_utils import TLSKwargs
from xorq.flight import (
    Backend,
    BasicAuth,
    FlightServer,
    FlightUrl,
)
from xorq.flight.action import AddExchangeAction
from xorq.flight.exchanger import EchoExchanger, PandasUDFExchanger
from xorq.flight.tests.conftest import do_agg, field_name, my_udf, return_type
from xorq.tests.util import assert_frame_equal


def make_flight_url(port, scheme="grpc", auth=None):
    if port is not None:
        assert not FlightUrl.port_in_use(port), f"Port {port} already in use"
    flight_url = (
        FlightUrl(port=port, scheme=scheme)
        if auth is None
        else FlightUrl(
            port=port, scheme=scheme, username=auth.username, password=auth.password
        )
    )
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
            make_connection=connection,
        ) as _:
            # entering the above context releases the port
            # so we won't raise until we enter the second context and try to use it
            flight_url2 = FlightUrl(port=port)  # noqa: F841


class Answer42Action(A.AbstractAction):
    @classproperty
    def name(cls):
        return "answer-42"

    @classmethod
    def description(cls):
        return (
            "the answer to the ultimate question of life, the universe, and everything"
        )

    @classmethod
    def do_action(cls, server, context, action):
        yield A.make_flight_result(42)


def test_list_actions():
    with FlightServer() as flight_server:
        actions = flight_server.client.do_action_one(A.ListActionsAction.name)
        assert actions == tuple(A.actions)


def test_add_action():
    with FlightServer() as flight_server:
        actions = flight_server.client.do_action_one(A.ListActionsAction.name)
        assert Answer42Action.name not in actions
        flight_server.client.do_action(A.AddActionAction.name, Answer42Action)
        actions = flight_server.client.do_action_one(A.ListActionsAction.name)
        assert Answer42Action.name in actions


def test_list_exchanges():
    with FlightServer() as flight_server:
        exchanges = flight_server.client.do_action_one(A.ListExchangesAction.name)
        assert exchanges == tuple(E.exchangers)


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
        make_connection=connection,
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


@pytest.mark.parametrize("auth", [None, BasicAuth("username", "password")])
@pytest.mark.parametrize("verify_client", [False, True])
def test_tls_encryption(auth, verify_client, tls_kwargs, mtls_kwargs):
    flight_url = make_flight_url(None, scheme="grpc+tls")

    tls_kwargs = TLSKwargs.from_common_name(verify_client=verify_client)

    with FlightServer(
        flight_url=flight_url,
        verify_client=verify_client,
        auth=auth,
        **tls_kwargs.server_kwargs,
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


def test_failed_auth(tls_kwargs):
    flight_url = make_flight_url(None, scheme="grpc+tls")

    kwargs = TLSKwargs.from_common_name().server_kwargs

    with FlightServer(
        flight_url=flight_url,
        auth=BasicAuth("username", "password"),
        **kwargs,
    ) as server:
        kwargs = {
            "host": server.flight_url.host,
            "port": server.flight_url.port,
            "username": server.auth.username,
            "password": "not_the_password",
            "tls_root_certs": kwargs["root_certificates"],
        }

        from pyarrow._flight import FlightUnauthenticatedError

        with pytest.raises(FlightUnauthenticatedError):
            instance = Backend()
            instance.do_connect(**kwargs)


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
        make_connection=connection,
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
        make_connection=connection,
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
        make_connection=connection,
    ) as main:
        client = main.client
        udf_exchanger = PandasUDFExchanger(
            my_f,
            schema_in=xo.schema({"a": int, "b": int}),
            name="x",
            typ=dt.int64,
            append=True,
        )
        client.do_action(AddExchangeAction.name, udf_exchanger, options=client._options)

        # a small example
        df_in = pd.DataFrame({"a": [1], "b": [2], "c": [100]})
        fut, rbr = client.do_exchange(
            udf_exchanger.command,
            xo.memtable(df_in),
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
        make_connection=connection,
    ) as server:
        fut, rbr = server.client.do_exchange_batches(
            EchoExchanger.command,
            pa.RecordBatchReader.from_stream(df_in),
        )
        df_out = rbr.read_pandas()
        assert df_in.equals(df_out)
    with server:
        fut, rbr = server.client.do_exchange_batches(
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
        make_connection=connection,
    )

    server.serve()
    fut, rbr = server.client.do_exchange_batches(
        EchoExchanger.command,
        pa.RecordBatchReader.from_stream(df_in),
    )
    df_out = rbr.read_pandas()
    assert df_in.equals(df_out)
    server.close()

    server.serve()
    fut, rbr = server.client.do_exchange_batches(
        EchoExchanger.command,
        pa.RecordBatchReader.from_stream(df_in),
    )
    df_out = rbr.read_pandas()
    assert df_in.equals(df_out)
    server.close()


def test_ctor_exchanger_registration():
    def dummy(df: pd.DataFrame):
        return pd.DataFrame({"row_count": [42]})

    schema_in = xo.schema({"dummy": "int64"})
    dummy_udxf = E.make_udxf(
        dummy,
        schema_in,
        xo.schema({"row_count": "int64"}),
    )
    flight_server = FlightServer(exchangers=[dummy_udxf])
    with flight_server:
        client = flight_server.client
        available = client.do_action_one(A.ListExchangesAction.name)
        assert dummy_udxf.command in available
        client.do_exchange(
            dummy_udxf.command,
            xo.memtable({"dummy": [0]}, schema=schema_in),
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
        make_connection=xo.duckdb.connect,
    ) as main:
        main.con.register(data, table_name="users")
        actual = main.client.execute(expr)
        assert isinstance(actual, pa.Table)
        assert len(actual) > 0


@pytest.mark.parametrize("block", [True, False])
def test_server_blocks(block):
    flight_url = make_flight_url(None)
    server = FlightServer(
        flight_url=flight_url,
        verify_client=False,
        make_connection=xo.duckdb.connect,
    )

    server_thread = threading.Thread(
        target=functools.partial(server.serve, block=block), daemon=True
    )

    is_blocking = True

    def check_if_still_running():
        nonlocal is_blocking
        time.sleep(1)
        is_blocking = server_thread.is_alive()

    server_thread.start()

    checker_thread = threading.Thread(target=check_if_still_running)
    checker_thread.start()
    checker_thread.join()

    # Try to stop the inner server
    server.server.shutdown()
    server_thread.join(timeout=2.0)

    assert is_blocking == block
    assert not server_thread.is_alive()


def test_exchange_server_from_udxf(con, diamonds, baseline):
    input_expr = diamonds.pipe(do_agg)
    process_df = operator.methodcaller("assign", **{field_name: my_udf.fn})
    maybe_schema_in = input_expr.schema()
    maybe_schema_out = xo.schema(input_expr.schema() | {field_name: return_type})
    command = "diamonds_exchange_command"
    expr = xo.expr.relations.flight_udxf(
        input_expr,
        process_df=process_df,
        maybe_schema_in=maybe_schema_in,
        maybe_schema_out=maybe_schema_out,
        con=con,
        make_udxf_kwargs={"name": my_udf.__name__, "command": command},
    ).order_by("cut")

    with FlightServer.from_udxf(expr) as server:
        client = server.client
        assert client is not None

        _, rbr = client.do_exchange(
            command,
            input_expr,
        )

        actual = rbr.read_pandas().sort_values("cut", ignore_index=True)
        expected = baseline.sort_values("cut", ignore_index=True)
        assert_frame_equal(
            actual,
            expected,
            check_exact=False,
        )
