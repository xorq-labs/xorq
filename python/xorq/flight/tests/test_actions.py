import os
import signal
import threading

import pandas as pd
import pyarrow as pa
import pytest

import xorq.api as xo
import xorq.flight.exchanger as E
from xorq.flight import FlightServer
from xorq.flight.action import (
    DropTableAction,
    GetExchangeAction,
    GetSchemaQueryAction,
    ListTablesAction,
    ShutdownAction,
)


def test_list_tables_kwargs():
    with FlightServer(
        verify_client=False,
        make_connection=xo.duckdb.connect,
    ) as main:
        # GIVEN
        data = pa.table(
            {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}
        ).to_pandas()
        main.con.create_table("users", data)
        expected = ["users"]

        # WHEN
        actual = main.client.do_action_one(ListTablesAction.name)

        # THEN
        assert actual == expected


def test_drop_table():
    with FlightServer(
        verify_client=False,
        make_connection=xo.duckdb.connect,
    ) as main:
        # GIVEN
        data = pa.table(
            {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}
        ).to_pandas()
        main.con.register(data, table_name="users")
        assert main.client.do_action_one(ListTablesAction.name)

        # WHEN
        main.client.do_action_one(DropTableAction.name, action_body={"name": "users"})

        # THEN
        assert not main.client.do_action_one(ListTablesAction.name)


def test_get_schema_query():
    with FlightServer(
        verify_client=False,
        make_connection=xo.duckdb.connect,
    ) as main:
        # GIVEN
        data = pa.table(
            {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}
        ).to_pandas()
        main.con.register(data, table_name="users")
        expected = xo.Schema(
            {
                "id": pa.int64(),
                "name": pa.string(),
            }
        )
        # WHEN
        actual = main.client.do_action_one(
            GetSchemaQueryAction.name, action_body="SELECT * FROM users;"
        )
        # THEN
        assert actual == expected


def test_get_exchange():
    def dummy(df: pd.DataFrame):
        return pd.DataFrame({"row_count": [42]})

    schema_in = xo.schema({"dummy": "int64"})
    dummy_udxf = E.make_udxf(
        dummy,
        schema_in,
        xo.schema({"row_count": "int64"}),
    )
    with FlightServer(exchangers=[dummy_udxf]) as flight_server:
        client = flight_server.client
        exchange = client.do_action_one(
            GetExchangeAction.name, action_body=dummy_udxf.command
        )
        assert exchange is not None


def test_shutdown_action_stops_wait():
    server = FlightServer(make_connection=xo.duckdb.connect)
    server.serve()
    waiter = threading.Thread(target=server.wait)
    waiter.start()

    server.client.do_action_one(ShutdownAction.name)
    waiter.join(timeout=10)

    assert not waiter.is_alive()
    assert server.server is None
    with pytest.raises(pa.flight.FlightUnavailableError):
        server.client.do_action_one(ListTablesAction.name)


def test_shutdown_action_ends_wait_after_a_poll():
    # gRPC runs handlers on native threads, so drive this one from a Python thread
    server = FlightServer(make_connection=xo.duckdb.connect)
    server.serve()
    threading.Timer(
        1.5, lambda: list(ShutdownAction.do_action(server.server, None, None))
    ).start()

    server.wait()

    assert server.server is None


def test_wait_closes_server_on_keyboard_interrupt():
    server = FlightServer(make_connection=xo.duckdb.connect)
    server.serve()
    threading.Timer(0.2, os.kill, (os.getpid(), signal.SIGINT)).start()

    server.wait()

    assert server.server is None
