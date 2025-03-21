import pyarrow as pa

import xorq as xo
from xorq.flight import FlightServer
from xorq.flight.action import DropTableAction, ListTablesAction
from xorq.flight.tests.test_server import make_flight_url


def test_list_tables_kwargs():
    flight_url = make_flight_url(8816)

    with FlightServer(
        flight_url=flight_url,
        verify_client=False,
        connection=xo.duckdb.connect,
    ) as main:
        # GIVEN
        data = pa.table(
            {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}
        ).to_pandas()
        main.con.register(data, table_name="users")
        expected = ["users"]

        # WHEN
        actual = main.client.do_action_one(ListTablesAction.name)

        # THEN
        assert actual == expected


def test_drop_table():
    flight_url = make_flight_url(8816)

    with FlightServer(
        flight_url=flight_url,
        verify_client=False,
        connection=xo.duckdb.connect,
    ) as main:
        # GIVEN
        data = pa.table(
            {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}
        ).to_pandas()
        main.con.register(data, table_name="users")
        assert main.client.do_action_one(ListTablesAction.name)

        # WHEN
        main.client.do_action_one(
            DropTableAction.name, action_body={"table_name": "users"}
        )

        # THEN
        assert not main.client.do_action_one(ListTablesAction.name)
