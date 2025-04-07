import pyarrow as pa

import xorq as xo
from xorq.flight import FlightServer
from xorq.flight.action import DropTableAction, GetSchemaQueryAction, ListTablesAction


def test_list_tables_kwargs():
    with FlightServer(
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
    with FlightServer(
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
        main.client.do_action_one(DropTableAction.name, action_body={"name": "users"})

        # THEN
        assert not main.client.do_action_one(ListTablesAction.name)


def test_get_schema_query():
    with FlightServer(
        verify_client=False,
        connection=xo.duckdb.connect,
    ) as main:
        # GIVEN
        data = pa.table(
            {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}
        ).to_pandas()
        main.con.register(data, table_name="users")
        expected = {
            "id": pa.int64(),
            "name": pa.string(),
        }
        # WHEN
        actual = main.client.do_action_one(
            GetSchemaQueryAction.name, action_body="SELECT * FROM users;"
        )
        # THEN
        assert actual == expected
