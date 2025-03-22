import os
import random
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import pyarrow as pa
from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import LongType, NestedField, StringType

import xorq as xo
from xorq.flight import FlightServer, FlightUrl
from xorq.flight.client import FlightClient


warehouse_path = "warehouse"
port = 8816
table_name = "concurrent_test"
namespace = "default"


class IcebergConnector:
    def __init__(self, warehouse_path):
        self.warehouse_path = Path(warehouse_path).absolute()
        os.makedirs(self.warehouse_path, exist_ok=True)

        self.catalog_params = {
            "type": "sql",
            "uri": f"sqlite:///{self.warehouse_path}/pyiceberg_catalog.db",
            "warehouse": f"file://{self.warehouse_path}",
        }

        self.catalog = load_catalog("default", **self.catalog_params)
        namespaces = [n[0] for n in self.catalog.list_namespaces()]

        if namespace not in namespaces:
            self.catalog.create_namespace(namespace)

        self.con = xo.duckdb.connect()
        self.con.raw_sql("INSTALL iceberg;")
        self.con.raw_sql("LOAD iceberg;")
        self.con.raw_sql("SET unsafe_enable_version_guessing=true;")
        self._reflect_views()

    def _reflect_views(self):
        table_identifiers = self.catalog.list_tables("default")
        table_names = [t[1] for t in table_identifiers]
        for t in table_names:
            full_table_name = f"{namespace}.{t}"
            table_path = f"{self.warehouse_path}/{namespace}.db/{t}"
            self.con.raw_sql("INSTALL iceberg;")
            self.con.raw_sql("LOAD iceberg;")
            self.con.raw_sql("SET unsafe_enable_version_guessing=true;")

            self.con.raw_sql(f"""
                CREATE OR REPLACE VIEW {t} AS
                SELECT * FROM iceberg_scan(
                    '{table_path}', 
                    version='?',
                    allow_moved_paths=true
                )
            """)
            print(f"Created view: {t} for table: {full_table_name}")

    @property
    def tables(self):
        catalog = load_catalog("default", **self.catalog_params)
        return [table_id[1] for table_id in catalog.list_tables((namespace,))]

    def create_table(self, table_name, data):
        full_table_name = f"{namespace}.{table_name}"
        catalog = load_catalog("default", **self.catalog_params)

        if catalog.table_exists(full_table_name):
            return True

        iceberg_fields = []
        for i, field in enumerate(data.schema, 1):
            if pa.types.is_int64(field.type):
                iceberg_type = LongType()
            elif pa.types.is_string(field.type):
                iceberg_type = StringType()
            else:
                iceberg_type = StringType()

            iceberg_fields.append(
                NestedField(i, field.name, iceberg_type, required=True)
            )

        iceberg_schema = Schema(*iceberg_fields)
        iceberg_table = catalog.create_table(
            identifier=full_table_name,
            schema=iceberg_schema,
        )

        iceberg_table.append(data)
        return True

    def insert(self, table_name, data):
        full_table_name = f"{namespace}.{table_name}"

        iceberg_table = self.catalog.load_table(full_table_name)
        iceberg_table.refresh()

        with iceberg_table.transaction() as transaction:
            transaction.append(data)

        return True

    def to_pyarrow_batches(self, expr, **kwargs):
        self._reflect_views()

        return self.con.to_pyarrow_batches(expr)

    def sql(self, sql):
        self._reflect_views()
        return self.con.con.sql(sql)


def run_server(warehouse_path, table_name, port):
    server = FlightServer(
        FlightUrl(port=port),
        connection=partial(IcebergConnector, warehouse_path),
    )
    server.serve()
    table = pa.Table.from_pylist(
        [
            {"id": 1, "value": "sample_value_1"},
            {"id": 2, "value": "sample_value_2"},
        ],
        schema=pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("value", pa.string(), nullable=False),
            ]
        ),
    )
    server.server._conn.create_table(table_name, table)

    print(f"Flight server started at grpc://localhost:{port}")
    while server.server is not None:
        time.sleep(1)


def run_reader(table_name, port):
    client = FlightClient(port=port)

    while True:
        expr = xo.table({"id": int, "value": str}, name=table_name).count().as_table()
        result = client.execute_query(expr)
        print(f"{datetime.now().isoformat()} count: {result}")


def run_writer(table_name, port):
    client = FlightClient(port=port)

    while True:
        try:
            data = pa.Table.from_pylist(
                [{"id": int(time.time()), "value": f"val-{random.randint(100, 999)}"}],
                schema=pa.schema(
                    [
                        pa.field("id", pa.int64(), nullable=False),
                        pa.field("value", pa.string(), nullable=False),
                    ]
                ),
            )

            client.upload_data(table_name, data)
            print(f"{datetime.now().isoformat()} - Uploaded: {data.to_pydict()}")
        except Exception as e:
            print(f"Error writing data: {e}")

        time.sleep(1)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("serve", "read", "write"))
    parser.add_argument("-w", "--warehouse-path", default=warehouse_path)
    parser.add_argument("-p", "--port", default=port, type=int)
    parser.add_argument("-n", "--table-name", default=table_name)

    args = parser.parse_args()

    if args.command == "serve":
        run_server(args.warehouse_path, args.table_name, args.port)
    elif args.command == "read":
        run_reader(args.table_name, args.port)
    elif args.command == "write":
        run_writer(args.table_name, args.port)


if __name__ == "__main__":
    main()
