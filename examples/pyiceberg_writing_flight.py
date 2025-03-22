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


warehouse_path = "/tmp/warehouse"
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

        if namespace not in self.catalog.list_namespaces():
            self.catalog.create_namespace(namespace)

        self.con = xo.duckdb.connect()

    @property
    def tables(self):
        try:
            catalog = load_catalog("default", **self.catalog_params)
            return [table_id[1] for table_id in catalog.list_tables((namespace,))]
        except Exception as e:
            print(f"Error listing tables: {e}")
            return []

    def create_table(self, table_name, data):
        try:
            full_table_name = f"{namespace}.{table_name}"
            catalog = load_catalog("default", **self.catalog_params)

            if catalog.table_exists(full_table_name):
                catalog.drop_table(full_table_name)

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
                identifier=full_table_name, schema=iceberg_schema
            )

            iceberg_table.append(data)
            return True
        except Exception as e:
            print(f"Error creating table: {e}")
            return False

    def insert(self, table_name, data):
        try:
            full_table_name = f"{namespace}.{table_name}"
            catalog = load_catalog("default", **self.catalog_params)

            if catalog.table_exists(full_table_name):
                iceberg_table = catalog.load_table(full_table_name)
                iceberg_table.append(data)
                return True
            else:
                return False
        except Exception as e:
            print(f"Error inserting data: {e}")
            return False

    def to_pyarrow_batches(self, expr, **kwargs):
        try:
            table_name = "concurrent_test"

            print(f"Using known table name: {table_name}")
            full_table_name = f"{namespace}.{table_name}"

            catalog = load_catalog("default", **self.catalog_params)

            if catalog.table_exists(full_table_name):
                print(f"Table {full_table_name} exists")
                iceberg_table = catalog.load_table(full_table_name)
                iceberg_table.refresh()
                reader = iceberg_table.scan().to_arrow_batch_reader()
                return reader
            else:
                print(f"Table {full_table_name} does not exist")

            return pa.RecordBatchReader.from_batches(pa.schema([]), [])
        except Exception as e:
            print(f"Error reading data: {e}")
            import traceback

            traceback.print_exc()
            return pa.RecordBatchReader.from_batches(pa.schema([]), [])


def run_server(warehouse_path, table_name, port):
    server = FlightServer(
        FlightUrl(port=port),
        connection=partial(IcebergConnector, warehouse_path),
    )
    server.serve()

    server.server._conn.create_table(
        table_name,
        pa.Table.from_pylist(
            [],
            schema=pa.schema(
                [
                    pa.field("id", pa.int64(), nullable=False),
                    pa.field("value", pa.string(), nullable=False),
                ]
            ),
        ),
    )

    print(f"Flight server started at grpc://localhost:{port}")
    while server.server is not None:
        time.sleep(1)


def run_reader(table_name, port):
    client = FlightClient(port=port)

    while True:
        try:
            table_ref = xo.table({"id": int, "value": str}, name=table_name)
            print(f"Reading table with ref: {table_ref}, name: {table_name}")
            result = client.execute_query(table_ref)

            if result is not None:
                df = result.to_pandas()
                count = len(df)
                print(f"{datetime.now().isoformat()} count: {count}")
                if count > 0:
                    print(f"Latest record: {df.iloc[-1].to_dict()}")
            else:
                print(f"{datetime.now().isoformat()} count: 0")

        except Exception as e:
            print(f"Error reading data: {e}")
            print(f"{datetime.now().isoformat()} count: 0")

        time.sleep(1)


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
