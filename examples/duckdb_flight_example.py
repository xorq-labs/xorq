"""Concurrent Flight server with DuckDB supporting concurrent read/write operations.

Traditional approach: You would build a multi-threaded Flight server backed by
DuckDB, manually manage connection pooling, handle concurrent access to the
database, and write separate CLI entry points for serving, reading, and writing.

With xorq: FlightServer accepts a make_connection factory for the DuckDB
backend and handles concurrent reads and writes. The example provides a CLI
interface (serve/read/write) using standard argparse, with FlightClient
handling all protocol details.
"""

import argparse
import random
import time
import traceback
from datetime import datetime
from functools import partial
from pathlib import Path

import pyarrow as pa

import xorq.api as xo
from xorq.flight import (
    FlightServer,
    FlightUrl,
)
from xorq.flight.client import (
    FlightClient,
)


db_path = "multi_duck.db"
port = 8816
table_name = "concurrent_test"


def read_data(expr, client):
    result = client.execute(expr)
    try:
        ((count,),) = result.to_pandas().values
        print(f"{datetime.now().isoformat()} count: {count}")
    except Exception:
        print(f"result: {result}")
        traceback.print_exc()


def write_data(table_name, client):
    data = pa.Table.from_pylist(
        (
            {
                "id": int(time.time()),
                "value": f"val-{random.randint(100, 999)}",
            },
        )
    )
    client.upload_data(table_name, data)
    print(f"{datetime.now().isoformat()} - Uploaded data: {data.to_pydict()}")


def run_server(db_path, table_name, port):
    db_path = Path(db_path).absolute()
    db_path.unlink(missing_ok=True)
    flight_server = FlightServer(
        FlightUrl(port=port),
        make_connection=partial(xo.duckdb.connect, db_path),
    )
    flight_server.serve()
    flight_server.server._conn.create_table(
        table_name, schema=xo.schema({"id": int, "value": str})
    )
    print(f"DuckDB Flight server started at grpc://localhost:{port}")
    while flight_server.server is not None:
        time.sleep(1)


def run_reader(table_name, port):
    expr = xo.table({"id": int}, name=table_name).count()
    client = FlightClient(port=port)
    while True:
        read_data(expr, client)
        time.sleep(1)


def run_writer(table_name, port):
    client = FlightClient(port=port)
    while True:
        write_data(table_name, client)
        time.sleep(1)


def parse_args(override=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("serve", "read", "write"),
    )
    parser.add_argument(
        "-d",
        "--db-path",
        default=db_path,
    )
    parser.add_argument(
        "-p",
        "--port",
        default=port,
        type=int,
    )
    parser.add_argument(
        "-n",
        "--table-name",
        default=table_name,
    )
    args = parser.parse_args(override)
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.command == "serve":
        run_server(args.db_path, args.table_name, args.port)
    elif args.command == "read":
        run_reader(args.table_name, args.port)
    elif args.command == "write":
        run_writer(args.table_name, args.port)
    else:
        raise ValueError
