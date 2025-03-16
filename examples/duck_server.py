import time

import pandas as pd

import xorq as xo
from xorq.flight import FlightServer, FlightUrl
from xorq.flight.exchanger import make_udxf


db_path = "multi_duck.db"
flight_port = 8815


writer_con = xo.duckdb.connect(db_path)
writer_con.raw_sql("""
    CREATE TABLE IF NOT EXISTS concurrent_test (id INTEGER, value VARCHAR);
""")
writer_con.raw_sql("INSERT INTO concurrent_test VALUES (1, 'initial'), (2, 'data');")


def reader_udxf(df: pd.DataFrame):
    reader_con = xo.duckdb.connect(db_path)
    return reader_con.tables["concurrent_test"].agg(row_count=xo._.id.count()).execute()


schema_in = xo.schema({"dummy": "int64"})
schema_out = xo.schema({"row_count": "int64"})

duckdb_reader_udxf = make_udxf(
    reader_udxf, schema_in.to_pyarrow(), schema_out.to_pyarrow()
)

flight_server = FlightServer(
    FlightUrl(port=flight_port), exchangers=[duckdb_reader_udxf]
)
flight_server.serve()

print("DuckDB Flight server started at grpc://localhost:8815")

while True:
    time.sleep(5)
    writer_con.raw_sql("""
        INSERT INTO concurrent_test
        SELECT max(id)+1, 'val-' || cast(random()*1000 as int)
        FROM concurrent_test
    """)
    print("Inserted incremental row.")
