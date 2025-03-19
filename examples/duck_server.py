import xorq as xo
from xorq.flight import FlightServer, FlightUrl


db_path = "multi_duck.db"
flight_port = 8815

flight_server = FlightServer(
    FlightUrl(port=flight_port), connection=lambda: xo.duckdb.connect(db_path)
)
flight_server.serve()

print(f"DuckDB Flight server started at grpc://localhost:{flight_port}")
