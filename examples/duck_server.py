from pathlib import Path

import xorq as xo
from xorq.flight import FlightServer, FlightUrl


db_path = Path("./multi_duck.db").absolute()
flight_port = 8816


db_path.unlink(missing_ok=True)
flight_server = FlightServer(
    FlightUrl(port=flight_port), connection=lambda: xo.duckdb.connect(db_path)
)
flight_server.serve()
print(f"DuckDB Flight server started at grpc://localhost:{flight_port}")
