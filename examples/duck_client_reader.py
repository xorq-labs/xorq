import time
from datetime import datetime

import xorq as xo
from xorq.flight.client import FlightClient


name = "concurrent_test"
port = 8816
expr = xo.table({"id": "int64"}, name=name).count()


def read_data(expr, client):
    result = client.execute_query(expr)
    df = result.to_pandas()
    print(f"{datetime.now().isoformat()} count:")
    print(df)


client = FlightClient(port=port)
while True:
    read_data(expr, client)
    time.sleep(1)
