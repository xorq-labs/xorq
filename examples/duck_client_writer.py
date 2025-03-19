import random
import time
from datetime import datetime

import pyarrow as pa

from xorq.flight.client import FlightClient


name = "concurrent_test"
port = 8816


def write_data(name, client):
    data = pa.Table.from_pylist(
        (
            {
                "id": int(time.time()),
                "value": f"val-{random.randint(100, 999)}",
            },
        )
    )
    client.upload_data(name, data)
    print(f"{datetime.now().isoformat()} - Uploaded data: {data.to_pydict()}")


client = FlightClient(port=port)
while True:
    write_data(name, client)
    time.sleep(1)
