import random
import time
from datetime import datetime

import pyarrow as pa

from xorq.flight.client import FlightClient


client = FlightClient(port=8815)

table_name = "concurrent_test"

while True:
    data = pa.Table.from_pydict(
        {"id": [int(time.time())], "value": [f"val-{random.randint(100, 999)}"]}
    )
    client.upload_data(table_name, data)

    print(f"{datetime.now().isoformat()} - Uploaded data: {data.to_pydict()}")

    time.sleep(1)
