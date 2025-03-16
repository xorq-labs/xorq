import time
from datetime import datetime

import xorq as xo


client = xo.flight.client.FlightClient(port=8815)

expr = xo.table({"id": "int64"}, name="concurrent_test").count()

while True:
    result = client.execute_query(expr)
    df = result.to_pandas()

    print(f"{datetime.now().isoformat()} count:")
    print(df)

    time.sleep(1)
