import time
from datetime import datetime

import toolz

import xorq as xo
from xorq.flight.client import FlightClient


client = FlightClient(port=8815)

commands = client.do_action_one("list-exchanges")
print("Available exchanges:", commands)

command = [
    cmd
    for cmd in commands
    if cmd not in {"echo", "row-sum", "row-sum-append", "url-response-length"}  # :(
][0]

schema_in = xo.schema({"dummy": "int64"})
z = xo.memtable([{"dummy": 1}], schema=schema_in)

do_exchange = toolz.curry(client.do_exchange, command)

while True:
    _, rbr_out = do_exchange(z.to_pyarrow_batches())
    result_df = rbr_out.read_pandas()
    print(f"{datetime.now().isoformat()} count:")
    print(result_df)

    time.sleep(3)
