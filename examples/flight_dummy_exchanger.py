import pandas as pd

import xorq as xo
from xorq.flight import FlightServer, FlightUrl
from xorq.flight.action import AddExchangeAction
from xorq.flight.exchanger import make_udxf


flight_port = 8815


def dummy(df: pd.DataFrame):
    return pd.DataFrame({"row_count": [42]})


schema_in = xo.schema({"dummy": "int64"})
schema_out = xo.schema({"row_count": "int64"})

dummy_udxf = make_udxf(dummy, schema_in.to_pyarrow(), schema_out.to_pyarrow())

flight_server = FlightServer(FlightUrl(port=flight_port))
flight_server.serve()

client = flight_server.client
client.do_action(AddExchangeAction.name, dummy_udxf, options=client._options)


def dummy_do_exchange(input_data):
    input_table = xo.memtable(input_data, schema=schema_in)
    rbr_in = input_table.to_pyarrow_batches()
    _, rbr_out = client.do_exchange(dummy_udxf.command, rbr_in)
    return rbr_out.read_pandas()


dummy_do_exchange(xo.memtable({"dummy": [0]}, schema=schema_in))
