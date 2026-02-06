"""Simple Flight server with a dummy UDXF that returns fixed data.

Traditional approach: You would implement a pyarrow.flight.FlightServerBase
subclass, handle do_exchange manually, and manage serialization and schema
negotiation yourself. Every new exchanger requires boilerplate protocol code.

With xorq: Define an exchanger function with input/output schemas and pass it
to make_udxf. Register it with FlightServer via a list of exchangers, and the
framework handles all protocol details, so you only write the transform logic.
"""

import pandas as pd
import toolz

import xorq.api as xo
from xorq.flight import FlightServer
from xorq.flight.exchanger import make_udxf


def dummy(df: pd.DataFrame):
    return pd.DataFrame({"row_count": [42]})


schema_in = xo.schema({"dummy": "int64"})
schema_out = xo.schema({"row_count": "int64"})
dummy_udxf = make_udxf(dummy, schema_in, schema_out)
flight_server = FlightServer(exchangers=[dummy_udxf])


if __name__ in ("__pytest_main__", "__main__"):
    flight_server.serve()
    client = flight_server.client
    do_exchange = toolz.curry(client.do_exchange, dummy_udxf.command)
    do_exchange(xo.memtable({"dummy": [0]}, schema=schema_in))
    pytest_examples_passed = True
