import xorq as xo
from xorq.flight import FlightServer
from xorq.flight.tests.test_server import make_flight_url


def test_connect():
    flight_url = make_flight_url(None)

    with FlightServer(
        flight_url=flight_url,
        connection=xo.duckdb.connect,
    ) as _:
        con = xo.flight.connect(host=flight_url.host, port=flight_url.port)
        assert con is not None
