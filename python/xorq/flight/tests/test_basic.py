import pytest

import xorq as xo
from xorq.common.utils.tls_utils import TLSKwargs
from xorq.flight import BasicAuth, FlightServer
from xorq.flight.tests.test_server import make_flight_url


@pytest.mark.parametrize("auth", [None, BasicAuth("username", "password")])
@pytest.mark.parametrize("verify_client", [False, True])
def test_connect(auth, verify_client):
    tls_kwargs = TLSKwargs.from_common_name(verify_client=verify_client)
    flight_url = make_flight_url(None, scheme="grpc+tls", auth=auth)

    with FlightServer(
        flight_url=flight_url,
        verify_client=verify_client,
        auth=auth,
        **tls_kwargs.server_kwargs,
    ) as _:
        con = xo.flight.connect(flight_url, tls_kwargs)
        assert con is not None
        con.list_tables()
