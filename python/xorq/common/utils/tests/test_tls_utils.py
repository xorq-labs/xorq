from pathlib import Path

import pytest
from cryptography.exceptions import (
    InvalidSignature,
)

import xorq as xo
from xorq.common.utils.tls_utils import (
    TLSCert,
    pem_decode_cert,
    pem_decode_private_key,
    pem_encode_cert,
    pem_encode_private_key,
)


def test_cert_roundtrip():
    tls_cert = TLSCert.from_common_name()
    cert = tls_cert.cert
    assert cert == pem_decode_cert(pem_encode_cert(cert))


@pytest.mark.parametrize("password", (None, b"password"))
def test_private_key_roundtrip(password):
    tls_cert = TLSCert.from_common_name()
    private_key = tls_cert.private_key
    assert (
        private_key.private_numbers()
        == pem_decode_private_key(
            pem_encode_private_key(private_key, password), password
        ).private_numbers()
    )


@pytest.mark.parametrize("password", (None, b"password"))
def test_tlscert_roundtrip(password, tmpdir):
    tls_cert = TLSCert.from_common_name()
    tmpdir = Path(str(tmpdir))
    cert_path, private_key_path = (
        tmpdir.joinpath("cert"),
        tmpdir.joinpath("private_key"),
    )
    tls_cert.to_disk(cert_path, private_key_path, password)
    other = TLSCert.from_disk(cert_path, private_key_path, password)
    assert tls_cert == other


def test_single_creation():
    TLSCert.from_common_name()


def test_tls_creation():
    (tls_certs, *_) = ((ca_tlscert, server_tlscert), client_kwargs, server_kwargs) = (
        TLSCert.create_tls_kwargs(
            ca_kwargs={
                "common_name": "root_cert",
            },
            server_kwargs={
                "common_name": "server",
                "sans": ("localhost",),
            },
        )
    )
    ca_tlscert.verify(server_tlscert)


def test_mtls_creation():
    ((ca_tlscert, server_tlscert, client_tlscert), client_kwargs, server_kwargs) = (
        TLSCert.create_mtls_kwargs(
            ca_kwargs={
                "common_name": "root_cert",
            },
            server_kwargs={
                "common_name": "server",
                "sans": ("localhost",),
            },
            client_kwargs={
                "common_name": "client",
            },
        )
    )
    ca_tlscert.verify(server_tlscert)
    ca_tlscert.verify(client_tlscert)


def test_client_verify_fails():
    pair0 = TLSCert.from_common_name("pair0")
    pair1 = TLSCert.from_common_name("pair1")
    with pytest.raises(InvalidSignature):
        pair0.verify(pair1)


def test_tls_flight_server():
    ((ca_tlscert, server_tlscert), server_kwargs, client_kwargs) = (
        TLSCert.create_tls_kwargs(
            ca_kwargs={
                "common_name": "root_cert",
            },
            server_kwargs={
                "common_name": "server",
                "sans": ("localhost",),
            },
        )
    )
    flight_url = xo.flight.FlightUrl(scheme="grpc+tls")
    server = xo.flight.FlightServer(
        # FIXME: in FlightServer: auto set scheme if flight_url is not passed
        flight_url=flight_url,
        verify_client=False,
        **server_kwargs,
    )
    server.serve()

    # test that server can talk to its own client
    client = server.client
    client.do_action_one(xo.flight.action.ListActionsAction.name)

    # test that an independent client can talk to it
    client = xo.flight.client.FlightClient(
        **server.flight_url.client_kwargs,
        **client_kwargs,
    )
    client.do_action_one(xo.flight.action.ListActionsAction.name)


def test_tls_flight_server_fails():
    ((ca_tlscert, server_tlscert), server_kwargs, client_kwargs) = (
        TLSCert.create_tls_kwargs(
            ca_kwargs={
                "common_name": "root_cert",
            },
            server_kwargs={
                "common_name": "server",
                "sans": ("localhost",),
            },
        )
    )
    flight_url = xo.flight.FlightUrl(scheme="grpc+tls")
    with pytest.raises(Exception):
        # grpc+tls but no ca cert
        server = xo.flight.FlightServer(
            flight_url=flight_url,
            verify_client=False,
        )

    server = xo.flight.FlightServer(
        flight_url=flight_url,
        verify_client=False,
        **server_kwargs,
    )
    server.serve()
    # server can talk to itself
    client = server.client
    client.do_action_one(xo.flight.action.ListActionsAction.name)

    # but client doesn't trust signer
    tls_cert = TLSCert.from_common_name()
    with pytest.raises(Exception):
        client = xo.flight.client.FlightClient(
            **server.flight_url.client_kwargs,
            tls_root_certs=tls_cert.cert_bytes,
        )


def test_mtls_flight_server():
    ((ca_tlscert, server_tlscert, client_tlscert), server_kwargs, client_kwargs) = (
        TLSCert.create_mtls_kwargs(
            ca_kwargs={
                "common_name": "root_cert",
            },
            server_kwargs={
                "common_name": "server",
                "sans": ("localhost",),
            },
            client_kwargs={
                "common_name": "client",
            },
        )
    )
    flight_url = xo.flight.FlightUrl(scheme="grpc+tls")
    server = xo.flight.FlightServer(
        # FIXME: in FlightServer: auto set scheme if flight_url is not passed
        flight_url=flight_url,
        verify_client=True,
        **server_kwargs,
    )
    server.serve()
    # test that its own client can talk to it
    client = server.client
    client.do_action_one(xo.flight.action.ListActionsAction.name)


def test_mtls_flight_client():
    ((ca_tlscert, server_tlscert, client_tlscert), server_kwargs, client_kwargs) = (
        TLSCert.create_mtls_kwargs(
            ca_kwargs={
                "common_name": "root_cert",
            },
            server_kwargs={
                "common_name": "server",
                "sans": ("localhost",),
            },
            client_kwargs={
                "common_name": "client",
            },
        )
    )
    flight_url = xo.flight.FlightUrl(scheme="grpc+tls")
    server = xo.flight.FlightServer(
        # FIXME: in FlightServer: auto set scheme if flight_url is not passed
        flight_url=flight_url,
        verify_client=True,
        **server_kwargs,
    )
    server.serve()

    # test that an independent client can talk to it
    client = xo.flight.client.FlightClient(
        **server.flight_url.client_kwargs,
        **client_kwargs,
    )
    client.do_action_one(xo.flight.action.ListActionsAction.name)


def test_mtls_flight_client_failure():
    ((ca_tlscert, server_tlscert, client_tlscert), server_kwargs, client_kwargs) = (
        TLSCert.create_mtls_kwargs(
            ca_kwargs={
                "common_name": "root_cert",
            },
            server_kwargs={
                "common_name": "server",
                "sans": ("localhost",),
            },
            client_kwargs={
                "common_name": "client",
            },
        )
    )
    flight_url = xo.flight.FlightUrl(scheme="grpc+tls")
    server = xo.flight.FlightServer(
        # FIXME: in FlightServer: auto set scheme if flight_url is not passed
        flight_url=flight_url,
        verify_client=True,
        **server_kwargs,
    )
    server.serve()

    tls_cert = TLSCert.from_common_name()
    bad_client_kwargs = client_kwargs | {
        "tls_root_certs": tls_cert.cert_bytes,
    }
    with pytest.raises(Exception):
        xo.flight.client.FlightClient(
            **server.flight_url.client_kwargs,
            **bad_client_kwargs,
        )

    tls_cert = TLSCert.from_common_name()
    bad_client_kwargs = client_kwargs | {
        "cert_chain": tls_cert.cert_bytes,
        "private_key": tls_cert.private_key_bytes,
    }
    with pytest.raises(Exception):
        xo.flight.client.FlightClient(
            **server.flight_url.client_kwargs,
            **bad_client_kwargs,
        )
