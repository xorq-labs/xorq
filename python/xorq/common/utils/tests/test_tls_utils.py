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


def test_single_creation():
    TLSCert.from_common_name()


def test_cert_roundtrip():
    tlscert = TLSCert.from_common_name()
    cert = tlscert.cert
    assert cert == pem_decode_cert(pem_encode_cert(cert))


@pytest.mark.parametrize("password", (None, b"password"))
def test_private_key_roundtrip(password):
    tlscert = TLSCert.from_common_name()
    private_key = tlscert.private_key
    assert (
        private_key.private_numbers()
        == pem_decode_private_key(
            pem_encode_private_key(private_key, password), password
        ).private_numbers()
    )


@pytest.mark.parametrize("password", (None, b"password"))
def test_tlscert_roundtrip(password, tmpdir):
    tlscert = TLSCert.from_common_name()
    tmpdir = Path(str(tmpdir))
    cert_path, private_key_path = (
        tmpdir.joinpath("cert"),
        tmpdir.joinpath("private_key"),
    )
    tlscert.to_disk(cert_path, private_key_path, password)
    other = TLSCert.from_disk(cert_path, private_key_path, password)
    assert tlscert == other


def test_tls_creation():
    (tlscerts, *_) = ((ca_tlscert, server_tlscert), client_kwargs, server_kwargs) = (
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
    tlscert0 = TLSCert.from_common_name("tlscert0")
    tlscert1 = TLSCert.from_common_name("tlscert1")
    with pytest.raises(InvalidSignature):
        tlscert0.verify(tlscert1)


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
    server = xo.flight.FlightServer(
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
    tlscert = TLSCert.from_common_name()
    with pytest.raises(Exception):
        client = xo.flight.client.FlightClient(
            **server.flight_url.client_kwargs,
            tls_root_certs=tlscert.cert_bytes,
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
    server = xo.flight.FlightServer(
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
    server = xo.flight.FlightServer(
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
    server = xo.flight.FlightServer(
        verify_client=True,
        **server_kwargs,
    )
    server.serve()

    tlscert = TLSCert.from_common_name()
    bad_client_kwargs = client_kwargs | {
        "tls_root_certs": tlscert.cert_bytes,
    }
    with pytest.raises(Exception):
        xo.flight.client.FlightClient(
            **server.flight_url.client_kwargs,
            **bad_client_kwargs,
        )

    tlscert = TLSCert.from_common_name()
    bad_client_kwargs = client_kwargs | {
        "cert_chain": tlscert.cert_bytes,
        "private_key": tlscert.private_key_bytes,
    }
    with pytest.raises(Exception):
        xo.flight.client.FlightClient(
            **server.flight_url.client_kwargs,
            **bad_client_kwargs,
        )
