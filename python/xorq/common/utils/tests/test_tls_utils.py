import socket
from pathlib import Path

import pytest
from attr import (
    evolve,
)
from cryptography.exceptions import (
    InvalidSignature,
)

import xorq as xo
from xorq.common.utils.tls_utils import (
    TLSCert,
    TLSKwargs,
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
    tls_kwargs = TLSKwargs.from_common_name(verify_client=False)
    tls_kwargs.ca_tlscert.verify(tls_kwargs.server_tlscert)


def test_mtls_creation():
    tls_kwargs = TLSKwargs.from_common_name(verify_client=True)

    tls_kwargs.ca_tlscert.verify(tls_kwargs.server_tlscert)
    tls_kwargs.ca_tlscert.verify(tls_kwargs.client_tlscert)


@pytest.mark.parametrize(
    "client_values",
    [
        (True, True),
        (False, True),
        (False, False),
    ],
)
def test_tls_kwargs_from_constructor(client_values):
    verify_client, with_client = client_values

    ca_kwargs = {
        "common_name": "root_cert",
    }
    server_kwargs = {
        "common_name": socket.gethostname(),
        "sans": ("localhost",),
    }
    client_kwargs = {
        "common_name": "client",
    }

    ca_tlscert = TLSCert.from_common_name(**ca_kwargs)
    server_tlscert = TLSCert.from_common_name(sign_with=ca_tlscert, **server_kwargs)
    client_tlscert = (
        TLSCert.from_common_name(sign_with=ca_tlscert, **client_kwargs)
        if with_client
        else None
    )
    tls_kwargs = TLSKwargs(verify_client, ca_tlscert, server_tlscert, client_tlscert)

    assert tls_kwargs is not None
    tls_kwargs.ca_tlscert.verify(tls_kwargs.server_tlscert)
    if tls_kwargs.verify_client:
        tls_kwargs.ca_tlscert.verify(tls_kwargs.client_tlscert)


@pytest.mark.parametrize("use_none", (True, False))
def test_tls_kwargs_from_constructor_fails(use_none):
    ca_kwargs = {
        "common_name": "root_cert",
    }
    server_kwargs = {
        "common_name": socket.gethostname(),
        "sans": ("localhost",),
    }

    ca_tlscert = TLSCert.from_common_name(**ca_kwargs)
    server_tlscert = TLSCert.from_common_name(sign_with=ca_tlscert, **server_kwargs)

    with pytest.raises(AssertionError):
        TLSKwargs(True, ca_tlscert, server_tlscert, None) if use_none else TLSKwargs(
            True, ca_tlscert, server_tlscert
        )


def test_client_verify_fails():
    tlscert0 = TLSCert.from_common_name("tlscert0")
    tlscert1 = TLSCert.from_common_name("tlscert1")
    with pytest.raises(InvalidSignature):
        tlscert0.verify(tlscert1)


def test_tls_flight_server():
    tls_kwargs = TLSKwargs.from_common_name(verify_client=False)
    server = xo.flight.FlightServer(
        verify_client=tls_kwargs.verify_client,
        **tls_kwargs.server_kwargs,
    )
    server.serve()

    # test that server can talk to its own client
    client = server.client
    client.do_action_one(xo.flight.action.ListActionsAction.name)

    # test that an independent client can talk to it
    client = xo.flight.client.FlightClient(
        **server.flight_url.client_kwargs,
        **tls_kwargs.client_kwargs,
    )
    client.do_action_one(xo.flight.action.ListActionsAction.name)


def test_tls_flight_server_fails():
    tls_kwargs = TLSKwargs.from_common_name(verify_client=False)
    flight_url = xo.flight.FlightUrl(scheme="grpc+tls")
    with pytest.raises(Exception):
        # grpc+tls but no ca cert
        server = xo.flight.FlightServer(
            flight_url=flight_url,
            verify_client=tls_kwargs.verify_client,
        )

    server = xo.flight.FlightServer(
        flight_url=flight_url,
        verify_client=tls_kwargs.verify_client,
        **tls_kwargs.server_kwargs,
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
    tls_kwargs = TLSKwargs.from_common_name(verify_client=True)
    server = xo.flight.FlightServer(
        verify_client=tls_kwargs.verify_client,
        **tls_kwargs.server_kwargs,
    )
    server.serve()
    # test that its own client can talk to it
    client = server.client
    client.do_action_one(xo.flight.action.ListActionsAction.name)


def test_mtls_flight_client():
    tls_kwargs = TLSKwargs.from_common_name(verify_client=True)

    server = xo.flight.FlightServer(
        verify_client=tls_kwargs.verify_client,
        **tls_kwargs.server_kwargs,
    )
    server.serve()

    # test that an independent client can talk to it
    client = xo.flight.client.FlightClient(
        **server.flight_url.client_kwargs,
        **tls_kwargs.client_kwargs,
    )
    client.do_action_one(xo.flight.action.ListActionsAction.name)


def test_mtls_flight_client_failure():
    tls_kwargs = TLSKwargs.from_common_name(verify_client=True)

    server = xo.flight.FlightServer(
        verify_client=tls_kwargs.verify_client,
        **tls_kwargs.server_kwargs,
    )
    server.serve()

    tlscert = TLSCert.from_common_name()
    bad_client_kwargs = tls_kwargs.client_kwargs | {
        "tls_root_certs": tlscert.cert_bytes,
    }
    with pytest.raises(Exception):
        xo.flight.client.FlightClient(
            **server.flight_url.client_kwargs,
            **bad_client_kwargs,
        )

    tlscert = TLSCert.from_common_name()
    bad_client_kwargs = tls_kwargs.client_kwargs | {
        "cert_chain": tlscert.cert_bytes,
        "private_key": tlscert.private_key_bytes,
    }
    with pytest.raises(Exception):
        xo.flight.client.FlightClient(
            **server.flight_url.client_kwargs,
            **bad_client_kwargs,
        )


@pytest.mark.parametrize("verify_client", (True, False))
@pytest.mark.parametrize("password", (None, b"password"))
def test_tls_kwargs_from_disk_roundtrip(verify_client, password, tmpdir):
    """Test that TLSKwargs can be saved to disk and loaded back correctly."""
    original_tls_kwargs = TLSKwargs.from_common_name(verify_client=verify_client)
    tmpdir = Path(str(tmpdir))

    path_kwargs = {
        f"{name}_path": tmpdir / name
        for name in (
            "ca_cert",
            "ca_private_key",
            "server_cert",
            "server_private_key",
            "client_cert",
            "client_private_key",
        )
    }

    original_tls_kwargs.to_disk(
        **path_kwargs,
        ca_password=password,
        server_password=password,
        client_password=password,
    )

    loaded_tls_kwargs = TLSKwargs.from_disk(
        **path_kwargs,
        verify_client=verify_client,
        ca_password=password,
        server_password=password,
        client_password=password,
    )

    assert loaded_tls_kwargs == original_tls_kwargs
    assert loaded_tls_kwargs != evolve(
        original_tls_kwargs, verify_client=not original_tls_kwargs.verify_client
    )
