import pytest


@pytest.fixture(scope="session")
def tls_kwargs():
    return {
        "ca_kwargs": {
            "common_name": "root_cert",
        },
        "server_kwargs": {
            "common_name": "server",
            "sans": ("localhost",),
        },
    }


@pytest.fixture(scope="session")
def mtls_kwargs(tls_kwargs):
    return tls_kwargs | {
        "client_kwargs": {
            "common_name": "client",
        },
    }
