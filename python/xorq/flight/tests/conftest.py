import pytest

from xorq.common.utils.tls_utils import create_tls_keypair


@pytest.fixture(scope="session")
def tls_key_pair(tmp_path_factory):
    tls_dir = tmp_path_factory.mktemp("tls")

    cert_file = str(tls_dir / "server.crt")
    key_file = str(tls_dir / "server.key")

    create_tls_keypair(cert_file, key_file)

    return cert_file, key_file
