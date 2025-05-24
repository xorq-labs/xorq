import pytest

from xorq.common.utils.tls_utils import make_and_write_tls_cert_and_key


@pytest.fixture(scope="session")
def tls_key_pair(tmp_path_factory):
    tls_dir = tmp_path_factory.mktemp("tls")

    cert_file = str(tls_dir / "server.crt")
    key_file = str(tls_dir / "server.key")

    make_and_write_tls_cert_and_key(cert_file, key_file)

    return cert_file, key_file
