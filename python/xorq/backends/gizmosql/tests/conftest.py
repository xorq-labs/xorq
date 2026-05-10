from __future__ import annotations

import contextlib
import datetime
from pathlib import Path

import pytest

import xorq.api as xo
from xorq.vendor.ibis import util


# ── Constants ────────────────────────────────────────────────────────────────
GIZMOSQL_USERNAME = "ibis"
GIZMOSQL_PASSWORD = "ibis_password"

ROOT_DIR = Path(__file__).resolve().parents[5]  # xorq repo root
DATA_DIR = ROOT_DIR / "ci" / "ibis-testing-data"

PARQUET_TABLES = (
    "functional_alltypes",
    "diamonds",
    "batting",
    "awards_players",
    "astronauts",
)


def _generate_self_signed_cert(out_dir: Path) -> tuple[Path, Path]:
    """Mint a self-signed RSA cert + key for ``localhost`` and write both as
    PEM into ``out_dir``. Used to enable TLS on the test server so the test
    suite still exercises the encrypted Flight SQL path (the previous Docker
    image baked this in; the bare server binary needs explicit cert files).
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    cert_path = out_dir / "gizmosql_test_cert.pem"
    key_path = out_dir / "gizmosql_test_key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


@pytest.fixture(scope="session")
def gizmosql_server(tmp_path_factory):
    """Start the GizmoSQL server as a managed subprocess via the
    [gizmosql](https://pypi.org/project/gizmosql/) PyPI package.

    The package downloads + caches the matching server binary on first use,
    auto-picks a free port, and tears the server down on exit — no Docker
    is needed for the test fixture. A short-lived self-signed cert is
    generated at session start and passed via ``--tls`` so the encrypted
    Flight SQL path is still exercised by the tests.
    """
    gizmosql = pytest.importorskip("gizmosql")

    cert_dir = tmp_path_factory.mktemp("gizmosql-tls")
    cert_path, key_path = _generate_self_signed_cert(cert_dir)

    with gizmosql.Server(
        username=GIZMOSQL_USERNAME,
        password=GIZMOSQL_PASSWORD,
        extra_args=["--tls", str(cert_path), str(key_path)],
    ) as srv:
        yield srv


@pytest.fixture(scope="session")
def con(gizmosql_server):
    """GizmoSQL connection with test data loaded."""
    conn = xo.gizmosql.connect(
        host=gizmosql_server.host,
        user=gizmosql_server.username,
        password=gizmosql_server.password,
        port=gizmosql_server.port,
        use_encryption=True,
        # The test cert is self-signed and freshly minted per session; skip
        # cert-chain verification rather than wiring a CA bundle into the
        # client for what is purely a loopback test connection.
        disable_certificate_verification=True,
    )

    # Load standard test tables from parquet
    parquet_dir = DATA_DIR / "parquet"
    for table_name in PARQUET_TABLES:
        parquet_path = parquet_dir / f"{table_name}.parquet"
        if parquet_path.exists():
            conn.read_parquet(parquet_path, table_name=table_name)

    return conn


@pytest.fixture(scope="session")
def alltypes(con):
    return con.table("functional_alltypes")


@pytest.fixture(scope="session")
def alltypes_df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope="session")
def awards_players(con):
    return con.table("awards_players")


@pytest.fixture(scope="session")
def batting(con):
    return con.table("batting")


@pytest.fixture
def temp_table(con):
    name = util.gen_name("temp_table")
    yield name
    with contextlib.suppress(Exception):
        with con._safe_raw_sql(f'DROP TABLE IF EXISTS "{name}"'):
            pass
