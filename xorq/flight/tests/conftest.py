"""Utilities for generating self-signed TLS certificates using pyOpenSSL."""

import logging
import os
import socket
from pathlib import Path

import pytest
from OpenSSL import crypto


logger = logging.getLogger()


def _gen_pyopenssl(common_name: str) -> tuple[bytes, bytes]:
    """Generate a self-signed certificate using pyOpenSSL."""

    # Generate RSA private key
    private_key = crypto.PKey()
    private_key.generate_key(crypto.TYPE_RSA, 2048)

    # Generate X.509 certificate
    cert = crypto.X509()

    # Set certificate subject and issuer to be the same (self-signed)
    cert.get_subject().CN = common_name
    cert.set_issuer(cert.get_subject())

    # Set public key
    cert.set_pubkey(private_key)

    # Set certificate serial number
    cert.set_serial_number(int.from_bytes(os.urandom(8), "big"))

    # Set validity period (valid from now to 5 years in the future)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(5 * 365 * 24 * 60 * 60)  # 5 years

    # Add subject alternative names
    san_list = [
        f"DNS:{common_name}",
        f"DNS:*.{common_name}",
        "DNS:localhost",
        "DNS:*.localhost",
    ]
    hostname = socket.gethostname()
    if hostname != common_name:
        san_list.append(f"DNS:{hostname}")
        san_list.append(f"DNS:*.{hostname}")

    san_extension = crypto.X509Extension(
        b"subjectAltName", False, ", ".join(san_list).encode()
    )
    cert.add_extensions(
        [san_extension, crypto.X509Extension(b"basicConstraints", True, b"CA:FALSE")]
    )

    # Sign the certificate with the private key
    cert.sign(private_key, "sha256")

    # Convert certificate and private key to PEM format
    cert_pem = crypto.dump_certificate(crypto.FILETYPE_PEM, cert)
    private_key_pem = crypto.dump_privatekey(crypto.FILETYPE_PEM, private_key)

    return cert_pem, private_key_pem


def gen_self_signed_cert(common_name: str) -> tuple[bytes, bytes]:
    """Return (cert, key) as ASCII PEM strings using pyOpenSSL."""
    return _gen_pyopenssl(common_name=common_name)


def create_tls_keypair(
    cert_file: str,
    key_file: str,
    overwrite: bool = False,
    common_name: str = socket.gethostname(),
):
    """Create a self-signed TLS key pair and write to disk."""

    cert_file_path = Path(cert_file)
    key_file_path = Path(key_file)

    if cert_file_path.exists() or key_file_path.exists():
        if not overwrite:
            raise RuntimeError(
                f"The TLS Cert file(s): '{cert_file_path}' or '{key_file_path}' exist - "
                "and overwrite is False, aborting."
            )

        cert_file_path.unlink(missing_ok=True)
        key_file_path.unlink(missing_ok=True)

    cert, key = gen_self_signed_cert(common_name=common_name)

    cert_file_path.parent.mkdir(parents=True, exist_ok=True)
    with cert_file_path.open(mode="wb") as cert_file:
        cert_file.write(cert)

    with key_file_path.open(mode="wb") as key_file:
        key_file.write(key)


@pytest.fixture(scope="session")
def tls_key_pair(tmp_path_factory):
    tls_dir = tmp_path_factory.mktemp("tls")

    cert_file = str(tls_dir / "server.crt")
    key_file = str(tls_dir / "server.key")

    create_tls_keypair(cert_file, key_file)

    return cert_file, key_file
