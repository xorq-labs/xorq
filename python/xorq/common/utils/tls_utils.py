"""Utilities for generating self-signed TLS certificates using pyOpenSSL."""

import datetime
import logging
import os
import socket
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)
from cryptography.x509.oid import NameOID


logger = logging.getLogger()


def _gen_cryptography(common_name: str) -> tuple[bytes, bytes]:
    # Generate RSA private key (2048-bit)
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )

    # Configure subject and issuer (self-signed)
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ]
    )

    # Certificate metadata
    serial_number = int.from_bytes(os.urandom(8), "big")
    not_before = datetime.datetime.now(datetime.timezone.utc)
    not_after = not_before + datetime.timedelta(days=5 * 365)  # 5 years

    # Subject Alternative Names
    san_list = [
        x509.DNSName(common_name),
        x509.DNSName(f"*.{common_name}"),
        x509.DNSName("localhost"),
        x509.DNSName("*.localhost"),
    ]
    hostname = socket.gethostname()
    if hostname != common_name:
        san_list.extend([x509.DNSName(hostname), x509.DNSName(f"*.{hostname}")])

    # Build certificate with extensions
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(serial_number)
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(private_key, hashes.SHA256(), default_backend())
    )

    # PEM serialization
    return (
        cert.public_bytes(Encoding.PEM),
        private_key.private_bytes(
            Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()
        ),
    )


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

    cert, key = _gen_cryptography(common_name)

    cert_file_path.parent.mkdir(parents=True, exist_ok=True)
    with cert_file_path.open(mode="wb") as cert_file:
        cert_file.write(cert)

    with key_file_path.open(mode="wb") as key_file:
        key_file.write(key)
