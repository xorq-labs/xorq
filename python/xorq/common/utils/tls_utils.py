"""Utilities for generating self-signed TLS certificates using pyOpenSSL."""

import datetime
import logging
import os
import socket
from pathlib import Path

import OpenSSL
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

CERTIFICATE_VERSION: int = 3 - 1


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


def create_ca_keypair(
    ca_cert_file: str,
    ca_key_file: str,
    overwrite: bool = True,
    ca_common_name: str = "flight_server_ca",
):
    ca_cert_file_path = Path(ca_cert_file)
    ca_key_file_path = Path(ca_key_file)

    if ca_cert_file_path.exists() or ca_key_file_path.exists():
        if not overwrite:
            raise RuntimeError(
                f"The CA Cert file(s): '{ca_cert_file_path.as_posix()}' or '{ca_key_file_path.as_posix()}' - exist - and overwrite is False, aborting."
            )
        else:
            ca_cert_file_path.unlink(missing_ok=True)
            ca_key_file_path.unlink(missing_ok=True)

    # Generate a new key pair for the CA
    key = OpenSSL.crypto.PKey()
    key.generate_key(OpenSSL.crypto.TYPE_RSA, 2048)

    # Generate a self-signed CA certificate
    ca_cert = OpenSSL.crypto.X509()
    ca_cert.get_subject().CN = ca_common_name
    ca_cert.set_version(CERTIFICATE_VERSION)
    ca_cert.set_serial_number(1000)
    ca_cert.gmtime_adj_notBefore(0)
    ca_cert.gmtime_adj_notAfter(365 * 24 * 60 * 60)  # 1-year validity
    ca_cert.set_issuer(ca_cert.get_subject())
    ca_cert.set_pubkey(key)
    ca_cert.add_extensions(
        [
            OpenSSL.crypto.X509Extension(b"basicConstraints", True, b"CA:TRUE"),
            OpenSSL.crypto.X509Extension(
                b"subjectKeyIdentifier", False, b"hash", subject=ca_cert
            ),
        ]
    )
    ca_cert.sign(key, "sha256")

    ca_cert_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(Path(ca_cert_file_path), "wb") as f:
        f.write(OpenSSL.crypto.dump_certificate(OpenSSL.crypto.FILETYPE_PEM, ca_cert))

    ca_key_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(Path(ca_key_file_path), "wb") as f:
        f.write(OpenSSL.crypto.dump_privatekey(OpenSSL.crypto.FILETYPE_PEM, key))


def create_client_keypair(
    ca_cert_file: str,
    ca_key_file: str,
    client_cert_file: str,
    client_key_file: str,
    overwrite: bool = False,
    client_common_name: str = "flight_client",
):
    ca_cert_file_path = Path(ca_cert_file)
    ca_key_file_path = Path(ca_key_file)

    if not ca_cert_file_path.exists():
        raise RuntimeError(
            f"The CA Cert file: '{ca_cert_file_path.as_posix()}' does not exist, aborting"
        )

    if not ca_key_file_path.exists():
        raise RuntimeError(
            f"The CA Key file: '{ca_key_file_path.as_posix()}' does not exist, aborting"
        )

    client_cert_file_path = Path(client_cert_file)
    client_key_file_path = Path(client_key_file)

    if client_cert_file_path.exists() or client_key_file_path.exists():
        if not overwrite:
            raise RuntimeError(
                f"The Client Cert file(s): '{client_cert_file_path.as_posix()}' or '{client_key_file_path.as_posix()}' - exist - and overwrite is False, aborting."
            )
        else:
            client_cert_file_path.unlink(missing_ok=True)
            client_key_file_path.unlink(missing_ok=True)

    # Generate a new key pair for the client
    key = OpenSSL.crypto.PKey()
    key.generate_key(OpenSSL.crypto.TYPE_RSA, 2048)

    # Generate a certificate signing request (CSR) for the client
    req = OpenSSL.crypto.X509Req()
    req.get_subject().CN = client_common_name
    req.set_pubkey(key)
    req.sign(key, "sha256")

    # Load the CA certificate and key from disk
    with open(ca_cert_file_path, "rb") as f:
        ca_cert = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_PEM, f.read())
    with open(ca_key_file_path, "rb") as f:
        ca_key = OpenSSL.crypto.load_privatekey(OpenSSL.crypto.FILETYPE_PEM, f.read())

    # Create a new certificate for the client, signed by the CA
    client_cert = OpenSSL.crypto.X509()
    client_cert.set_version(CERTIFICATE_VERSION)
    client_cert.set_subject(req.get_subject())
    client_cert.set_serial_number(2000)
    client_cert.gmtime_adj_notBefore(0)
    client_cert.gmtime_adj_notAfter(365 * 24 * 60 * 60)  # 1 year validity
    client_cert.set_issuer(ca_cert.get_subject())
    client_cert.set_pubkey(req.get_pubkey())
    client_cert.add_extensions(
        [
            OpenSSL.crypto.X509Extension(b"basicConstraints", True, b"CA:FALSE"),
            OpenSSL.crypto.X509Extension(
                b"subjectKeyIdentifier", False, b"hash", subject=client_cert
            ),
        ]
    )
    client_cert.sign(ca_key, "sha256")

    # Write the client certificate and key to disk
    with open(client_cert_file_path, "wb") as f:
        f.write(
            OpenSSL.crypto.dump_certificate(OpenSSL.crypto.FILETYPE_PEM, client_cert)
        )
    with open(client_key_file_path, "wb") as f:
        f.write(OpenSSL.crypto.dump_privatekey(OpenSSL.crypto.FILETYPE_PEM, key))
