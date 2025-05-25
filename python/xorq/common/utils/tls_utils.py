"""Utilities for generating self-signed TLS certificates using pyca/cryptography."""

import datetime
import operator
import socket

from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
)
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import (
    padding,
    rsa,
)
from cryptography.hazmat.primitives.serialization import (
    BestAvailableEncryption,
    Encoding,
    NoEncryption,
    PrivateFormat,
    load_pem_private_key,
)
from cryptography.x509.oid import NameOID


try:
    utc = datetime.UTC
except AttributeError:
    utc = datetime.timezone.utc


pem_encode_cert = operator.methodcaller("public_bytes", Encoding.PEM)
pem_decode_cert = x509.load_pem_x509_certificate


def pem_encode_private_key(private_key, password=None):
    encryption = (
        NoEncryption() if password is None else BestAvailableEncryption(password)
    )
    return private_key.private_bytes(
        Encoding.PEM, PrivateFormat.TraditionalOpenSSL, encryption
    )


def pem_decode_private_key(byts, password=None):
    return load_pem_private_key(byts, password)


def make_tls_cert_and_private_key(
    common_name, *sans, duration=datetime.timedelta(5 * 365)
) -> tuple[x509.Certificate, rsa.RSAPrivateKey]:
    # Generate RSA private key (2048-bit)
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ]
    )

    # Build certificate with extensions
    now = datetime.datetime.now(utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + duration)
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(san) for san in sans]),
            critical=False,
        )
        .add_extension(
            # can sign leaf but not intermediary
            x509.BasicConstraints(ca=True, path_length=0),
            critical=True,
        )
        .sign(
            private_key,
            hashes.SHA256(),
            default_backend(),
        )
    )

    return (cert, private_key)


@frozen
class TLSCert:
    cert = field(validator=instance_of(x509.Certificate))
    private_key = field(
        validator=instance_of(rsa.RSAPrivateKey),
        eq=operator.methodcaller("private_numbers"),
    )

    def __equals__(self, other):
        return (
            self.cert == other.cert
            and self.private_key.private_numbers()
            == other.private_key.private_numbers()
        )

    @property
    def cert_bytes(self):
        return pem_encode_cert(self.cert)

    @property
    def private_key_bytes(self):
        return pem_encode_private_key(self.private_key)

    @property
    def tls_certificates(self):
        return [(self.cert_bytes, self.private_key_bytes)]

    def sign_csr(self, csr):
        return sign_csr(self.cert, self.private_key, csr)

    def create_csr(self):
        return create_csr(self.cert, self.private_key)

    def verify(self, other):
        verify(self.cert, other.cert)

    def signed_with(self, ca_tlscert):
        csr = self.create_csr()
        cert = ca_tlscert.sign_csr(csr)
        return type(self)(cert, self.private_key)

    def to_disk(self, cert_path, private_key_path, password=None):
        cert_path.write_bytes(pem_encode_cert(self.cert))
        private_key_path.write_bytes(pem_encode_private_key(self.private_key, password))

    @classmethod
    def from_disk(cls, cert_path, private_key_path, password=None):
        return cls(
            pem_decode_cert(cert_path.read_bytes()),
            pem_decode_private_key(private_key_path.read_bytes(), password),
        )

    @classmethod
    def from_common_name(
        cls,
        common_name=socket.gethostname(),
        *,
        sans=(),
        sign_with=None,
    ):
        (cert, private_key) = make_tls_cert_and_private_key(common_name, *sans)
        self = cls(cert, private_key)
        if sign_with:
            self = self.signed_with(sign_with)
        return self

    @classmethod
    def create_tls_kwargs(cls, ca_kwargs, server_kwargs):
        if server_kwargs.get(
            "common_name"
        ) != "localhost" or "localhost" not in server_kwargs.get("sans", ()):
            # FIXME: issue a warning
            pass
        ca_tlscert = cls.from_common_name(**ca_kwargs)
        server_tlscert = cls.from_common_name(sign_with=ca_tlscert, **server_kwargs)
        server_kwargs = {
            "tls_certificates": server_tlscert.tls_certificates,
            "root_certificates": ca_tlscert.cert_bytes,
        }
        client_kwargs = {
            "tls_root_certs": ca_tlscert.cert_bytes,
        }
        return (
            # must keep a reference to the tlscert objects else they disappear
            (ca_tlscert, server_tlscert),
            server_kwargs,
            client_kwargs,
        )

    @classmethod
    def create_mtls_kwargs(cls, ca_kwargs, server_kwargs, client_kwargs):
        ((ca_tlscert, server_tlscert), *_) = cls.create_tls_kwargs(
            ca_kwargs, server_kwargs
        )
        client_tlscert = cls.from_common_name(sign_with=ca_tlscert, **client_kwargs)
        server_kwargs = {
            "root_certificates": ca_tlscert.cert_bytes,
            "tls_certificates": server_tlscert.tls_certificates,
        }
        client_kwargs = {
            "tls_root_certs": ca_tlscert.cert_bytes,
            "cert_chain": client_tlscert.cert_bytes,
            "private_key": client_tlscert.private_key_bytes,
        }
        return (
            # must keep a reference to the tlscert objects else they disappear
            (ca_tlscert, server_tlscert, client_tlscert),
            server_kwargs,
            client_kwargs,
        )


def get_san(cert):
    cls = x509.SubjectAlternativeName
    san = cls([])
    try:
        ext = cert.extensions.get_extension_for_class(cls)
        san = ext.value
    except Exception:
        pass
    return san


def get_common_name(cert):
    (attr, *rest) = cert.subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)
    assert not rest
    return attr.value


def create_csr(cert, private_key):
    csr = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(cert.subject)
        .add_extension(get_san(cert), critical=False)
        .sign(private_key, hashes.SHA256())
    )
    return csr


def sign_csr(
    signing_cert, signing_private_key, csr, duration=datetime.timedelta(days=5 * 365)
):
    now = datetime.datetime.now(utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(csr.subject)
        .issuer_name(signing_cert.subject)
        .public_key(csr.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + duration)
        .add_extension(get_san(csr), critical=False)
        .sign(signing_private_key, hashes.SHA256())
    )
    return cert


def verify(ca_cert, other_cert):
    public_key = ca_cert.public_key()
    public_key.verify(
        other_cert.signature,
        other_cert.tbs_certificate_bytes,
        padding.PKCS1v15(),
        other_cert.signature_hash_algorithm,
    )
