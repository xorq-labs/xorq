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
    optional,
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


@frozen
class TLSKwargs:
    verify_client = field(validator=instance_of(bool))
    ca_tlscert = field(validator=instance_of(TLSCert))
    server_tlscert = field(validator=instance_of(TLSCert))
    client_tlscert = field(
        validator=optional(instance_of(TLSCert)),
        default=None,
    )

    def __attrs_post_init__(self):
        self.ca_tlscert.verify(self.server_tlscert)
        if self.verify_client:
            assert self.client_tlscert is not None
            self.ca_tlscert.verify(self.client_tlscert)

    @property
    def server_kwargs(self):
        return {
            "tls_certificates": self.server_tlscert.tls_certificates,
            "root_certificates": self.ca_tlscert.cert_bytes,
        }

    @property
    def client_kwargs(self):
        client_kwargs = {
            "tls_root_certs": self.ca_tlscert.cert_bytes,
        }

        if self.verify_client:
            client_kwargs |= {
                "cert_chain": self.client_tlscert.cert_bytes,
                "private_key": self.client_tlscert.private_key_bytes,
            }

        return client_kwargs

    @classmethod
    def from_kwargs(cls, ca_kwargs, server_kwargs, client_kwargs, verify_client=True):
        if server_kwargs.get(
            "common_name"
        ) != "localhost" or "localhost" not in server_kwargs.get("sans", ()):
            # FIXME: issue a warning
            pass
        ca_tlscert = TLSCert.from_common_name(**ca_kwargs)
        server_tlscert = TLSCert.from_common_name(sign_with=ca_tlscert, **server_kwargs)
        client_tlscert = TLSCert.from_common_name(sign_with=ca_tlscert, **client_kwargs)

        return cls(verify_client, ca_tlscert, server_tlscert, client_tlscert)

    @classmethod
    def from_common_name(cls, verify_client=True, common_name=socket.gethostname()):
        ca_kwargs = {
            "common_name": "root_cert",
        }
        server_kwargs = {
            "common_name": common_name,
            "sans": ("localhost",),
        }
        client_kwargs = {
            "common_name": "client",
        }

        return cls.from_kwargs(
            ca_kwargs, server_kwargs, client_kwargs, verify_client=verify_client
        )

    @classmethod
    def from_disk(
        cls,
        ca_cert_path,
        ca_private_key_path,
        server_cert_path,
        server_private_key_path,
        client_cert_path=None,
        client_private_key_path=None,
        verify_client=True,
        ca_password=None,
        server_password=None,
        client_password=None,
    ):
        """
        Create TLSKwargs by loading certificates and private keys from disk.

        Args:
            ca_cert_path: Path to CA certificate file
            ca_private_key_path: Path to CA private key file
            server_cert_path: Path to server certificate file
            server_private_key_path: Path to server private key file
            client_cert_path: Path to client certificate file (optional)
            client_private_key_path: Path to client private key file (optional)
            verify_client: Whether client verification is required
            ca_password: Password for CA private key (optional)
            server_password: Password for server private key (optional)
            client_password: Password for client private key (optional)

        Returns:
            TLSKwargs instance with certificates loaded from disk
        """
        # Load CA certificate
        ca_tlscert = TLSCert.from_disk(ca_cert_path, ca_private_key_path, ca_password)

        # Load server certificate
        server_tlscert = TLSCert.from_disk(
            server_cert_path, server_private_key_path, server_password
        )

        # Load client certificate if paths provided or if client verification is required
        client_tlscert = None
        if client_cert_path is not None and client_private_key_path is not None:
            client_tlscert = TLSCert.from_disk(
                client_cert_path, client_private_key_path, client_password
            )
        elif verify_client:
            raise ValueError(
                "Client certificate paths must be provided when verify_client=True"
            )

        return cls(
            verify_client=verify_client,
            ca_tlscert=ca_tlscert,
            server_tlscert=server_tlscert,
            client_tlscert=client_tlscert,
        )

    def to_disk(
        self,
        ca_cert_path,
        ca_private_key_path,
        server_cert_path,
        server_private_key_path,
        client_cert_path=None,
        client_private_key_path=None,
        ca_password=None,
        server_password=None,
        client_password=None,
    ):
        """
        Save all TLS certificates and private keys to disk.

        Args:
            ca_cert_path: Path where CA certificate will be saved
            ca_private_key_path: Path where CA private key will be saved
            server_cert_path: Path where server certificate will be saved
            server_private_key_path: Path where server private key will be saved
            client_cert_path: Path where client certificate will be saved (optional)
            client_private_key_path: Path where client private key will be saved (optional)
            ca_password: Password to encrypt CA private key (optional)
            server_password: Password to encrypt server private key (optional)
            client_password: Password to encrypt client private key (optional)

        Raises:
            ValueError: If client paths are not provided when client_tlscert exists
        """
        # Save CA certificate and key
        self.ca_tlscert.to_disk(ca_cert_path, ca_private_key_path, ca_password)

        # Save server certificate and key
        self.server_tlscert.to_disk(
            server_cert_path, server_private_key_path, server_password
        )

        # Save client certificate and key if it exists
        if self.client_tlscert is not None:
            if client_cert_path is None or client_private_key_path is None:
                raise ValueError(
                    "Client certificate paths must be provided when client_tlscert exists"
                )
            self.client_tlscert.to_disk(
                client_cert_path, client_private_key_path, client_password
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
