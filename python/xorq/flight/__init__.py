import functools
import socket

import toolz
from pydantic import AnyUrl, UrlConstraints

import xorq as xo
from xorq.flight.backend import Backend
from xorq.flight.server import (
    BasicAuthServerMiddlewareFactory,
    FlightServerDelegate,
    NoOpAuthHandler,
)


DEFAULT_AUTH_MIDDLEWARE = {
    "basic": BasicAuthServerMiddlewareFactory(
        {
            "test": "password",
        }
    )
}


class FlightUrl(AnyUrl):
    default_scheme = "grpc"
    default_host = "localhost"
    default_port = 5005
    _constraints = UrlConstraints(
        allowed_schemes=(default_scheme,),
        default_host=default_host,
        default_port=default_port,
    )

    def to_location(self):
        return str(self)

    def port_in_use(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((self.host, self.port))
                return False
            except socket.error:
                return True

    @classmethod
    def from_defaults(cls, **kwargs):
        (scheme, *_) = cls._constraints.allowed_schemes
        _kwargs = toolz.merge(
            dict(
                scheme=scheme,
                host=cls._constraints.default_host,
                port=cls._constraints.default_port,
            ),
            kwargs,
        )
        return cls.build(**_kwargs)


class BasicAuth:
    def __init__(self, username, password):
        self.username = username
        self.password = password


def to_basic_auth_middleware(basic_auth: BasicAuth) -> dict:
    assert basic_auth is not None

    return {
        "basic": BasicAuthServerMiddlewareFactory(
            {
                basic_auth.username: basic_auth.password,
            }
        )
    }


class FlightServer:
    def __init__(
        self,
        flight_url=FlightUrl.from_defaults(),
        certificate_path=None,
        key_path=None,
        verify_client=False,
        root_certificates=None,
        auth: BasicAuth = None,
        connection=xo.connect,
    ):
        self.flight_url = flight_url
        self.certificate_path = certificate_path
        self.key_path = key_path
        self.root_certificates = root_certificates
        self.auth = auth

        if self.flight_url.port_in_use():
            raise ValueError(
                f"Port {self.flight_url.port} already in use (flight_url={self.flight_url})"
            )
        self.server = FlightServerDelegate(
            connection,
            flight_url.to_location(),
            verify_client=verify_client,
            **self.auth_kwargs,
        )

    @property
    def auth_kwargs(self):
        kwargs = {
            "root_certificates": self.root_certificates,
        }

        if self.key_path is not None and self.certificate_path is not None:
            with open(self.certificate_path, "rb") as cert_file:
                tls_cert_chain = cert_file.read()

            with open(self.key_path, "rb") as key_file:
                tls_private_key = key_file.read()

            kwargs["tls_certificates"] = [(tls_cert_chain, tls_private_key)]

        if self.auth is not None:
            kwargs["auth_handler"] = NoOpAuthHandler()
            kwargs["middleware"] = to_basic_auth_middleware(self.auth)

        return kwargs

    @property
    @functools.cache
    def con(self) -> Backend:
        kwargs = {
            "host": self.flight_url.host,
            "port": self.flight_url.port,
        }

        if self.auth is not None:
            kwargs["username"] = self.auth.username
            kwargs["password"] = self.auth.password

        if self.certificate_path is not None:
            kwargs["tls_roots"] = self.certificate_path

        instance = Backend()
        instance.do_connect(**kwargs)
        return instance

    @property
    def client(self):
        return self.con.con

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.server.__exit__(*args)


__all__ = ["FlightServer", "BasicAuth"]
