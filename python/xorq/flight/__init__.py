import functools
import socket
from typing import Optional
from urllib.parse import urlunparse

from attrs import field, frozen, validators

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


allowed_schemes = ("grpc",)
default_host = "localhost"
default_port = 5005


@frozen
class FlightUrl:
    scheme: str = field(default="grpc", validator=validators.in_(allowed_schemes))
    host: str = field(default=default_host)
    username: Optional[str] = field(default=None)
    password: Optional[str] = field(default=None)
    port: Optional[int] = field(default=default_port)
    path: Optional[str] = field(default="")
    query: Optional[str] = field(default="")
    fragment: Optional[str] = field(default="")

    def to_location(self):
        components = (
            self.scheme,
            f"{self.host}:{self.port}",
            self.path,
            "",
            self.query,
            self.fragment,
        )
        return str(urlunparse(components))

    def port_in_use(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((self.host, self.port))
                return False
            except socket.error:
                return True

    @classmethod
    def from_defaults(cls, **kwargs):
        _kwargs = {attr.name: attr.default for attr in cls.__attrs_attrs__} | kwargs
        return cls(**_kwargs)


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
