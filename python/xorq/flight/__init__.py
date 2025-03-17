import functools
import random
import socket
from typing import Optional
from urllib.parse import urlunparse

from attrs import (
    define,
    field,
)
from attrs.validators import (
    in_,
    instance_of,
    optional,
)

import xorq as xo
from xorq.flight.action import AddExchangeAction
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


@define
class FlightUrl:
    scheme: str = field(default="grpc", validator=in_(allowed_schemes))
    host: str = field(default=default_host, validator=instance_of(str))
    username: Optional[str] = field(default=None, validator=optional(instance_of(str)))
    password: Optional[str] = field(default=None, validator=optional(instance_of(str)))
    port: Optional[int] = field(default=None, validator=optional(instance_of(int)))
    path: Optional[str] = field(default="", validator=instance_of(str))
    query: Optional[str] = field(default="", validator=instance_of(str))
    fragment: Optional[str] = field(default="", validator=instance_of(str))
    _socket = field(default=None, init=False)

    def __attrs_post_init__(self):
        if self.port is None:
            self.find_and_bind_socket()
        else:
            self.bind_socket()

    def find_and_bind_socket(self):
        while True:
            port = random.randint(5000, 8000)
            try:
                self._socket = self._bind_socket(self.host, port)
                self.port = port
                break
            except socket.error:
                pass

    def bind_socket(self):
        self._socket = self._bind_socket(self.host, self.port)

    def unbind_socket(self):
        self._socket = None

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

    @staticmethod
    def _bind_socket(host, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host, port))
        return s

    @staticmethod
    def port_in_use(port, host="localhost"):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return False
            except socket.error:
                return True


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
        flight_url=None,
        certificate_path=None,
        key_path=None,
        verify_client=False,
        root_certificates=None,
        auth: BasicAuth = None,
        connection=xo.connect,
        exchangers=(),
    ):
        self.flight_url = flight_url or FlightUrl()
        self.certificate_path = certificate_path
        self.key_path = key_path
        self.root_certificates = root_certificates
        self.auth = auth
        self.connection = connection
        self.verify_client = verify_client
        self.server = None
        self.exchangers = exchangers

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
        self.serve()
        return self

    def serve(self):
        self.flight_url.unbind_socket()
        self.server = FlightServerDelegate(
            self.connection,
            self.flight_url.to_location(),
            verify_client=self.verify_client,
            **self.auth_kwargs,
        )
        for udxf in self.exchangers:
            self.client.do_action(
                AddExchangeAction.name, udxf, options=self.client._options
            )

    def close(self, *args):
        args = args or (None, None, None)
        self.server.__exit__(*args)
        self.server = None
        self.flight_url.bind_socket()

    def __exit__(self, *args):
        self.close(*args)


__all__ = ["FlightServer", "BasicAuth"]
