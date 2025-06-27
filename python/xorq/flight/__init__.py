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

allowed_schemes = ("grpc", "grpc+tls")
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

    @property
    def client_kwargs(self):
        return {
            k: getattr(self, k)
            for k in (
                "host",
                "port",
                "username",
                "password",
            )
        }

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
        tls_certificates=(),  # list of (key, value) pairs
        verify_client=False,
        root_certificates=None,
        auth: BasicAuth = None,
        make_connection=xo.connect,
        exchangers=(),
    ):
        required_scheme = (
            "grpc+tls"
            if tls_certificates or verify_client or root_certificates
            else "grpc"
        )
        self.flight_url = flight_url or FlightUrl(scheme=required_scheme)
        assert self.flight_url.scheme == required_scheme
        self.tls_certificates = tls_certificates
        self.root_certificates = root_certificates
        self.auth = auth
        self.make_connection = make_connection
        self.verify_client = verify_client
        self.server = None
        self.exchangers = exchangers

    @classmethod
    def from_udxf(cls, expr, host=None, port=None, make_connection=None, **kwargs):
        from xorq.common.utils.graph_utils import walk_nodes
        from xorq.expr.relations import FlightUDXF

        # Find all FlightUDXF nodes in the expression
        exchangers = tuple(set(node.udxf for node in walk_nodes((FlightUDXF,), expr)))

        flight_url_kwargs = {
            key: value for key, value in (("host", host), ("port", port)) if value
        }
        flight_url = FlightUrl(**flight_url_kwargs)

        server_kwargs = {
            key: value
            for key, value in (
                ("exchangers", exchangers),
                ("make_connection", make_connection),
            )
            if value
        }

        return cls(
            flight_url,
            **kwargs | server_kwargs,
        )

    @property
    def auth_kwargs(self):
        kwargs = {
            "root_certificates": self.root_certificates,
            "tls_certificates": list(self.tls_certificates),
        }
        if self.auth:
            kwargs |= {
                "auth_handler": NoOpAuthHandler(),
                "middleware": to_basic_auth_middleware(self.auth),
            }
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

        if self.root_certificates:
            # this mapping is for Backend's FlightClient
            kwargs["tls_root_certs"] = self.root_certificates

        if self.tls_certificates:
            ((cert_chain, private_key), *rest) = self.tls_certificates
            if rest:
                # FIXME: what to do with multiple certs?
                raise ValueError
            kwargs["cert_chain"] = cert_chain
            kwargs["private_key"] = private_key

        instance = Backend()
        instance.do_connect(**kwargs)
        return instance

    @property
    def client(self):
        return self.con.con

    def __enter__(self):
        self.serve()
        return self

    def serve(self, block=False):
        self.flight_url.unbind_socket()
        self.server = FlightServerDelegate(
            self.make_connection,
            self.flight_url.to_location(),
            verify_client=self.verify_client,
            **self.auth_kwargs,
        )
        for udxf in self.exchangers:
            self.client.do_action(
                AddExchangeAction.name, udxf, options=self.client._options
            )
        if block:
            # https://arrow.apache.org/docs/python/generated/pyarrow.flight.FlightServerBase.html#pyarrow.flight.FlightServerBase.serve
            self.server.serve()

    def close(self, *args):
        args = args or (None, None, None)
        self.server.__exit__(*args)
        self.server = None
        self.flight_url.bind_socket()

    def __exit__(self, *args):
        self.close(*args)


def connect(url, tls_kwargs=None):
    new_backend = Backend()
    new_backend.do_connect(
        host=url.host,
        port=url.port,
        username=url.username,
        password=url.password,
        **(tls_kwargs.client_kwargs if tls_kwargs else {}),
    )

    return new_backend


__all__ = ["FlightServer", "FlightUrl", "BasicAuth", "connect"]
