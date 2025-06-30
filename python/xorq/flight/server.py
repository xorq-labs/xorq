import base64
import logging
import secrets
import threading

import pyarrow as pa
import pyarrow.flight
from cloudpickle import loads

import xorq.flight.action as A
import xorq.flight.exchanger as E
from xorq.common.utils.func_utils import (
    maybe_log_excepts,
)
from xorq.common.utils.rbr_utils import (
    copy_rbr_batches,
)
from xorq.flight.metrics import (
    InstrumentedReader,
    InstrumentedWriter,
    instrument_reader,
    instrument_rpc,
)


logger = logging.getLogger(__name__)


class BasicAuthServerMiddlewareFactory(pa.flight.ServerMiddlewareFactory):
    """
    Middleware that implements username-password authentication.

    Parameters
    ----------
    creds: Dict[str, str]
        A dictionary of username-password values to accept.
    """

    def __init__(self, creds):
        self.creds = creds
        # Map generated bearer tokens to users
        self.tokens = {}

    def start_call(self, info, headers):
        """Validate credentials at the start of every call."""
        # Search for the authentication header (case-insensitive)
        auth_header = None
        for header in headers:
            if header.lower() == "authorization":
                auth_header = headers[header][0]
                break

        if not auth_header:
            raise pa.flight.FlightUnauthenticatedError("No credentials supplied")

        # The header has the structure "AuthType TokenValue", e.g.
        # "Basic <encoded username+password>" or "Bearer <random token>".
        auth_type, _, value = auth_header.partition(" ")

        if auth_type == "Basic":
            # Initial "login". The user provided a username/password
            # combination encoded in the same way as HTTP Basic Auth.
            decoded = base64.b64decode(value).decode("utf-8")
            username, _, password = decoded.partition(":")
            if not password or password != self.creds.get(username):
                raise pa.flight.FlightUnauthenticatedError(
                    "Unknown user or invalid password"
                )
            # Generate a secret, random bearer token for future calls.
            token = secrets.token_urlsafe(32)
            self.tokens[token] = username
            return BasicAuthServerMiddleware(token)
        elif auth_type == "Bearer":
            # An actual call. Validate the bearer token.
            username = self.tokens.get(value)
            if username is None:
                raise pa.flight.FlightUnauthenticatedError("Invalid token")
            return BasicAuthServerMiddleware(value)

        raise pa.flight.FlightUnauthenticatedError("No credentials supplied")


class BasicAuthServerMiddleware(pa.flight.ServerMiddleware):
    """Middleware that implements username-password authentication."""

    def __init__(self, token):
        self.token = token

    def sending_headers(self):
        """Return the authentication token to the client."""
        return {"authorization": f"Bearer {self.token}"}


class NoOpAuthHandler(pa.flight.ServerAuthHandler):
    """
    A handler that implements username-password authentication.

    This is required only so that the server will respond to the internal
    Handshake RPC call, which the client calls when authenticate_basic_token
    is called. Otherwise, it should be a no-op as the actual authentication is
    implemented in middleware.
    """

    def authenticate(self, outgoing, incoming):
        pass

    def is_valid(self, token):
        return ""


class FlightServerDelegate(pyarrow.flight.FlightServerBase):
    def __init__(
        self,
        con_callable,
        location=None,
        tls_certificates=None,
        verify_client=False,
        root_certificates=None,
        auth_handler=None,
        middleware=None,
    ):
        super(FlightServerDelegate, self).__init__(
            location=location,
            auth_handler=auth_handler,
            tls_certificates=tls_certificates,
            verify_client=verify_client,
            root_certificates=root_certificates,
            middleware=middleware,
        )
        self._conn = con_callable()
        self._location = location
        self.exchangers = dict(E.exchangers)
        self.actions = A.actions
        self.lock = threading.Lock()

    def _make_flight_info(self, query):
        """
        Create Flight info for a given SQL query

        Args:
            query: SQL query string
        """
        # Execute query to get schema and metadata
        kwargs = loads(query)
        expr = kwargs.pop("expr")
        schema = expr.as_table().schema().to_pyarrow()
        descriptor = pyarrow.flight.FlightDescriptor.for_command(query)
        endpoints = [pyarrow.flight.FlightEndpoint(query, [self._location])]

        return pyarrow.flight.FlightInfo(schema, descriptor, endpoints, -1, -1)

    def get_flight_info(self, context, descriptor):
        """
        Get info about a specific query
        """
        query = descriptor.command
        return self._make_flight_info(query)

    @instrument_rpc("do_get")
    @maybe_log_excepts
    def do_get(self, context, rec, ticket):
        kwargs = loads(ticket.ticket)
        expr = kwargs.pop("expr")
        with self.lock:
            rbr = self._conn.to_pyarrow_batches(expr)
        rbr = instrument_reader(rbr, rec, direction="out")
        return pyarrow.flight.RecordBatchStream(rbr)

    @instrument_rpc("do_put")
    @maybe_log_excepts
    def do_put(self, context, rec, descriptor, reader, writer):
        reader = instrument_reader(reader, rec, direction="in")

        cmd = loads(descriptor.command)
        if isinstance(cmd, bytes):
            table = cmd.decode("utf-8")
            kwargs = {}
        else:
            table = cmd["table_name"]
            kwargs = cmd.get("kwargs", {})

        data = copy_rbr_batches(reader).read_all()
        with self.lock:
            if table in self._conn.tables:
                self._conn.insert(table, data, **kwargs)
            else:
                self._conn.create_table(table, data, **kwargs)

    @maybe_log_excepts
    def do_action(self, context, action):
        with self.lock:
            cls = self.actions.get(action.type)
            if cls:
                logger.info(f"doing action: {action.type}")
                yield from cls.do_action(self, context, action)
            else:
                raise KeyError("Unknown action {!r}".format(action.type))

    @instrument_rpc("do_exchange")
    @maybe_log_excepts
    def do_exchange(self, context, rec, descriptor, reader, writer):
        with self.lock:
            cmd = descriptor.command.decode()
            handler = self.exchangers.get(cmd)
            if handler is None:
                raise pa.ArrowInvalid(f"Unknown exchange: {cmd}")

            reader = InstrumentedReader(reader, rec, direction="in")
            iw = InstrumentedWriter(writer, rec, direction="out")

            return handler.exchange_f(context, reader, iw)
