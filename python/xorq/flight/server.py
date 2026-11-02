import base64
import functools
import logging
import secrets
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, TypedDict

import pyarrow as pa
import pyarrow.flight
from attrs import Factory, field, frozen
from cloudpickle import loads

import xorq.flight.action as A
import xorq.flight.exchanger as E
from xorq.common.utils.func_utils import (
    maybe_log_excepts,
)
from xorq.common.utils.rbr_utils import (
    copy_rbr_batches,
)
from xorq.flight.enums import UploadMode
from xorq.flight.metrics import (
    InstrumentedReader,
    InstrumentedWriter,
    instrument_reader,
    instrument_rpc,
)
from xorq.vendor.ibis.backends import BaseBackend


if TYPE_CHECKING:
    from xorq.flight.metrics import _Recorder

logger = logging.getLogger(__name__)

# gRPC caps the status trailers at 8 KB, and the message travels in them twice
# (percent-encoded `grpc-message` plus the binary status detail): 4 000 ASCII
# characters already overflow, 1 000 two-byte ones do, and the traceback that
# pyarrow appends to a plain Python exception overflows on its own. Measured
# in UTF-8 bytes; the worst case lands at 4 KB on the wire.
STATUS_MESSAGE_BUDGET = 1024


def bounded_message(exc: BaseException) -> str:
    """``Type: message`` with the middle elided so the status fits the budget."""
    text = f"{type(exc).__name__}: {exc}".encode()
    if len(text) <= STATUS_MESSAGE_BUDGET:
        return text.decode()
    (head, tail) = (
        text[: STATUS_MESSAGE_BUDGET * 3 // 4].decode(errors="ignore"),
        text[-STATUS_MESSAGE_BUDGET // 4 :].decode(errors="ignore"),
    )
    return f"{head} [... {len(text) - STATUS_MESSAGE_BUDGET} bytes elided ...] {tail}"


def bounded_status(f):
    """Re-raise any failure as a ``FlightServerError`` that fits the status.

    A plain Python exception reaches the client only if its message plus
    traceback fit gRPC's metadata cap; past that, the client sees ``received
    initial metadata size exceeds limit`` and nothing else. A ``FlightError``
    travels without the traceback, which stays in the server log.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except BaseException as exc:
            raise pa.flight.FlightServerError(bounded_message(exc)) from exc

    return wrapper


class _PutCommandRequired(TypedDict):
    table_name: str


class PutCommand(_PutCommandRequired, total=False):
    """The pickled ``do_put`` command ``FlightClient`` sends.

    ``table_name`` is required. ``kwargs`` (default ``{}``) is forwarded to the
    backend write, and ``mode`` (default ``None``) is a ``UploadMode`` string
    value. A command from a client older than ``mode`` has no ``mode`` key.
    """

    kwargs: dict[str, Any]
    mode: str | None


def parse_put_command(
    cmd: bytes | PutCommand,
) -> tuple[str, dict[str, Any], str | None]:
    """The ``(table, kwargs, mode)`` a ``do_put`` command asks for.

    A bare bytes command is a table name with no kwargs and no mode.

    Parameters
    ----------
    cmd
        The unpickled ``descriptor.command``: UTF-8 bytes naming the table, or
        a ``PutCommand`` mapping.

    Returns
    -------
    tuple[str, dict[str, Any], str | None]
        The table name, the backend kwargs and the mode, which is not
        validated here (``put_table`` does that).

    Raises
    ------
    UnicodeDecodeError
        If a bytes command is not valid UTF-8 (implicit, ``cmd.decode``).
    KeyError
        If a mapping command has no ``table_name`` (implicit).
    TypeError
        If ``cmd`` is neither bytes nor a mapping, e.g. a ``str`` (implicit,
        from indexing it with ``"table_name"``).

    Contract
    --------
    Silences
        - A missing ``kwargs`` or ``mode`` key yields ``{}`` / ``None``, so a
          command from an older client keeps the append-or-create behaviour.
          Unknown keys are ignored without error.
    """
    match cmd:
        case bytes():
            return cmd.decode("utf-8"), {}, None
        case _:
            return cmd["table_name"], cmd.get("kwargs", {}), cmd.get("mode")


def put_table(
    con: BaseBackend,
    table: str,
    data: pa.Table,
    mode: UploadMode | str | None = None,
    **kwargs: Any,
) -> None:
    """Write an uploaded table to ``con`` as the upload's write mode asks.

    Parameters
    ----------
    con
        The server's backend connection.
    table
        Name of the target table.
    data
        The uploaded rows, a ``pa.Table``.
    mode
        A ``UploadMode`` or its string value, or ``None``. ``None`` appends
        when ``table`` exists and creates it otherwise; ``create`` creates,
        ``append`` inserts, ``replace`` creates or overwrites.
    **kwargs
        Passed to the backend's ``create_table`` or ``insert``.

    Raises
    ------
    ValueError
        If ``mode`` is not a ``UploadMode`` value (``'bogus' is not a valid
        UploadMode``), or ``mode`` is given together with an ``overwrite``
        kwarg. Both are explicit checks, before ``con`` is touched.
    Exception
        Whatever the backend raises, not caught. ``create`` on an existing
        table fails in ``create_table`` (duckdb ``CatalogException``, xorq
        ``ValueError``, pandas ``XorqError``); ``append`` on an absent table
        fails in ``insert`` (duckdb ``XorqError``, xorq ``ValueError``).
    AttributeError
        If the backend has no ``insert`` (pandas) and the write appends:
        ``mode="append"``, or ``mode=None`` on an existing table (implicit).
    TypeError
        If ``kwargs`` holds a keyword the chosen backend method does not
        accept (implicit).

    Contract
    --------
    Preconditions
        - The caller holds the server lock: ``mode=None`` checks
          ``table in con.tables`` and then writes, so a concurrent writer can
          change the answer in between. ``do_put`` enforces this; the function
          does not.
        - ``data`` is a fully materialised ``pa.Table``; the function does no
          streaming (``do_put`` explains why).

    Silences
        - With ``mode=None``, an existing name that is not insertable (a view)
          is routed to ``insert`` and fails there with the backend's error,
          not a mode error.
        - With ``mode=None``, an ``overwrite`` kwarg is passed through to
          ``insert`` / ``create_table`` unchanged; only an explicit mode
          rejects it.
    """
    mode = None if mode is None else UploadMode(mode)
    if mode is not None and "overwrite" in kwargs:
        raise ValueError("pass mode='replace' instead of overwrite= with a mode")
    match mode:
        case None:
            write = con.insert if table in con.tables else con.create_table
            write(table, data, **kwargs)
        case UploadMode.CREATE:
            con.create_table(table, data, **kwargs)
        case UploadMode.APPEND:
            con.insert(table, data, **kwargs)
        case UploadMode.REPLACE:
            con.create_table(table, data, overwrite=True, **kwargs)


@frozen
class CallScopedReader:
    """An exchange's input that stops reading once the RPC has returned.

    The gRPC stream under ``reader`` is freed when the handler returns, and
    reading it afterwards is a use-after-free that segfaults. A backend can
    still be pulling the input then: duckdb's scan thread keeps reading after
    a ``limit`` is satisfied and the handler has written its result. Once
    ``close`` has run, iteration ends instead of reading; ``close`` waits for
    a read already in progress.

    Parameters
    ----------
    reader
        The exchange's input, an ``InstrumentedReader`` over the gRPC stream.

    Raises
    ------
    Exception
        Whatever reading ``reader`` raises before ``close``, e.g. a client
        that disconnects mid-stream (external, not caught).

    Contract
    --------
    Preconditions
        - Whoever owns the RPC calls ``close`` before the handler returns
          (``do_exchange`` does, in a ``finally``); without it the guard does
          nothing.

    Silences
        - A read after ``close`` ends iteration as if the input were
          exhausted: the reader sees fewer chunks and no error. Only a
          consumer whose result is already discarded (a backend's prefetch
          after the handler returned) should hit this.
        - ``close`` blocks until a read in progress finishes; if the client
          stalls without closing its side, ``close`` waits with it.
    """

    reader: InstrumentedReader = field()
    lock: threading.Lock = field(factory=threading.Lock)
    closed: threading.Event = field(default=Factory(threading.Event))

    @property
    def schema(self) -> pa.Schema:
        return self.reader.schema

    def __iter__(self) -> Iterator[pa.flight.FlightStreamChunk]:
        chunks = iter(self.reader)
        while True:
            with self.lock:
                chunk = None if self.closed.is_set() else next(chunks, None)
            if chunk is None:
                return
            yield chunk

    def close(self) -> None:
        # set first, so the reading thread cannot start another read while
        # this waits; then take the lock to wait out a read in progress
        self.closed.set()
        with self.lock:
            pass


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
        self.shutdown_requested = threading.Event()

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
    def do_put(
        self,
        context: pa.flight.ServerCallContext,
        rec: "_Recorder",
        descriptor: pa.flight.FlightDescriptor,
        reader: pa.flight.MetadataRecordBatchReader,
        writer: pa.flight.FlightMetadataWriter,
    ) -> None:
        """Store an upload as the command's table, with ``put_table``.

        The upload stream is only readable while this call runs: reading it
        after the call returns is a use-after-free in Arrow Flight. So the
        whole upload is read here before it is written. To process batches as
        they arrive instead of storing them, use ``do_exchange`` with an
        ``UnboundExprExchanger``. The stream is read outside ``self.lock``, so
        other RPCs are not blocked while an upload arrives; the write, with
        ``put_table``'s existence check, runs under it.

        Parameters
        ----------
        context
            The Flight server call context (unused).
        rec
            Metrics recorder, injected by ``instrument_rpc``.
        descriptor
            Carries the pickled command: UTF-8 bytes naming the table, or a
            ``PutCommand``.
        reader
            The client's upload stream.
        writer
            Metadata writer back to the client (unused; nothing is sent).

        Raises
        ------
        Exception
            Anything ``parse_put_command`` or ``put_table`` raises (bad command,
            unknown mode, ``mode`` with ``overwrite=``, backend write failure),
            not caught. The client receives it as ``FlightServerError``, or
            ``ArrowInvalid`` for a ``ValueError``, at ``writer.close()``.
        pyarrow.ArrowException
            If the upload stream fails while it is read, e.g. the client goes
            away (external, from ``read_all``). Nothing is written: the write
            happens only after the whole stream has been read.
        MemoryError
            If the upload does not fit in server memory (external, from
            ``read_all``).

        Contract
        --------
        Preconditions
            - Only trusted clients may call this. ``descriptor.command`` is
              unpickled with ``cloudpickle.loads`` before anything is checked,
              so a client that can reach ``do_put`` can run arbitrary code on
              the server. Restrict access with the server's auth middleware.
        """
        reader = instrument_reader(reader, rec, direction="in")

        table, kwargs, mode = parse_put_command(loads(descriptor.command))
        data = copy_rbr_batches(reader).read_all()
        with self.lock:
            put_table(self._conn, table, data, mode, **kwargs)

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
    @bounded_status
    @maybe_log_excepts
    def do_exchange(
        self,
        context: pa.flight.ServerCallContext,
        rec: "_Recorder",
        descriptor: pa.flight.FlightDescriptor,
        reader: pa.flight.MetadataRecordBatchReader,
        writer: pa.flight.MetadataRecordBatchWriter,
    ) -> None:
        """Run the exchanger ``descriptor`` names over the client's stream.

        Parameters
        ----------
        context
            The Flight server call context, passed to the exchanger.
        rec
            Metrics recorder, injected by ``instrument_rpc``.
        descriptor
            Its ``command`` is the UTF-8 name of a registered exchanger.
        reader
            The client's input stream; wrapped in ``CallScopedReader`` so
            nothing reads it after this call returns.
        writer
            The result stream back to the client.

        Raises
        ------
        pyarrow.flight.FlightServerError
            For any failure, via ``bounded_status``: an unknown exchange
            name (explicit ``ArrowInvalid``), a non-UTF-8 command (implicit
            ``UnicodeDecodeError``), or whatever the exchanger raises. The
            client sees ``Type: message`` with messages over
            ``STATUS_MESSAGE_BUDGET`` bytes cut in the middle; the original
            exception type and traceback stay on the server.

        Contract
        --------
        Preconditions
            - The whole call runs under ``self.lock``, so every other RPC on
              this server waits until the exchange ends, including one the
              exchanger itself would need.
        """
        with self.lock:
            cmd = descriptor.command.decode()
            handler = self.exchangers.get(cmd)
            if handler is None:
                raise pa.ArrowInvalid(f"Unknown exchange: {cmd}")

            reader = CallScopedReader(InstrumentedReader(reader, rec, direction="in"))
            iw = InstrumentedWriter(writer, rec, direction="out")

            try:
                return handler.exchange_f(context, reader, iw)
            finally:
                reader.close()
