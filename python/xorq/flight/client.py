from __future__ import annotations

import itertools
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from cloudpickle import dumps, loads
from pyarrow.flight import (
    FlightClient as _FlightClient,
)

from xorq.common.utils.logging_utils import get_logger
from xorq.common.utils.rbr_utils import (
    copy_rbr_batches,
)
from xorq.flight.enums import UploadMode
from xorq.vendor import ibis


if TYPE_CHECKING:
    from xorq.flight.server import PutCommand


executor = ThreadPoolExecutor()


logger = get_logger(__name__)


def make_put_descriptor(
    table_name: str, mode: UploadMode | str | None, kwargs: dict[str, Any]
) -> pa.flight.FlightDescriptor:
    """Pickle a ``do_put`` command into a ``FlightDescriptor``.

    Parameters
    ----------
    table_name
        Name of the target table on the server.
    mode
        A ``UploadMode`` or its string value, or ``None`` for append-or-create.
    kwargs
        Forwarded to the server backend's write.

    Returns
    -------
    pyarrow.flight.FlightDescriptor
        A command descriptor carrying a ``PutCommand``.

    Raises
    ------
    ValueError
        If ``mode`` is not a ``UploadMode`` value (explicit, ``UploadMode(mode)``),
        before anything is sent.
    TypeError
        If a value in ``kwargs`` cannot be pickled, e.g. a generator (implicit,
        from ``cloudpickle.dumps``).
    """
    command: PutCommand = {
        "table_name": table_name,
        "kwargs": kwargs,
        "mode": None if mode is None else UploadMode(mode).value,
    }
    return pa.flight.FlightDescriptor.for_command(dumps(command))


class FlightClient:
    def __init__(
        self,
        host="localhost",
        port=8815,
        username=None,
        password=None,
        cert_chain=None,
        private_key=None,
        tls_root_certs=None,
        healthcheck_n_tries=40,
        **kwargs,
    ):
        """
        Initialize the Ibis Backend Flight Client

        Args:
            host: Server host
            port: Server port
            username: User username
            password: User password
            tls_roots: TLS Root path
        """
        bytes_kwargs = {
            k: v
            for k, v in (
                ("cert_chain", cert_chain),
                ("private_key", private_key),
                ("tls_root_certs", tls_root_certs),
            )
            if v
        }
        scheme = "grpc+tls" if bytes_kwargs else "grpc"
        self._client = _FlightClient(
            f"{scheme}://{host}:{port}", **bytes_kwargs, **kwargs
        )
        self._wait_on_healthcheck(healthcheck_n_tries)

        if username and password:
            token_pair = self._client.authenticate_basic_token(
                username.encode(), password.encode()
            )
            self._options = pa.flight.FlightCallOptions(headers=[token_pair])
        else:
            self._options = None

    def _wait_on_healthcheck(self, n_tries=None, sleep_n=1):
        for attempt_i in itertools.islice(itertools.count(), n_tries):  # noqa: B007
            try:
                self.do_action(
                    "healthcheck",
                    options=pa.flight.FlightCallOptions(timeout=1),
                )
                logger.info("done healthcheck")
                return
            except pa.ArrowIOError as e:
                if "Deadline" in str(e):
                    logger.info("Server is not ready, waiting...")
                else:
                    raise e
            except pa.flight.FlightUnavailableError as e:
                matches = (
                    "Ssl handshake failed: SSL_ERROR_SSL: error:0A000086:SSL routines::certificate verify failed",
                    "Socket closed",
                )
                if any(to_match in str(e) for to_match in matches):
                    raise e
            except pa.flight.FlightUnauthenticatedError:
                return
            except Exception:
                pass
            logger.info(f"Flight server unavailable, sleeping {sleep_n} seconds")
            time.sleep(sleep_n)
        raise RuntimeError(f"failed to connect after {attempt_i + 1} attempts")

    # FIXME: rename to execute_table, add execute that return pd.DataFrame
    def execute(self, expr, **kwargs):
        """
        Execute Expr and return results as Arrow table

        Args:
            expr: xorq Expression

        Returns:
            pa.Table
        """

        batches = self.execute_batches(expr, **kwargs)
        return batches.read_all()

    def execute_batches(self, expr, **kwargs):
        # Get FlightInfo
        flight_info = self._client.get_flight_info(
            pa.flight.FlightDescriptor.for_command(
                dumps(
                    {
                        "expr": expr,
                        **kwargs,
                    }
                )
            ),
            options=self._options,
        )

        # Get the first endpoint
        endpoint = flight_info.endpoints[0]

        # Get the result
        reader = self._client.do_get(endpoint.ticket, options=self._options)

        return reader

    def upload(self, table_name, expr, **kwargs):
        return self.upload_batches(
            self, table_name, expr.to_pyarrow_batches(), **kwargs
        )

    def upload_table(
        self,
        table_name: str,
        data: pa.Table,
        mode: UploadMode | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Upload a ``pa.Table`` to the server as ``table_name``.

        Parameters
        ----------
        table_name
            Name of the target table on the server.
        data
            The rows to upload.
        mode
            A ``UploadMode`` or its string value: ``create``, ``append`` or
            ``replace``. ``None`` appends when the table exists and creates it
            otherwise.
        **kwargs
            Passed to the server backend's ``create_table`` or ``insert``.

        Raises
        ------
        ValueError
            If ``mode`` is not a ``UploadMode`` value, before anything is sent.
        TypeError
            If a value in ``kwargs`` cannot be pickled (from
            ``make_put_descriptor``), before anything is sent.
        pyarrow.ArrowException
            If the server's write fails, e.g. ``create`` on an existing table:
            ``FlightServerError``, or ``ArrowInvalid`` when the backend raised
            ``ValueError``. It surfaces at ``writer.close()``, after all of
            ``data`` has been sent.
        pyarrow.flight.FlightUnavailableError
            If the server cannot be reached (external, from ``do_put``).

        Contract
        --------
        Preconditions
            - ``kwargs`` must suit the server backend's ``create_table`` or
              ``insert``; the client cannot check this, so a bad keyword fails
              on the server, after the upload.
        """
        writer, _ = self._client.do_put(
            make_put_descriptor(table_name, mode, kwargs),
            data.schema,
            options=self._options,
        )
        writer.write_table(data)
        writer.close()

    def upload_batches(
        self,
        table_name: str,
        reader: pa.RecordBatchReader,
        mode: UploadMode | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Upload a ``pa.RecordBatchReader``; see ``upload_table``.

        Parameters, modes and errors are as for ``upload_table``.

        Contract
        --------
        Preconditions
            - ``reader`` is consumed, and a single-use reader cannot be
              replayed: after a failed upload, the caller needs a new reader
              to retry.
        """
        writer, _ = self._client.do_put(
            make_put_descriptor(table_name, mode, kwargs),
            reader.schema,
            options=self._options,
        )

        for batch in copy_rbr_batches(reader):
            writer.write_batch(batch)
        writer.done_writing()
        writer.close()

    def _do_action(
        self,
        action_type: str,
        action_body: Any = None,
        options: pa.flight.FlightCallOptions | None = None,
    ) -> Iterator[Any]:
        if action_body is None:
            action_body = {}

        action = pa.flight.Action(
            action_type,
            dumps(action_body),
        )
        logger.info(f"Running action {action_type}")
        return map(
            loads,
            (
                result.body.to_pybytes()
                for result in self._client.do_action(
                    action, options=options or self._options
                )
            ),
        )

    def do_action_one(self, action_type, action_body=None, options=None):
        return next(
            self._do_action(
                action_type, action_body=action_body, options=options or self._options
            )
        )

    def do_action(self, action_type, action_body=None, options=None):
        return tuple(
            self._do_action(
                action_type, action_body=action_body, options=options or self._options
            )
        )

    def do_exchange_batches(self, command, reader):
        def do_writes(writer, reader):
            writer.begin(reader.schema)
            i = -1
            for i, batch in enumerate(reader, 1):  # noqa: B007
                writer.write_batch(batch)
            writer.done_writing()
            return i

        def do_reads(_reader, queue):
            i = -1
            for i, batch in enumerate(_reader, 1):  # noqa: B007
                queue.put(batch.data)
            return i

        def do_writes_reads(command, reader, queue):
            # The consumer blocks until it sees the sentinel, so every exit path
            # must put one -- including a failure to open the exchange, which is
            # why `do_exchange` is inside the `try`. Callers drop `fut`, so an
            # exception left there goes unseen; it goes on the queue instead.
            try:
                descriptor = pa.flight.FlightDescriptor.for_command(command)
                writer, _reader = self._client.do_exchange(descriptor, self._options)
                # `with writer` must happen inside a future
                # # so its context remains alive during enclosed writes and reads
                with writer:
                    do_writes_fut = executor.submit(do_writes, writer, reader)
                    do_reads_fut = executor.submit(do_reads, _reader, queue)
                    (n_writes, n_reads) = (
                        do_writes_fut.result(),
                        do_reads_fut.result(),
                    )
            except BaseException as e:
                queue.put(e)
                raise
            queue.put(None)
            return {"n_writes": n_writes, "n_reads": n_reads}

        def queue_to_rbr(schema, queue):
            def queue_to_gen(queue):
                while (value := queue.get()) is not None:
                    # re-raise where the caller can see it: the pulling thread
                    if isinstance(value, BaseException):
                        raise value
                    yield value

            return pa.RecordBatchReader.from_batches(schema, queue_to_gen(queue))

        def get_output_schema(command, reader):
            (dct,) = self.do_action("query-exchange", command, options=self._options)
            assert dct["schema-in-condition"](ibis.schema(reader.schema))
            output_schema = dct["calc-schema-out"](
                ibis.schema(reader.schema)
            ).to_pyarrow()
            return output_schema

        queue = Queue()
        output_schema = get_output_schema(command, reader)
        fut = executor.submit(do_writes_reads, command, reader, queue)
        rbr = queue_to_rbr(output_schema, queue)
        return fut, rbr

    def do_exchange(self, command, expr):
        return self.do_exchange_batches(command, expr.to_pyarrow_batches())
