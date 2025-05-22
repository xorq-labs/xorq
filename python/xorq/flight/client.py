import logging
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import pyarrow as pa
import pyarrow.flight
from cloudpickle import dumps, loads

from xorq.common.utils.rbr_utils import (
    copy_rbr_batches,
)


executor = ThreadPoolExecutor()


logger = logging.getLogger(__name__)


# fixme: make port dynamic and automagically find an open one
class FlightClient:
    def __init__(
        self,
        host="localhost",
        port=8815,
        username=None,
        password=None,
        tls_roots=None,
        mlts=None,
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
        kwargs = {}
        if tls_roots:
            with open(tls_roots, "rb") as root_certs:
                kwargs = {
                    "tls_root_certs": root_certs.read(),
                    "disable_server_verification": False,
                }

        if mlts and not tls_roots:
            raise ValueError("TLS must be enabled in order to use MTLS, aborting.")
        elif mlts:
            cert_file_path, key_file_path = mlts
            with open(cert_file_path, "rb") as cert_file:
                mtls_cert_chain = cert_file.read()

            with open(key_file_path, "rb") as key_file:
                mtls_private_key = key_file.read()

            kwargs["cert_chain"] = mtls_cert_chain
            kwargs["private_key"] = mtls_private_key

        scheme = "grpc+tls" if tls_roots else "grpc"

        self._client = pa.flight.FlightClient(f"{scheme}://{host}:{port}", **kwargs)
        self._wait_on_healthcheck()

        if username and password:
            token_pair = self._client.authenticate_basic_token(
                username.encode(), password.encode()
            )
            self._options = pa.flight.FlightCallOptions(headers=[token_pair])
        else:
            self._options = None

    def _wait_on_healthcheck(self):
        while True:
            try:
                self.do_action(
                    "healthcheck",
                    options=pa.flight.FlightCallOptions(timeout=1),
                )
                logger.info("done healthcheck")
                break
            except pa.ArrowIOError as e:
                if "Deadline" in str(e):
                    logger.info("Server is not ready, waiting...")
                else:
                    raise e
            except pa.flight.FlightUnavailableError:
                pass
            except pa.flight.FlightUnauthenticatedError:
                break
            except Exception as ee:
                print(ee)
            finally:
                n_seconds = 1
                logger.info(f"Flight server unavailable, sleeping {n_seconds} seconds")
                time.sleep(n_seconds)

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

    def upload_data(self, table_name, data, **kwargs):
        """
        Upload data to create or replace a table

        Args:
            table_name: Name of the table to create
            data: pa.Table containing the data
            **kwargs: Additional metadata (e.g., "target": "duckdb" or "iceberg")
        """
        command_payload = {"table_name": table_name, "kwargs": kwargs}

        writer, _ = self._client.do_put(
            pa.flight.FlightDescriptor.for_command(dumps(command_payload)),
            data.schema,
            options=self._options,
        )
        writer.write_table(data)
        writer.close()

    def upload_batches(self, table_name, reader, **kwargs):
        command_payload = {"table_name": table_name, "kwargs": kwargs}

        writer, _ = self._client.do_put(
            pa.flight.FlightDescriptor.for_command(dumps(command_payload)),
            reader.schema,
            options=self._options,
        )

        for batch in copy_rbr_batches(reader):
            writer.write_batch(batch)
        writer.done_writing()
        writer.close()

    def _do_action(self, action_type, action_body=None, options=None):
        if action_body is None:
            action_body = {}

        try:
            action = pa.flight.Action(
                action_type,
                dumps(action_body),
            )
            logger.info(f"Running action {action_type}")
            return map(
                loads,
                (
                    result.body.to_pybytes()
                    for result in self._client.do_action(action, options=options)
                ),
            )

        except pa.lib.ArrowIOError as e:
            logger.debug(f"Error calling action: {e}")

    def do_action_one(self, action_type, action_body=None, options=None):
        return next(
            self._do_action(action_type, action_body=action_body, options=options)
        )

    def do_action(self, action_type, action_body=None, options=None):
        return tuple(
            self._do_action(action_type, action_body=action_body, options=options)
        )

    def do_exchange_batches(self, command, reader):
        def do_writes(writer, reader):
            writer.begin(reader.schema)
            i = -1
            for i, batch in enumerate(reader, 1):
                writer.write_batch(batch)
            writer.done_writing()
            return i

        def do_reads(_reader, queue):
            i = -1
            for i, batch in enumerate(_reader, 1):
                queue.put(batch.data)
            queue.put(None)
            return i

        def do_writes_reads(command, reader, queue):
            descriptor = pa.flight.FlightDescriptor.for_command(command)
            writer, _reader = self._client.do_exchange(descriptor, self._options)
            # `with writer` must happen inside a future
            # # so its context remains alive during enclosed writes and reads
            with writer:
                do_writes_fut = executor.submit(do_writes, writer, reader)
                do_reads_fut = executor.submit(do_reads, _reader, queue)
                (n_writes, n_reads) = (do_writes_fut.result(), do_reads_fut.result())
            return {"n_writes": n_writes, "n_reads": n_reads}

        def queue_to_rbr(schema, queue):
            def queue_to_gen(queue):
                while (value := queue.get()) is not None:
                    yield value

            return pa.RecordBatchReader.from_batches(schema, queue_to_gen(queue))

        def get_output_schema(command, reader):
            (dct,) = self.do_action("query-exchange", command, options=self._options)
            assert dct["schema-in-condition"](reader.schema)
            output_schema = dct["calc-schema-out"](reader.schema)
            return output_schema

        queue = Queue()
        output_schema = get_output_schema(command, reader)
        fut = executor.submit(do_writes_reads, command, reader, queue)
        rbr = queue_to_rbr(output_schema, queue)
        return fut, rbr

    do_exchange = do_exchange_batches
