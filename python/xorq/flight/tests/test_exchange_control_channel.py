"""The exchange queue's control protocol: what ends a stream, and what fails it.

The queue between ``do_reads`` and the consumer carries record batches, an
end-of-stream marker and failures down one channel. These tests pin which is
which:

* a **read-side** failure is the stream's failure -- it must reach the consumer
  (that is the deadlock fix from #2231, guarded here against regression)
* a **write-side** failure of an exchange whose output already arrived is not --
  the result is complete and must be delivered
* a metadata-only chunk is data-shaped ``None`` and must not be mistaken for the
  end of the stream

Each drain runs under a deadline: the failure mode being guarded against is a
hang, and a hang inside pytest is a faulthandler dump rather than a failure.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pyarrow as pa
import pytest

import xorq.api as xo
from xorq.common.utils import classproperty
from xorq.flight import FlightServer
from xorq.flight.action import AddExchangeAction
from xorq.flight.exchanger import EchoExchanger


SCHEMA = pa.schema([("a", pa.int64())])


def make_batches(n_batches: int, n_rows: int = 100) -> pa.RecordBatchReader:
    batch = pa.RecordBatch.from_pydict({"a": list(range(n_rows))}, schema=SCHEMA)
    return pa.RecordBatchReader.from_batches(SCHEMA, (batch for _ in range(n_batches)))


def batches_then_raise(settle: float = 2.0) -> pa.RecordBatchReader:
    """One batch, a pause long enough for the server to answer and close, then a raise.

    The pause is what makes the test deterministic. Provoking the write-side
    error through gRPC flow control instead -- writing past the window until the
    half-closed stream rejects a batch -- races the server's reply: whether the
    already-sent output batch survives the torn-down call is up to the
    transport, and it did not on CI. Failing the *client's* input reader after
    the reads have demonstrably finished isolates the same rule with no
    transport race: the write side fails, and the completed stream must survive
    it.
    """
    batch = pa.RecordBatch.from_pydict({"a": list(range(100))}, schema=SCHEMA)

    def gen():
        yield batch
        time.sleep(settle)
        raise ValueError("write side died")

    return pa.RecordBatchReader.from_batches(SCHEMA, gen())


def drain_with_deadline(rbr: pa.RecordBatchReader, seconds: int = 60) -> pa.Table:
    """``rbr.read_all()`` on a daemon thread, so a deadlock fails the test."""
    box = {}

    def run() -> None:
        try:
            box["value"] = rbr.read_all()
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        raise AssertionError(f"exchange did not finish within {seconds}s: deadlocked")
    if "error" in box:
        raise box["error"]
    return box["value"]


class EarlyStopExchanger(EchoExchanger):
    """Writes the first batch back, then stops reading -- like any `limit`.

    The client is left writing into a half-closed stream, which is where the
    write-side error comes from.
    """

    @classproperty
    def exchange_f(cls) -> Any:
        def exchange(context, reader, writer, options=None, **kwargs) -> None:
            for chunk in reader:
                if chunk.data is not None:
                    writer.begin(chunk.data.schema, options=options)
                    writer.write_batch(chunk.data)
                    return

        return exchange

    @classproperty
    def command(cls) -> str:
        return "early-stop"


class MetadataChunkExchanger(EchoExchanger):
    """Emits a metadata-only chunk between two data batches."""

    @classproperty
    def exchange_f(cls) -> Any:
        def exchange(context, reader, writer, options=None, **kwargs) -> None:
            started = False
            for chunk in reader:
                if chunk.data is None:
                    continue
                if not started:
                    writer.begin(chunk.data.schema, options=options)
                    started = True
                writer.write_batch(chunk.data)
                writer.write_metadata(b"progress")
                writer.write_batch(chunk.data)

        return exchange

    @classproperty
    def command(cls) -> str:
        return "metadata-chunks"


class BoomExchanger(EchoExchanger):
    """Fails on the server side, mid-exchange."""

    @classproperty
    def exchange_f(cls) -> Any:
        def exchange(context, reader, writer, options=None, **kwargs) -> None:
            raise ValueError("boom")

        return exchange

    @classproperty
    def command(cls) -> str:
        return "boom"


@contextmanager
def serving(exchanger: type[EchoExchanger]) -> Iterator[Any]:
    with FlightServer(verify_client=False) as server:
        client = server.client
        client.do_action(AddExchangeAction.name, exchanger, options=client._options)
        yield client


def test_write_side_failure_does_not_fail_a_completed_exchange() -> None:
    """A server that stops reading early still returns its (complete) output.

    Regression guard for #2247: termination moved to `do_writes_reads`, which
    made *any* exception fatal to the stream -- including the client's own
    `Destination already closed` on an exchange the server had already answered
    in full.
    """
    with serving(EarlyStopExchanger) as client:
        (fut, rbr) = client.do_exchange_batches(
            EarlyStopExchanger.command, batches_then_raise()
        )
        table = drain_with_deadline(rbr)
    # the server answered in full before the write side died, so the consumer
    # gets the whole result
    assert table.num_rows == 100
    # and the write side's failure is the caller's to inspect -- never the
    # stream's
    with pytest.raises(ValueError, match="write side died"):
        fut.result()


def test_metadata_only_chunk_does_not_truncate_the_stream() -> None:
    """`chunk.data is None` is not end-of-stream (#2247)."""
    with serving(MetadataChunkExchanger) as client:
        (fut, rbr) = client.do_exchange_batches(
            MetadataChunkExchanger.command, make_batches(1)
        )
        table = drain_with_deadline(rbr)
        fut.result()
    # both data batches, not just the one before the metadata chunk
    assert table.num_rows == 200


def test_read_side_failure_still_reaches_the_consumer() -> None:
    """The #2231 deadlock fix, guarded: a server-side failure is the stream's."""
    with serving(BoomExchanger) as client:
        (_fut, rbr) = client.do_exchange_batches(BoomExchanger.command, make_batches(1))
        with pytest.raises(Exception, match="boom"):
            drain_with_deadline(rbr)


def test_udxf_failure_still_raises_end_to_end() -> None:
    """The same guard one layer up, through `flight_udxf`."""

    def boom(df: Any) -> Any:
        raise ValueError("udxf boom")

    schema = xo.schema({"unit": "int64"})
    udxf = xo.expr.relations.flight_udxf(
        process_df=boom,
        maybe_schema_in=schema,
        maybe_schema_out=schema,
        name="Boom",
    )
    expr = udxf(xo.memtable([{"unit": 1}], name="unit_tbl"))
    box = {}

    def run() -> None:
        try:
            box["value"] = xo.execute(expr)
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(60)
    assert not thread.is_alive(), "execute deadlocked"
    assert isinstance(box.get("error"), Exception)
    assert "udxf boom" in str(box["error"])
