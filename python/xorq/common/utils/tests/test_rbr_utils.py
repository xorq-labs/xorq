from __future__ import annotations

import decimal
import threading
from contextlib import closing

import pyarrow as pa
import pyarrow.flight as flight

import xorq.api as xo
from xorq.common.utils.rbr_utils import (
    copy_rbr_batches,
    make_filtered_reader,
)


class _BatchServer(flight.FlightServerBase):
    """Minimal vanilla PyArrow Flight server that streams pre-built batches."""

    def __init__(self, location: str, batches: list[pa.RecordBatch]) -> None:
        super().__init__(location)
        self._batches = batches
        self._schema = batches[0].schema

    def do_get(
        self,
        context: flight.ServerCallContext,
        ticket: flight.Ticket,
    ) -> flight.GeneratorStream:
        return flight.GeneratorStream(self._schema, iter(self._batches))


def _decimal_batches(n_batches: int = 16, rows: int = 7) -> list[pa.RecordBatch]:
    """Ordinary ``decimal128`` batches.

    ``decimal128`` requires 16-byte buffer alignment, but the PyArrow (C++)
    Flight client only hands back 8-byte-aligned buffers, so every decimal
    data buffer arrives misaligned -- no buffer surgery required.
    """
    return [
        pa.record_batch(
            {
                # a tiny bool column shifts the following buffer's offset
                "pad": pa.array([True] * rows),
                "amount": pa.array(
                    [decimal.Decimal(f"{k}.0{i}") for i in range(rows)],
                    pa.decimal128(18, 2),
                ),
            }
        )
        for k in range(n_batches)
    ]


def _fetch_raw_flight_batches(
    batches: list[pa.RecordBatch],
) -> list[pa.RecordBatch]:
    """Round-trip ``batches`` through a real Flight server/client and return the
    RAW received chunks (the buffers as the C++ Flight client laid them out).

    We iterate ``reader`` chunk-by-chunk rather than calling ``read_all()`` on
    purpose: ``read_all()`` re-consolidates into freshly aligned buffers and
    would hide the misalignment. Iterating ``chunk.data`` is exactly what
    ``make_filtered_reader`` does in the live ingest path.
    """
    server = _BatchServer("grpc://127.0.0.1:0", batches)
    try:
        threading.Thread(target=server.serve, daemon=True).start()
        with closing(flight.FlightClient(f"grpc://127.0.0.1:{server.port}")) as client:
            reader = client.do_get(flight.Ticket(b"all"))
            return [chunk.data for chunk in reader if chunk.data is not None]
    finally:
        server.shutdown()


def test_copy_rbr_batches_realigns_misaligned_flight_decimals() -> None:
    """``copy_rbr_batches`` is the workaround for apache/arrow-rs#6471 and is
    still load-bearing on the versions we pin.

    Reproduces the real, non-synthetic trigger: ordinary ``decimal128`` data
    fetched over Arrow Flight. The PyArrow/C++ Flight client returns buffers
    aligned to only 8 bytes, but ``decimal128`` needs 16-byte alignment, so the
    received buffers are misaligned. Feeding them straight into datafusion's
    ``read_record_batches`` (a ``RecordBatchReader`` C-stream FFI import) makes
    the Rust side reject them -- the batches are silently dropped and the table
    comes back empty.

    The upstream fix (arrow-rs#6472) only realigns the single-batch
    ``RecordBatch.from_pyarrow_bound`` path; the maintainers deliberately kept
    the general ``ffi::from_ffi`` import (which the reader path uses) strict, so
    this cannot be relied on and ``copy_rbr_batches`` -- which reallocates every
    batch through ``batch.copy_to(default_cpu_memory_manager())`` -- is still
    required. Turn it into a passthrough and this test fails with 0 rows.
    """
    batches = _decimal_batches()
    raw = _fetch_raw_flight_batches(batches)
    expected_rows = sum(b.num_rows for b in batches)

    # precondition: the Flight client really did hand back misaligned decimal
    # buffers, otherwise this would pass vacuously even if the copy were a no-op.
    assert any(b.column("amount").buffers()[1].address % 16 != 0 for b in raw), (
        "expected misaligned decimal128 buffers from the Flight client"
    )

    con = xo.connect()
    reader = copy_rbr_batches(pa.RecordBatchReader.from_batches(batches[0].schema, raw))
    con.read_record_batches(reader, table_name="t")
    out = con.table("t").execute()
    assert len(out) == expected_rows


def test_copy_rbr_batches_preserves_values_and_schema() -> None:
    """The copy is a pure realignment: same schema, same data, same batching."""
    batches = [
        pa.record_batch({"a": pa.array([1, 2, 3]), "b": pa.array(["x", "y", "z"])}),
        pa.record_batch({"a": pa.array([4, 5]), "b": pa.array(["p", "q"])}),
    ]
    reader = pa.RecordBatchReader.from_batches(batches[0].schema, batches)
    out = copy_rbr_batches(reader).read_all()
    assert out.schema.equals(batches[0].schema)
    assert out.equals(pa.Table.from_batches(batches))


def test_make_filtered_reader_unwraps_flight_chunks_and_drops_empty() -> None:
    """``make_filtered_reader`` adapts a Flight stream (which yields
    ``FlightStreamChunk`` wrappers, each carrying ``.data`` -- a RecordBatch or
    ``None`` for metadata-only messages) into a plain ``RecordBatchReader``,
    unwrapping ``.data`` and dropping chunks with no data.
    """
    flight_schema = pa.schema([("a", pa.int64())])
    real = pa.record_batch({"a": pa.array([1, 2, 3])})

    class _Chunk:
        def __init__(self, data: pa.RecordBatch | None) -> None:
            self.data = data

    class _FlightLikeReader:
        schema = flight_schema

        def __iter__(self):  # noqa: ANN204
            yield _Chunk(None)  # metadata-only chunk -> dropped
            yield _Chunk(real)  # real batch -> unwrapped and kept

    out = make_filtered_reader(_FlightLikeReader()).read_all()
    assert out.equals(pa.Table.from_batches([real]))


def test_make_filtered_reader_schema_survives_all_empty() -> None:
    """A stream of only metadata-only chunks still yields a schema-correct,
    zero-row table rather than raising."""
    flight_schema = pa.schema([("a", pa.int64())])

    class _Chunk:
        def __init__(self, data: pa.RecordBatch | None) -> None:
            self.data = data

    class _FlightLikeReader:
        schema = flight_schema

        def __iter__(self):  # noqa: ANN204
            yield _Chunk(None)

    out = make_filtered_reader(_FlightLikeReader()).read_all()
    assert out.schema.equals(flight_schema)
    assert out.num_rows == 0
