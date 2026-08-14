"""``instrument_reader`` on an empty stream.

It used to pull the first batch with an explicit ``next(reader)`` inside a
generator, so an empty stream raised ``StopIteration`` there -- which PEP 479
turns into ``RuntimeError: generator raised StopIteration``. Reachable via
``do_instrument_reader=True``, i.e. exactly when instrumentation is on to debug
a failure that produced an empty result.
"""

from __future__ import annotations

import pyarrow as pa

from xorq.common.utils.rbr_utils import instrument_reader


SCHEMA = pa.schema([("a", pa.int64())])


def test_instrument_reader_handles_an_empty_stream() -> None:
    empty = pa.RecordBatchReader.from_batches(SCHEMA, iter(()))
    assert instrument_reader(empty, "test: ").read_all().num_rows == 0


def test_instrument_reader_yields_every_batch() -> None:
    batch = pa.RecordBatch.from_pydict({"a": [1, 2, 3]}, schema=SCHEMA)
    reader = pa.RecordBatchReader.from_batches(SCHEMA, iter((batch, batch)))
    assert instrument_reader(reader, "test: ").read_all().num_rows == 6
