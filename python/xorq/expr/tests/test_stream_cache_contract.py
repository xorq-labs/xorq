"""Contract tests for the batchcorder ``StreamCache`` primitive xorq relies on.

``register_and_transform_remote_tables`` builds one ``StreamCache`` per
RemoteTable and passes ``max_readers`` from ``count_remote_table_readers``.
Three properties of that primitive are load-bearing for the fan-out fix and
are pinned here directly (rather than only through a backend):

1. Lazy, bounded ingestion -- the upstream stream is pulled on demand, not
   fully materialized up front (this is the whole reason StreamCache was
   chosen over pre-ingesting into a ``pa.Table``; see ADR-0013).
2. Replay from zero -- every reader up to ``max_readers`` replays the full
   stream independently, which is what makes a self-join over a one-shot
   remote reader work.
3. Hard cap -- the ``max_readers + 1``-th reader raises ``ValueError``. This
   is why an under-count is a correctness bug, not a memory miss (see the
   duckdb partition-window regression test).

True memory eviction (batches freed once all readers pass) is batchcorder's
own concern and is not observable through its public API, so it is not
asserted here.
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from batchcorder import StreamCache, StreamCacheReader


SCHEMA = pa.schema([("x", pa.int64())])


def _stream(n: int) -> pa.RecordBatchReader:
    def gen():
        for i in range(n):
            yield pa.record_batch({"x": [i]})

    return pa.RecordBatchReader.from_batches(SCHEMA, gen())


def _drain(reader: StreamCacheReader) -> list[int]:
    return [batch.column("x")[0].as_py() for batch in reader]


def test_ingestion_is_lazy_and_bounded() -> None:
    cache = StreamCache(_stream(50), max_readers=1)
    reader = iter(cache.reader())
    consumed = [next(reader) for _ in range(3)]
    assert len(consumed) == 3
    # only the demanded prefix was pulled from upstream; the rest is untouched
    assert cache.ingested_count == 3
    assert not cache.upstream_exhausted


def test_each_reader_replays_full_stream_from_zero() -> None:
    cache = StreamCache(_stream(5), max_readers=2)
    first = _drain(cache.reader())
    second = _drain(cache.reader())
    assert first == [0, 1, 2, 3, 4]
    # the second reader sees the whole stream from the start, not the remainder
    assert second == first
    assert cache.upstream_exhausted


def test_reader_beyond_max_readers_raises() -> None:
    cache = StreamCache(_stream(5), max_readers=2)
    _drain(cache.reader())
    _drain(cache.reader())
    # the 3rd reader exceeds the cap -- the failure that an undercounted
    # max_readers turns into a crash mid-query
    with pytest.raises(ValueError, match="Maximum number of readers"):
        _drain(cache.reader())


def test_unbounded_cache_allows_arbitrary_readers() -> None:
    # max_readers=None is the non-SQL / compile-failure fallback: no cap, no
    # eviction, always safe.
    cache = StreamCache(_stream(3), max_readers=None)
    for _ in range(5):
        assert _drain(cache.reader()) == [0, 1, 2]
