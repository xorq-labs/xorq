"""Integrity guarantees for the parquet cache.

Regression cover for a corruption reported against Redshift and reproduced
locally: two processes caching the same expression derived the same cache key,
and therefore the same key-only temp path, and interleaved their writes into
it. The later finisher renamed its footer over the mixed body and returned
successfully; `exists()` then reported that artifact as a cache hit forever.
"""

import datetime
import multiprocessing as mp
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import xorq.caching.storage as storage_module
from xorq.caching.storage import (
    ParquetStorage,
    _reap_stale_tmp,
    _verify_parquet,
)
from xorq.common.exceptions import CacheIntegrityError


SCHEMA = pa.schema([("i", pa.int64()), ("s", pa.string())])


class _Reader:
    """Stands in for the pyarrow RecordBatchReader that put() consumes."""

    def __init__(self, n_batches, rows_per_batch, width, tag, raise_at=None):
        self.schema = SCHEMA
        self._args = (n_batches, rows_per_batch, width, tag, raise_at)

    def __iter__(self):
        n_batches, rows_per_batch, width, tag, raise_at = self._args
        for i in range(n_batches):
            if raise_at is not None and i == raise_at:
                raise ConnectionError("simulated stream drop")
            yield pa.record_batch(
                [
                    pa.array(range(i * rows_per_batch, (i + 1) * rows_per_batch)),
                    pa.array([f"{tag}-" + "y" * width] * rows_per_batch),
                ],
                schema=SCHEMA,
            )


class _Node:
    """Minimal stand-in for the Node that put() materializes."""

    def __init__(self, **reader_kwargs):
        self._reader_kwargs = reader_kwargs

    def to_expr(self):
        return self

    def to_pyarrow_batches(self):
        reader = _Reader(**self._reader_kwargs)

        class _CM:
            def __enter__(self):
                return reader

            def __exit__(self, *exc):
                return False

        return _CM()


def _storage(base):
    return ParquetStorage(relative_path=Path("."), base_path=Path(base))


def _write_one(base, key, n_batches, rows_per_batch, width, tag):
    """Module-level so it survives pickling into a spawned process."""
    try:
        _storage(base).put(
            key,
            _Node(
                n_batches=n_batches, rows_per_batch=rows_per_batch, width=width, tag=tag
            ),
        )
        return "ok"
    except Exception as e:  # noqa: BLE001 - the loser's rename legitimately fails
        return f"{type(e).__name__}"


def test_concurrent_writers_on_one_key_do_not_corrupt(tmp_path):
    """The reported failure: two writers, one key, asymmetric payloads.

    Payload shapes differ on purpose. Same-shaped writers produce byte-identical
    files whose interleaved writes land at matching offsets, which masks the
    damage -- before the fix this reproduced in 5 of 8 runs only once the
    writers diverged in batch count, row count and string width.
    """
    key = "xorq_cache-concurrent"
    ctx = mp.get_context("spawn")
    with ctx.Pool(2) as pool:
        pool.starmap(
            _write_one,
            [
                (str(tmp_path), key, 40, 5000, 120, "w0"),
                (str(tmp_path), key, 12, 11000, 37, "w1"),
            ],
        )

    published = _storage(tmp_path).get_path(key)
    assert published.exists(), "one writer must still publish a result"

    # The published artifact must decode, and its footer must agree with what
    # is actually readable. A valid footer over an undecodable body is exactly
    # the reported corruption.
    pf = pq.ParquetFile(published)
    readable = sum(batch.num_rows for batch in pf.iter_batches())
    assert readable == pf.metadata.num_rows
    # ...and it must be exactly one writer's output, not a blend of both.
    assert readable in {40 * 5000, 12 * 11000}

    assert not list(tmp_path.glob("*.tmp")), "no temp file may be left behind"


def test_failed_stream_publishes_nothing_and_leaves_no_orphan(tmp_path):
    storage = _storage(tmp_path)
    key = "xorq_cache-interrupted"

    with pytest.raises(ConnectionError):
        storage.put(
            key, _Node(n_batches=10, rows_per_batch=1000, width=64, tag="x", raise_at=5)
        )

    assert not storage.get_path(key).exists()
    assert storage.exists(key) is False
    assert not list(tmp_path.glob("*.tmp")), (
        "a failed write must not leave a full-size orphan behind"
    )


def test_put_refuses_to_publish_a_short_write(tmp_path, monkeypatch):
    """If what landed disagrees with what was streamed, nothing is published."""
    storage = _storage(tmp_path)
    key = "xorq_cache-shortwrite"

    real = storage_module._write_parquet

    def lying_write(path, batch_reader, parquet_metadata=None):
        real(path, batch_reader, parquet_metadata=parquet_metadata)
        return 10**9  # claim far more than was actually written

    monkeypatch.setattr("xorq.caching.storage._write_parquet", lying_write)

    with pytest.raises(CacheIntegrityError, match="inconsistent"):
        storage.put(key, _Node(n_batches=3, rows_per_batch=100, width=16, tag="x"))

    assert not storage.get_path(key).exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_verify_parquet_rejects_a_truncated_body(tmp_path):
    path = tmp_path / "truncated.parquet"
    pq.write_table(pa.table({"i": list(range(5000)), "s": ["z" * 64] * 5000}), path)
    full = path.read_bytes()
    # Keep the footer, drop the middle: the shape the reporter saw.
    path.write_bytes(full[: len(full) // 3] + full[-2048:])

    with pytest.raises(CacheIntegrityError):
        _verify_parquet(path, 5000)


def test_exists_rejects_an_unreadable_artifact(tmp_path):
    storage = _storage(tmp_path)
    key = "xorq_cache-garbage"
    storage._ensure_dir()
    storage.get_path(key).write_bytes(b"not a parquet file")
    assert storage.exists(key) is False


def test_reap_stale_tmp_spares_recent_writes(tmp_path):
    fresh = tmp_path / "a.parquet.tmp"
    stale = tmp_path / "b.parquet.tmp"
    for f in (fresh, stale):
        f.write_bytes(b"x")
    old = (datetime.datetime.now() - datetime.timedelta(days=3)).timestamp()
    os.utime(stale, (old, old))

    _reap_stale_tmp(tmp_path)

    assert fresh.exists(), "a temp file may belong to a write still streaming"
    assert not stale.exists()
