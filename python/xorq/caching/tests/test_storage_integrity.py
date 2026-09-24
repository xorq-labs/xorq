"""Integrity guarantees for the parquet cache.

Regression cover for a corruption reported against Redshift and reproduced
locally: two processes caching the same expression derived the same cache key,
and therefore the same key-only temp path, and interleaved their writes into
it. The later finisher renamed its footer over the mixed body and returned
successfully; `exists()` then reported that artifact as a cache hit forever.
"""

import datetime
import errno
import logging
import multiprocessing as mp
import os
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import xorq.api as xo
import xorq.caching.storage as storage_module
from xorq.caching import ParquetSnapshotCache
from xorq.caching.storage import (
    ParquetStorage,
    ParquetTTLStorage,
    _reap_stale_tmp,
    _reap_stale_tmp_once,
    verify_parquet,
)
from xorq.common.exceptions import CacheIntegrityError
from xorq.config import options


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


def _tmp_name(key="xorq_cache-x"):
    """A temp name of the shape put() writes, for the reaper to recognize."""
    return f"{key}.parquet.{os.getpid()}.{uuid.uuid4().hex}.tmp"


def _make_stale(path):
    path.write_bytes(b"x")
    old = (datetime.datetime.now() - datetime.timedelta(days=3)).timestamp()
    os.utime(path, (old, old))
    return path


def _lie_about_rows(monkeypatch):
    """Make _write_parquet claim more rows than it wrote."""
    real = storage_module._write_parquet

    def lying_write(path, batch_reader, parquet_metadata=None):
        real(path, batch_reader, parquet_metadata=parquet_metadata)
        return 10**9

    monkeypatch.setattr("xorq.caching.storage._write_parquet", lying_write)


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
    _lie_about_rows(monkeypatch)

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
        verify_parquet(path, 5000)


def test_exists_rejects_an_unreadable_artifact(tmp_path):
    storage = _storage(tmp_path)
    key = "xorq_cache-garbage"
    storage._ensure_dir()
    storage.get_path(key).write_bytes(b"not a parquet file")
    assert storage.exists(key) is False


def test_reap_stale_tmp_spares_recent_writes(tmp_path):
    fresh = tmp_path / _tmp_name("xorq_cache-fresh")
    fresh.write_bytes(b"x")
    stale = _make_stale(tmp_path / _tmp_name("xorq_cache-stale"))

    _reap_stale_tmp(tmp_path)

    assert fresh.exists(), "a temp file may belong to a write still streaming"
    assert not stale.exists()


def test_reap_stale_tmp_spares_temp_files_that_are_not_ours(tmp_path):
    """A cache directory is an ordinary directory other writers stage into.

    ``ParquetWriteThrough`` puts ``<name>.parquet.<random>.tmp`` beside its
    target; deleting one because it sits in the cache directory and ends in
    ``.tmp`` would destroy a write we know nothing about.
    """
    foreign = _make_stale(tmp_path / "out.parquet.a1b2c3d4.tmp")
    ours = _make_stale(tmp_path / _tmp_name())

    _reap_stale_tmp(tmp_path)

    assert foreign.exists()
    assert not ours.exists()


def test_reap_scans_a_directory_only_once_per_process(tmp_path):
    """put() is on the miss path of every query; the scan is not worth repeating."""
    first = _make_stale(tmp_path / _tmp_name("xorq_cache-first"))
    _reap_stale_tmp_once(tmp_path)
    assert not first.exists()

    second = _make_stale(tmp_path / _tmp_name("xorq_cache-second"))
    _reap_stale_tmp_once(tmp_path)
    assert second.exists()


def test_a_rejected_write_is_kept_for_inspection(tmp_path, monkeypatch):
    """The error names a file, so the file has to still be there to look at."""
    storage = _storage(tmp_path)
    key = "xorq_cache-evidence"
    _lie_about_rows(monkeypatch)

    with pytest.raises(CacheIntegrityError) as excinfo:
        storage.put(key, _Node(n_batches=3, rows_per_batch=100, width=16, tag="x"))

    (preserved,) = tmp_path.glob("*.corrupt")
    assert str(preserved) in str(excinfo.value)
    assert not storage.get_path(key).exists(), "nothing may be published"
    assert not list(tmp_path.glob("*.tmp"))


def test_verification_can_be_turned_off(tmp_path, monkeypatch):
    """The read-back doubles the cost of a write; a deployment may decline it."""
    storage = _storage(tmp_path)
    key = "xorq_cache-unverified"
    _lie_about_rows(monkeypatch)
    monkeypatch.setattr(options.cache, "verify_writes", False)

    storage.put(key, _Node(n_batches=3, rows_per_batch=100, width=16, tag="x"))

    assert storage.exists(key)


def test_ttl_storage_refuses_an_unreadable_artifact(tmp_path):
    """A TTL hit is still a hit: it must not serve a corrupt file for a whole day."""
    storage = ParquetTTLStorage(relative_path=Path("."), base_path=tmp_path)
    key = "xorq_cache-ttl-garbage"
    storage._ensure_dir()
    storage.get_path(key).write_bytes(b"not a parquet file")

    assert storage.is_present(key) is True
    assert storage.exists(key) is False


def test_exists_does_not_report_an_os_failure_as_a_miss(tmp_path, monkeypatch):
    """EMFILE says nothing about the artifact.

    A miss here would recompute the expression and then overwrite a file that
    was never bad -- silently, since a miss is the ordinary case.
    """
    storage = _storage(tmp_path)
    key = "xorq_cache-emfile"
    storage._ensure_dir()
    pq.write_table(pa.table({"i": [1, 2, 3]}), storage.get_path(key))
    assert storage.exists(key) is True

    def too_many_open_files(*args, **kwargs):
        raise OSError(errno.EMFILE, "Too many open files")

    monkeypatch.setattr(pq, "ParquetFile", too_many_open_files)

    with pytest.raises(OSError) as excinfo:
        storage.exists(key)
    assert excinfo.value.errno == errno.EMFILE


def test_a_corrupt_artifact_names_itself_and_can_be_dropped(tmp_path):
    """The two ways out of a corrupt artifact, through the public API.

    `exists` refuses it, so neither may be gated on `exists`: a diagnosis that
    said only KeyError would send the user looking for a missing key, and a
    drop that refused would leave `rm` as the only way to clear it.
    """
    cache = ParquetSnapshotCache.from_kwargs(
        relative_path=Path("."), base_path=tmp_path
    )
    expr = xo.memtable({"i": [1, 2, 3]})
    key = cache.calc_key(expr)
    cache.storage._ensure_dir()
    cache.storage.get_path(key).write_bytes(b"not a parquet file")

    with pytest.raises(CacheIntegrityError, match="could not be read"):
        cache.get(expr)

    cache.drop(expr)
    assert not cache.storage.get_path(key).exists()

    with pytest.raises(KeyError):
        cache.drop(expr)


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(
            pa.ArrowMemoryError("malloc of 1073741824 bytes failed"), id="oom"
        ),
        pytest.param(pa.ArrowCancelled("cancelled"), id="cancelled"),
        pytest.param(pa.ArrowCapacityError("array would exceed 2GB"), id="capacity"),
    ],
)
def test_exists_does_not_report_pyarrow_resource_failure_as_corruption(
    tmp_path, monkeypatch, exc
):
    """An OOM is about this machine, not about the bytes on disk.

    These are all `pa.ArrowException` subclasses, so testing the family rather
    than the specific corruption types swallows every one of them: `exists`
    would answer False, the caller would recompute, and a sound artifact would
    be overwritten -- the exact failure the errno check exists to prevent,
    arriving through the other branch.
    """
    storage = _storage(tmp_path)
    key = "xorq_cache-oom"
    storage._ensure_dir()
    pq.write_table(pa.table({"i": [1, 2, 3]}), storage.get_path(key))
    assert storage.exists(key) is True

    def raise_it(*args, **kwargs):
        raise exc

    monkeypatch.setattr(pq, "ParquetFile", raise_it)

    with pytest.raises(type(exc)):
        storage.exists(key)


def test_put_does_not_quarantine_a_good_write_on_a_resource_failure(
    tmp_path, monkeypatch
):
    """Read-back is the memory-hungry half of a write, so this is where an OOM lands.

    Quarantining here would move a perfectly good artifact to `.corrupt` and
    fail the write.
    """
    storage = _storage(tmp_path)
    key = "xorq_cache-oom-on-verify"

    def raise_oom(*args, **kwargs):
        raise pa.ArrowMemoryError("malloc failed during read-back")

    monkeypatch.setattr(pq, "ParquetFile", raise_oom)

    with pytest.raises(pa.ArrowMemoryError):
        storage.put(key, xo.memtable({"i": [1, 2, 3]}).op())

    assert not list(tmp_path.glob("*.corrupt")), "a good write was quarantined"


def test_a_corrupt_artifact_is_logged_before_it_is_treated_as_a_miss(tmp_path, caplog):
    """Healing is right; healing in silence is how the incident went undiagnosed.

    `exists` answers False so the caller recomputes and publishes over the
    damage. Without a log line that is indistinguishable from an ordinary
    miss: the user sees a slow run and never learns the cache was corrupt.
    """
    storage = _storage(tmp_path)
    key = "xorq_cache-noisy"
    storage._ensure_dir()
    storage.get_path(key).write_bytes(b"not a parquet file")

    with caplog.at_level(logging.WARNING, logger="xorq.caching.storage"):
        assert storage.exists(key) is False

    assert "corrupt cache artifact ignored" in caplog.text
    assert key in caplog.text
