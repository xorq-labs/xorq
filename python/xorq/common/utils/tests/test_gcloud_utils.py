"""Publish-safety for the GCS cache storage.

The local parquet storage stages every write under a name unique to the
writer, verifies it, and only then publishes it. GCStorage writes the same
artifacts to a bucket and owes the same guarantees: two processes caching one
expression derive one key, and streaming both into one object name leaves a
body no reader can decode.

Exercised against an in-memory fsspec filesystem -- what is under test is how
`put` stages, verifies and publishes, not gcsfs itself.
"""

import pyarrow as pa
import pytest

import xorq.api as xo
from xorq.common.exceptions import CacheIntegrityError


# fsspec, gcsfs and google-cloud-storage reach a test env only through the
# `examples` extra (via pins[gcs]), and `gcloud_utils` imports all three at
# module level. Without this, every job that collects this file -- including
# the ones that deselect it by marker -- dies during collection.
pytest.importorskip("fsspec")
pytest.importorskip("gcsfs")
pytest.importorskip("google.cloud.storage")

from fsspec.implementations.memory import MemoryFileSystem  # noqa: E402

import xorq.common.utils.gcloud_utils as gcloud_utils  # noqa: E402
from xorq.common.utils.gcloud_utils import GCStorage  # noqa: E402


# GCS tests are reserved for marked runs: the repo keeps the whole category
# off the default PR path. These particular ones need no credentials and no
# network -- they drive GCStorage against an in-memory filesystem -- but the
# rule is about where GCS-shaped tests belong, not what any one of them costs.
pytestmark = pytest.mark.gcs


SCHEMA = pa.schema([("i", pa.int64()), ("s", pa.string())])


class _Reader:
    """Stands in for the pyarrow RecordBatchReader that put() consumes."""

    def __init__(self, n_batches=3, rows_per_batch=100, raise_at=None):
        self.schema = SCHEMA
        self._args = (n_batches, rows_per_batch, raise_at)

    def __iter__(self):
        n_batches, rows_per_batch, raise_at = self._args
        for i in range(n_batches):
            if raise_at is not None and i == raise_at:
                raise ConnectionError("simulated stream drop")
            yield pa.record_batch(
                [
                    pa.array(range(i * rows_per_batch, (i + 1) * rows_per_batch)),
                    pa.array(["y" * 16] * rows_per_batch),
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
        return _Reader(**self._reader_kwargs)


@pytest.fixture
def storage(monkeypatch):
    # MemoryFileSystem keeps its store on the class, so it outlives instances.
    MemoryFileSystem.store.clear()
    MemoryFileSystem.pseudo_dirs.clear()
    # Stand in before construction: GCStorage builds its filesystem in
    # __attrs_post_init__, and resolving Google credentials to do it would
    # make these tests depend on the machine they run on.
    monkeypatch.setattr(gcloud_utils.gcsfs, "GCSFileSystem", MemoryFileSystem)
    return GCStorage(bucket_name="bucket", source=xo.connect())


def _tmp_objects(storage):
    return storage.fs.glob(f"{storage.bucket_name}/*.tmp")


def test_each_writer_stages_under_its_own_object_name(storage):
    key = "xorq_cache-concurrent"
    first, second = storage.get_tmp_path(key), storage.get_tmp_path(key)

    assert first != second
    assert first != storage.get_path(key)
    assert first.endswith(".tmp")


def test_put_publishes_nothing_when_the_stream_fails(storage):
    key = "xorq_cache-interrupted"

    with pytest.raises(ConnectionError):
        storage.put(key, _Node(n_batches=10, raise_at=5))

    assert not storage.fs.exists(storage.get_path(key))
    assert not _tmp_objects(storage), (
        "a failed write must not leave a full-size orphan on a bucket that bills for it"
    )


def test_put_refuses_and_preserves_a_short_write(storage, monkeypatch):
    key = "xorq_cache-shortwrite"
    real = gcloud_utils.rbr_to_fs

    def lying_write(fs, path, rbr, parquet_metadata=None, **kwargs):
        real(fs, path, rbr, parquet_metadata=parquet_metadata, **kwargs)
        return 10**9  # claim far more than was actually written

    monkeypatch.setattr(gcloud_utils, "rbr_to_fs", lying_write)

    with pytest.raises(CacheIntegrityError, match="inconsistent") as excinfo:
        storage.put(key, _Node())

    assert not storage.fs.exists(storage.get_path(key))
    assert not _tmp_objects(storage)
    (preserved,) = storage.fs.glob(f"{storage.bucket_name}/*.corrupt")
    assert preserved.lstrip("/") in str(excinfo.value)


def test_put_publishes_a_readable_object(storage):
    key = "xorq_cache-good"

    storage.put(key, _Node(n_batches=3, rows_per_batch=100))

    assert storage.exists(key) is True
    assert not _tmp_objects(storage)


def test_exists_rejects_an_unreadable_object(storage):
    key = "xorq_cache-garbage"
    with storage.fs.open(storage.get_path(key), "wb") as fh:
        fh.write(b"not a parquet file")

    assert storage.is_present(key) is True, "it is still there to be dropped"
    assert storage.exists(key) is False
