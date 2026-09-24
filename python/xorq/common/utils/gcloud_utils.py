import os
import uuid

import gcsfs
import pyarrow as pa
import pyarrow.parquet as pq
from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
)
from google.cloud import storage
from toolz import curry

from xorq.caching.storage import (
    CacheStorage,
    quarantine,
    read_parquet_metadata,
    verify_parquet,
    verify_writes_enabled,
    warn_corrupt_artifact,
)
from xorq.common.exceptions import CacheIntegrityError
from xorq.config import default_backend
from xorq.vendor.ibis.backends import BaseBackend


@curry
def rbr_from_fs(fs, path):
    def get_schema(fs, path):
        with fs.open(path, "rb") as fh:
            pf = pq.ParquetFile(fh)
            schema = pf.schema.to_arrow_schema()
            return schema

    def gen_batches(fs, path):
        with fs.open(path, "rb") as fh:
            pf = pq.ParquetFile(fh)
            yield from pf.iter_batches()

    rbr = pa.RecordBatchReader.from_batches(
        get_schema(fs, path),
        gen_batches(fs, path),
    )
    return rbr


@curry
def rbr_to_fs(fs, path, rbr, parquet_metadata=None, **kwargs):
    """Stream *rbr* to *path* on *fs*; return the number of rows written.

    The row count is the caller's only independent handle on what actually
    landed -- see `xorq.caching.storage._write_parquet`, which says the same of
    the local writer.
    """
    schema = rbr.schema
    if parquet_metadata is not None:
        from xorq.common.utils.provenance_utils import (  # noqa: PLC0415
            inject_metadata_into_schema,
        )

        schema = inject_metadata_into_schema(schema, parquet_metadata)
    n_rows = 0
    with fs.open(path, "wb") as fh:
        with pq.ParquetWriter(fh, schema, **kwargs) as writer:
            for batch in rbr:
                writer.write_batch(batch)
                n_rows += batch.num_rows
    return n_rows


@frozen
class GCStorage(CacheStorage):
    bucket_name: str = field(validator=instance_of(str))
    source = field(
        validator=instance_of(BaseBackend),
        factory=default_backend,
    )
    fs: gcsfs.GCSFileSystem = field(init=False)

    def __attrs_post_init__(self):
        assert hasattr(self.source, "read_record_batches")
        object.__setattr__(self, "fs", gcsfs.GCSFileSystem())

    def __dasher_tokenize__(self):
        return ("normalize_gc_storage", self.source, self.bucket_name)

    def get_path(self, key):
        path = f"{self.bucket_name}/{key}.parquet"
        return path

    def get_tmp_path(self, key):
        """Return an object name for this write that no other writer can hold.

        Same reasoning as the local storage: two processes caching the same
        expression derive the same key, and a key-only temp name would have
        them streaming into one object, whose interleaved body a later reader
        cannot decode.
        """
        return f"{self.get_path(key)}.{os.getpid()}.{uuid.uuid4().hex}.tmp"

    def is_present(self, key):
        return self.fs.exists(self.get_path(key))

    def check_integrity(self, key):
        path = self.get_path(key)
        if not self.fs.exists(path):
            return
        with self.fs.open(path, "rb") as fh:
            read_parquet_metadata(fh, name=path)

    def exists(self, key):
        if not self.is_present(key):
            return False
        try:
            self.check_integrity(key)
        except CacheIntegrityError as e:
            warn_corrupt_artifact(key, self.get_path(key), e)
            return False
        return True

    def get(self, key):
        path = self.get_path(key)
        rbr = rbr_from_fs(self.fs, path)
        op = self.source.read_record_batches(rbr).op()
        return op

    def put(self, key, value, parquet_metadata=None):
        path = self.get_path(key)
        tmp_path = self.get_tmp_path(key)
        try:
            rbr = value.to_expr().to_pyarrow_batches()
            n_rows = rbr_to_fs(
                self.fs, tmp_path, rbr, parquet_metadata=parquet_metadata
            )
            # Verify before publishing, never after: once the object is under
            # `path` it is indistinguishable from a good one.
            if verify_writes_enabled():
                with self.fs.open(tmp_path, "rb") as fh:
                    verify_parquet(fh, n_rows, name=tmp_path)
            # Not atomic the way a POSIX rename is -- gcsfs copies, then
            # deletes -- but nothing incomplete ever appears under `path`,
            # which is what a shared object name gave up.
            self.fs.mv(tmp_path, path)
        except CacheIntegrityError as e:
            raise quarantine(e, tmp_path, self.fs.mv) from e
        except BaseException:
            # Nothing was published, so the temp object is ours alone; leaving
            # it would leave a full-size orphan on a bucket that bills for it.
            try:
                self.fs.rm(tmp_path)
            except Exception:  # noqa: BLE001 - cleanup must not mask the error
                pass
            raise
        return self.get(key)

    def drop(self, key):
        path = self.get_path(key)
        self.fs.delete(path)


def get_file_metadata(uri, client=None):
    blob = storage.Blob.from_string(uri)
    # Refresh metadata (required for accurate timestamps)
    blob.reload(client or storage.Client.create_anonymous_client())

    # Extract relevant metadata
    metadata = tuple(
        (name, getattr(blob, name))
        for name in (
            "content_type",
            "updated",
            "size",
        )
    )

    return metadata
