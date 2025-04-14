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

import xorq as xo
from xorq.caching import CacheStorage


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
def rbr_to_fs(fs, path, rbr, **kwargs):
    with fs.open(path, "wb") as fh:
        with pq.ParquetWriter(fh, rbr.schema, **kwargs) as writer:
            for batch in rbr:
                writer.write_batch(batch)


@frozen
class _GCStorage(CacheStorage):
    bucket_name: str = field(validator=instance_of(str))
    source = field(
        validator=instance_of(xo.vendor.ibis.backends.BaseBackend),
        factory=xo.config._backend_init,
    )
    fs: gcsfs.GCSFileSystem = field(init=False)

    def __attrs_post_init__(self):
        assert hasattr(self.source, "read_record_batches")
        object.__setattr__(self, "fs", gcsfs.GCSFileSystem())

    def key_exists(self, key):
        path = self.calc_path(key)
        return self.fs.exists(path)

    def calc_path(self, key):
        path = f"{self.bucket_name}/{key}.parquet"
        return path

    def _get(self, key):
        path = self.calc_path(key)
        rbr = rbr_from_fs(self.fs, path)
        op = self.source.read_record_batches(rbr).op()
        return op

    def _put(self, key, value):
        path = self.calc_path(key)
        rbr = value.to_expr().to_pyarrow_batches()
        rbr_to_fs(self.fs, path, rbr)
        return self._get(key)

    def _drop(self, key):
        path = self.calc_path(key)
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
