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

from xorq.caching.storage import Storage
from xorq.config import _backend_init
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
def rbr_to_fs(fs, path, rbr, **kwargs):
    with fs.open(path, "wb") as fh:
        with pq.ParquetWriter(fh, rbr.schema, **kwargs) as writer:
            for batch in rbr:
                writer.write_batch(batch)


@frozen
class GCStorage(Storage):
    bucket_name: str = field(validator=instance_of(str))
    source = field(
        validator=instance_of(BaseBackend),
        factory=_backend_init,
    )
    fs: gcsfs.GCSFileSystem = field(init=False)

    def __attrs_post_init__(self):
        assert hasattr(self.source, "read_record_batches")
        object.__setattr__(self, "fs", gcsfs.GCSFileSystem())

    def get_path(self, key):
        path = f"{self.bucket_name}/{key}.parquet"
        return path

    def exists(self, key):
        path = self.get_path(key)
        return self.fs.exists(path)

    def get(self, key):
        path = self.get_path(key)
        rbr = rbr_from_fs(self.fs, path)
        op = self.source.read_record_batches(rbr).op()
        return op

    def put(self, key, value):
        path = self.get_path(key)
        rbr = value.to_expr().to_pyarrow_batches()
        rbr_to_fs(self.fs, path, rbr)
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
