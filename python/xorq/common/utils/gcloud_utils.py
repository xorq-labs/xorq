from __future__ import annotations

from typing import Any

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

from xorq.caching.storage import CacheStorage
from xorq.config import default_backend
from xorq.vendor.ibis.backends import BaseBackend


@curry
def rbr_from_fs(fs: Any, path: str) -> pa.RecordBatchReader:
    def get_schema(fs: Any, path: str) -> pa.Schema:
        with fs.open(path, "rb") as fh:
            pf = pq.ParquetFile(fh)
            schema = pf.schema.to_arrow_schema()
            return schema

    def gen_batches(fs: Any, path: str) -> Any:
        with fs.open(path, "rb") as fh:
            pf = pq.ParquetFile(fh)
            yield from pf.iter_batches()

    rbr = pa.RecordBatchReader.from_batches(
        get_schema(fs, path),
        gen_batches(fs, path),
    )
    return rbr


@curry
def rbr_to_fs(
    fs: Any,
    path: str,
    rbr: pa.RecordBatchReader,
    parquet_metadata: dict | None = None,
    **kwargs: Any,
) -> None:
    schema = rbr.schema
    if parquet_metadata is not None:
        from xorq.common.utils.provenance_utils import (  # noqa: PLC0415
            inject_metadata_into_schema,
        )

        schema = inject_metadata_into_schema(schema, parquet_metadata)
    with fs.open(path, "wb") as fh:
        with pq.ParquetWriter(fh, schema, **kwargs) as writer:
            for batch in rbr:
                writer.write_batch(batch)


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

    def __dask_tokenize__(self):
        from xorq.common.utils.dask_normalize.dask_normalize_utils import (  # noqa: PLC0415
            normalize_seq_with_caller,
        )

        return normalize_seq_with_caller(
            self.source, self.bucket_name, caller="normalize_gc_storage"
        )

    def get_path(self, key: str) -> str:
        path = f"{self.bucket_name}/{key}.parquet"
        return path

    def exists(self, key: str) -> bool:
        path = self.get_path(key)
        return self.fs.exists(path)

    def get(self, key: str) -> Any:
        path = self.get_path(key)
        rbr = rbr_from_fs(self.fs, path)
        op = self.source.read_record_batches(rbr).op()
        return op

    def put(self, key: str, value: Any, parquet_metadata: dict | None = None) -> Any:
        path = self.get_path(key)
        rbr = value.to_expr().to_pyarrow_batches()
        rbr_to_fs(self.fs, path, rbr, parquet_metadata=parquet_metadata)
        return self.get(key)

    def drop(self, key: str) -> None:
        path = self.get_path(key)
        self.fs.delete(path)


def get_file_metadata(uri: str, client: Any = None) -> tuple[tuple[str, Any], ...]:
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
