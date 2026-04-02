from __future__ import annotations

import datetime
import functools
from abc import (
    abstractmethod,
)
from pathlib import (
    Path,
)

from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
    optional,
)

from xorq.common.utils.caching_utils import (
    get_xorq_cache_dir,
)
from xorq.common.utils.defer_utils import (
    deferred_read_parquet,
)
from xorq.common.utils.func_utils import (
    if_not_none,
)
from xorq.config import _backend_init, options
from xorq.vendor import ibis


@frozen
class CacheStorage:
    @abstractmethod
    def exists(self, key):
        pass

    @abstractmethod
    def get(self, key):
        pass

    @abstractmethod
    def put(self, key, value, parquet_metadata=None):
        pass

    @abstractmethod
    def drop(self, key):
        pass


def _write_parquet_with_metadata(expr, path, metadata_dict):
    import pyarrow.parquet as pq  # noqa: PLC0415

    from xorq.caching.provenance import inject_metadata_into_schema  # noqa: PLC0415
    from xorq.expr.api import to_pyarrow_batches  # noqa: PLC0415

    with to_pyarrow_batches(expr) as batch_reader:
        schema = inject_metadata_into_schema(batch_reader.schema, metadata_dict)
        with pq.ParquetWriter(path, schema) as writer:
            for batch in batch_reader:
                writer.write_batch(batch)


@frozen
class ParquetStorage(CacheStorage):
    source = field(
        validator=instance_of(ibis.backends.BaseBackend),
        factory=_backend_init,
    )
    relative_path = field(
        validator=instance_of(Path),
        factory=functools.partial(options.get, "cache.default_relative_path"),
        converter=Path,
    )
    base_path = field(
        validator=optional(instance_of(Path)),
        default=None,
        converter=if_not_none(Path),
    )

    def __dask_tokenize__(self):
        from xorq.common.utils.dask_normalize.dask_normalize_utils import (  # noqa: PLC0415
            normalize_seq_with_caller,
        )

        return normalize_seq_with_caller(
            self.source,
            self.relative_path,
            self.base_path,
            caller="normalize_parquet_storage",
        )

    def __attrs_post_init__(self):
        self.path.mkdir(exist_ok=True, parents=True)

    @property
    def path(self):
        return (self.base_path or get_xorq_cache_dir()).joinpath(self.relative_path)

    def get_path(self, key):
        return self.path.joinpath(key + ".parquet")

    def exists(self, key):
        return self.get_path(key).exists()

    def get(self, key):
        op = deferred_read_parquet(
            path=self.get_path(key),
            con=self.source,
            table_name=key,
        ).op()
        return op

    def put(self, key, value, parquet_metadata=None):
        path = self.get_path(key)
        # move from temp location upon success to prevent empty files on failure
        tmp_path = path.with_name(path.name + ".tmp")
        if parquet_metadata is not None:
            _write_parquet_with_metadata(value.to_expr(), tmp_path, parquet_metadata)
        else:
            value.to_expr().to_parquet(tmp_path)
        tmp_path.rename(path)
        return self.get(key)

    def drop(self, key):
        path = self.get_path(key)
        path.unlink()


@frozen
class ParquetTTLStorage(ParquetStorage):
    ttl = field(
        validator=instance_of(datetime.timedelta), default=datetime.timedelta(days=1)
    )

    def __dask_tokenize__(self):
        from xorq.common.utils.dask_normalize.dask_normalize_utils import (  # noqa: PLC0415
            normalize_seq_with_caller,
        )

        return normalize_seq_with_caller(
            self.source,
            self.relative_path,
            self.base_path,
            self.ttl,
            caller="normalize_parquet_ttl_storage",
        )

    def exists(self, key):
        path = self.get_path(key)
        return path.exists() and self.satisfies_ttl(path)

    def satisfies_ttl(self, path):
        delta = datetime.datetime.now() - datetime.datetime.fromtimestamp(
            path.stat().st_mtime
        )
        return delta < self.ttl


@frozen
class SourceStorage(CacheStorage):
    source = field(
        validator=instance_of(ibis.backends.BaseBackend),
        factory=_backend_init,
    )

    def __dask_tokenize__(self):
        from xorq.common.utils.dask_normalize.dask_normalize_utils import (  # noqa: PLC0415
            normalize_seq_with_caller,
        )

        return normalize_seq_with_caller(self.source, caller="normalize_source_storage")

    def exists(self, key):
        return key in self.source.tables

    def get(self, key):
        return self.source.table(key).op()

    def put(self, key, value, parquet_metadata=None):
        def is_remote(value):
            name = value.to_expr()._find_backend().name
            # FIXME: add pyiceberg, trino
            return name in ("postgres", "snowflake")

        def is_single_backend(storage, value):
            from xorq.common.utils.graph_utils import find_all_sources  # noqa: PLC0415

            return (storage.source,) == find_all_sources(value.to_expr())

        if is_remote(value):
            if is_single_backend(self, value):
                from xorq.expr.api import _transform_expr  # noqa: PLC0415

                # must transform for Read ops: create_table expects a vanilla ibis expr
                (transformed, _) = _transform_expr(value.to_expr())
                self.source.create_table(key, transformed)
            else:
                assert hasattr(self.source, "read_record_batches")
                # read_record_batches will create durable table in out-of-core fashion
                # works for snowflake and postgres
                self.source.read_record_batches(
                    value.to_expr().to_pyarrow_batches(),
                    key,
                )
        else:
            self.source.create_table(key, value.to_expr().to_pyarrow())
        return self.get(key)

    def drop(self, key):
        self.source.drop_table(key)
