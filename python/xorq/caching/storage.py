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
    def put(self, key, value):
        pass

    @abstractmethod
    def drop(self, key):
        pass


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

    def put(self, key, value):
        path = self.get_path(key)
        # move from temp location upon success to prevent empty files on failure
        tmp_path = path.with_name(path.name + ".tmp")
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

    def exists(self, key):
        return key in self.source.tables

    def get(self, key):
        return self.source.table(key).op()

    def put(self, key, value):
        def is_remote(value):
            name = value.to_expr()._find_backend().name
            # FIXME: add pyiceberg, trino
            return name in ("postgres", "snowflake")

        def is_single_backend(storage, value):
            from xorq.common.utils.graph_utils import find_all_sources

            return (storage.source,) == find_all_sources(value.to_expr())

        if is_remote(value):
            if is_single_backend(self, value):
                from xorq.expr.api import _transform_expr

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
