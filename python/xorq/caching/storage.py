from __future__ import annotations

import datetime
from abc import abstractmethod
from pathlib import Path

from attr import field, frozen
from attr.validators import instance_of, optional


def _lazy_backend_validator(instance, attribute, value):
    from xorq.vendor.ibis.backends import BaseBackend  # noqa: PLC0415

    if not isinstance(value, BaseBackend):
        raise TypeError(
            f"'{attribute.name}' must be a BaseBackend "
            f"(got {value!r} that is a {type(value)!r})."
        )


def _lazy_default_backend():
    from xorq.config import default_backend  # noqa: PLC0415

    return default_backend()


def _lazy_default_relative_path():
    from xorq.config import options  # noqa: PLC0415

    return options.get("cache.default_relative_path")


def _convert_optional_path(value):
    return None if value is None else Path(value)


def resolve_parquet_cache_dir(
    relative_path: Path | str,
    base_path: Path | None = None,
) -> Path:
    """Return the directory that holds parquet cache files for the given storage params."""
    if base_path is None:
        from xorq.common.utils.caching_utils import get_xorq_cache_dir  # noqa: PLC0415

        base_path = get_xorq_cache_dir()
    return base_path / relative_path


def resolve_parquet_cache_path(
    relative_path: Path | str,
    key: str,
    base_path: Path | None = None,
) -> Path:
    """Return the full path of the parquet file for *key* under the given storage params."""
    return resolve_parquet_cache_dir(relative_path, base_path) / (key + ".parquet")


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


def _write_parquet(path, batch_reader, parquet_metadata=None):
    import pyarrow.parquet as pq  # noqa: PLC0415

    schema = batch_reader.schema
    if parquet_metadata is not None:
        from xorq.common.utils.provenance_utils import (  # noqa: PLC0415
            inject_metadata_into_schema,
        )

        schema = inject_metadata_into_schema(schema, parquet_metadata)
    with pq.ParquetWriter(str(path), schema) as writer:
        for batch in batch_reader:
            writer.write_batch(batch)


@frozen
class ParquetStorage(CacheStorage):
    source = field(
        validator=_lazy_backend_validator,
        factory=_lazy_default_backend,
    )
    relative_path = field(
        validator=instance_of(Path),
        factory=_lazy_default_relative_path,
        converter=Path,
    )
    base_path = field(
        validator=optional(instance_of(Path)),
        default=None,
        converter=_convert_optional_path,
    )

    def __dasher_tokenize__(self):
        return (
            "normalize_parquet_storage",
            self.source,
            self.relative_path,
            self.base_path,
        )

    def __attrs_post_init__(self):
        self.path.mkdir(exist_ok=True, parents=True)

    @property
    def path(self):
        return resolve_parquet_cache_dir(self.relative_path, self.base_path)

    def get_path(self, key):
        return resolve_parquet_cache_path(self.relative_path, key, self.base_path)

    def exists(self, key):
        return self.get_path(key).exists()

    def get(self, key):
        from xorq.common.utils.defer_utils import deferred_read_parquet  # noqa: PLC0415

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
        with value.to_expr().to_pyarrow_batches() as batch_reader:
            _write_parquet(tmp_path, batch_reader, parquet_metadata=parquet_metadata)
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

    def __dasher_tokenize__(self):
        return (
            "normalize_parquet_ttl_storage",
            self.source,
            self.relative_path,
            self.base_path,
            self.ttl,
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
class ParquetDummyStorage(ParquetStorage):
    def __attrs_post_init__(self):
        pass  # intentionally skips self.path.mkdir(…) from ParquetStorage


@frozen
class SourceStorage(CacheStorage):
    source = field(
        validator=_lazy_backend_validator,
        factory=_lazy_default_backend,
    )

    def __dasher_tokenize__(self):
        return ("normalize_source_storage", self.source)

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
                (transformed, _, _) = _transform_expr(value.to_expr())
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
