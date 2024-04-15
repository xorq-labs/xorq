import pathlib
import warnings
from collections import defaultdict
from typing import Any, Callable

import ibis.expr.types as ir
import pyarrow.parquet as pq

from cloudpickle import (
    dump as _dump,
    load as _load,
)


def dump(obj, path):
    with path.open("wb") as fh:
        _dump(obj, fh)


def load(path):
    with path.open("rb") as fh:
        return _load(fh)


class ParquetCacheStorage:
    cache_dir_name = ".letsql_cache"

    def __init__(
        self,
        *,
        populate: Callable[[str, Any], ir.Table],
        recreate: Callable[[str, Any], None],
        generate_name: Callable[[], str],
        key: Callable[[Any], Any] = None,
    ) -> None:
        self.populate = populate
        self.recreate = recreate
        # Somehow mypy needs a type hint here
        self.names: defaultdict = defaultdict(generate_name)
        self.key = key or (lambda x: x)
        self._set_cache_dir()

    def _set_cache_dir(self):
        cwd = pathlib.Path().absolute()
        cache_dir = pathlib.Path(__file__).parent.joinpath(self.cache_dir_name)
        if not cache_dir.exists():
            cache_dir = next(
                p for p in (cwd, *cwd.parents) if p.joinpath(".git").exists()
            ).joinpath(self.cache_dir_name)
            self.keys_cache_dir = cache_dir.joinpath("keys")
            self.values_cache_dir = cache_dir.joinpath("values")
            self.keys_cache_dir.mkdir(exist_ok=True, parents=True)
            self.values_cache_dir.mkdir(exist_ok=True, parents=True)

    def try_recreate(self):
        keys = sorted([k for k in self.keys_cache_dir.glob("*.key")])
        values = sorted([v for v in self.values_cache_dir.glob("*.parquet")])

        if len(keys) != len(values) or any(
            k.stem != v.stem for k, v in zip(keys, values)
        ):
            warnings.warn("Illegal state of cache")
        else:
            for expr_file, parquet_data in zip(keys, values):
                self.names[
                    self.key(load(expr_file))
                ] = expr_file.stem  # the filename without the extension
                self.recreate(expr_file.stem, pq.read_table(parquet_data))

    def clear(self):
        for k in self.keys_cache_dir.glob("*.key"):
            k.unlink(missing_ok=True)

        for v in self.values_cache_dir.glob("*.parquet"):
            v.unlink(missing_ok=True)

    def name(self, value):
        name = self.names[self.key(value)]
        return name

    def store(self, value):
        name = self.names[self.key(value)]
        temp_table = self.populate(name, value)
        dump(value, self.keys_cache_dir.joinpath(f"{name}.key"))
        temp_table.to_parquet(self.values_cache_dir.joinpath(f"{name}.parquet"))

    def __contains__(self, key):
        return key in self.names

    def __getitem__(self, item):
        return self.names[item]
