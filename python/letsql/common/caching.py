import operator
import pathlib
from abc import (
    ABC,
    abstractmethod,
)

import ibis
import toolz

from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
)
from cloudpickle import (
    dump as _dump,
    load as _load,
)


abs_path_converter = toolz.compose(operator.methodcaller("absolute"), pathlib.Path)


def dump(obj, path):
    with path.open("wb") as fh:
        _dump(obj, fh)


def load(path):
    with path.open("rb") as fh:
        return _load(fh)


class CacheStorage(ABC):
    @abstractmethod
    def exists(self, key):
        pass

    def get(self, key):
        if not self.exists(key):
            raise KeyError
        else:
            return self._get(key)

    @abstractmethod
    def _get(self, key):
        pass

    def put(self, key, value):
        if self.exists(key):
            raise ValueError
        else:
            return self._put(key, value)

    @abstractmethod
    def _put(self, key, value):
        pass

    def drop(self, key):
        if not self.exist(key):
            raise KeyError
        else:
            return self._drop(key)

    @abstractmethod
    def _drop(self, key):
        pass


@frozen
class ParquetCacheStorage(CacheStorage):
    path = field(validator=instance_of(pathlib.Path), converter=abs_path_converter)
    source = field(validator=instance_of(ibis.backends.BaseBackend))

    def __attrs_post_init__(self):
        self.path.parent.mkdir(exist_ok=True, parents=True)

    def get_loc(self, key):
        return self.path.joinpath(key + ".parquet")

    def exists(self, key):
        return self.get_loc(key).exists()

    def _get(self, key):
        op = self.source.read_parquet(self.get_loc(key), key).op()
        return op

    def _put(self, key, value):
        loc = self.get_loc(key)
        value.to_expr().to_parquet(loc)
        return self.get(key)

    def _drop(self, key):
        path = self.get_loc(key)
        path.unlink()
        # FIXME: what to do if table is not registered?
        self.source.drop_table(key)


@frozen
class SourceStorage(CacheStorage):
    source = field(validator=instance_of(ibis.backends.BaseBackend))

    def exists(self, key):
        return key in self.source.tables

    def _get(self, key):
        return self.source.table(key).op()

    def _put(self, key, value):
        self.source.create_table(key, value.to_expr())
        return self.get(key)

    def _drop(self, key):
        self.source.drop_table(key)
