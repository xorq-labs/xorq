import toolz

import dask
import ibis.expr.operations as ops
import ibis.expr.types.core
import ibis.expr.types.relations
from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
)
from ibis.common.exceptions import (
    IbisError,
)

import letsql
from letsql.common.caching import (
    SourceStorage,
)
from letsql.common.utils.hotfix_utils import (
    hotfix,
    none_tokenized,
)
from letsql.expr.operations.images import SegmentAnything, Rotate90
from letsql.expr.relations import (
    CachedNode,
    replace_cache_table,
)


@frozen
class LETSQLAccessor:
    expr = field(validator=instance_of(ibis.expr.types.core.Expr))
    node_types = (ops.DatabaseTable, ops.SQLQueryResult)

    @property
    def op(self):
        return self.expr.op()

    @property
    def cached_nodes(self):
        return self.op.find(CachedNode)

    @property
    def storage(self):
        if self.is_cached:
            return self.op.storage
        else:
            return None

    @property
    def storages(self):
        return tuple(node.storage for node in self.cached_nodes)

    @property
    def backends(self):
        (_backends, _) = self.expr._find_backends()
        return tuple(set(_backends))

    @property
    def is_multiengine(self):
        (_, *rest) = set(self.backends)
        return bool(rest)

    @property
    def dts(self):
        return self.op.find(self.node_types)

    @property
    def is_cached(self):
        return isinstance(self.op, CachedNode)

    @property
    def has_cached(self):
        return bool(tuple(self.op.find_topmost(CachedNode)))

    @property
    def uncached(self):
        if self.has_cached:
            return self.op.map_clear(replace_cache_table).to_expr()
        else:
            return self.expr

    @property
    def uncached_one(self):
        if self.is_cached:
            return self.op.parent.to_expr()
        else:
            return self.expr

    def get_key(self):
        if self.is_cached and (self.exists() or not self.uncached_one.ls.has_cached):
            return self.storage.get_key(
                self.storage.source._register_and_transform_cache_tables(
                    self.uncached_one
                )
            )
        else:
            return None

    def get_keys(self):
        if self.has_cached and self.cached_nodes[0].to_expr().ls.exists():
            # FIXME: yield storage with key
            return tuple(op.to_expr().ls.get_key() for op in self.cached_nodes)
        else:
            return None

    def exists(self):
        if self.is_cached:
            # must iterate from the bottom up else we execute downstream cached tables
            return all(
                cn.storage.exists(
                    cn.storage.source._register_and_transform_cache_tables(
                        cn.parent.to_expr()
                    )
                )
                for cn in self.cached_nodes[::-1]
            )
        else:
            return None


@hotfix(
    ibis.expr.types.core.Expr,
    "_find_backend",
    "9e19dfcd3404a043987ff26dce7a40ad",
)
def _letsql_find_backend(self, *, use_default=True):
    # FIXME: push this into LETSQLAccessor
    try:
        current_backend = self._find_backend._original(self, use_default=use_default)
    except IbisError as e:
        if "Multiple backends found" in e.args[0]:
            current_backend = letsql.options.backend
        else:
            raise e
    return current_backend


@hotfix(
    ibis.expr.types.relations.Table,
    "cache",
    "654b574765abdd475264851b89112881",
)
def letsql_cache(self, storage=None):
    # FIXME: push this into LETSQLAccessor
    try:
        current_backend = self._find_backend(use_default=True)
    except IbisError as e:
        if "Multiple backends found" in e.args[0]:
            current_backend = letsql.options.backend
        else:
            raise e
    storage = storage or SourceStorage(source=current_backend)
    op = CachedNode(
        schema=self.schema(),
        parent=self.op(),
        source=current_backend,
        storage=storage,
    )
    return op.to_expr()


@hotfix(
    ibis.expr.types.core.Expr,
    "ls",
    none_tokenized,
)
@property
def ls(self):
    return LETSQLAccessor(self)


@hotfix(ibis.expr.types.binary.BinaryColumn, "segment_anything", none_tokenized)
def segment_anything(
    self: ibis.expr.types.binary.BinaryColumn, model_name: str, seed: list
):
    return SegmentAnything(arg=self, model_name=model_name, seed=seed).to_expr()


@hotfix(ibis.expr.types.binary.BinaryColumn, "rotate90", none_tokenized)
def rotate90(self: ibis.expr.types.binary.BinaryColumn):
    return Rotate90(arg=self).to_expr()


@toolz.curry
def letsql_invoke(_methodname, self, *args, **kwargs):
    con = letsql.connect()
    for dt in self.op().find(ops.DatabaseTable):
        # fixme: use temp names to avoid collisions, remove / deregister after done
        if dt not in con._sources.sources:
            con.register(dt.to_expr(), dt.name)
    method = getattr(con, f"{_methodname}")
    return method(self, *args, **kwargs)


for typ, methodnames in (
    (
        ibis.expr.types.core.Expr,
        ("execute", "to_pyarrow", "to_pyarrow_batches"),
    ),
    (
        # Join.execute is the only case outside Expr.execute
        ibis.expr.types.joins.Join,
        ("execute",),
    ),
):
    for methodname in methodnames:
        hotfix(
            typ,
            methodname,
            dask.base.tokenize(getattr(typ, methodname)),
            letsql_invoke(methodname),
        )
