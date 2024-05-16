from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import dask
import pandas as pd
import pyarrow as pa
import pyarrow_hotfix  # noqa: F401
from ibis import BaseBackend
from ibis.expr import types as ir
from ibis.expr.schema import SchemaLike
from sqlglot import exp, parse_one

import letsql.common.utils.dask_normalize  # noqa: F401
from letsql.backends.datafusion import Backend as DataFusionBackend
from letsql.common.caching import (
    SourceStorage,
)
from letsql.expr.relations import (
    CachedNode,
    replace_cache_table,
    replace_source_factory,
)
from letsql.expr.translate import sql_to_ibis
from letsql.internal import SessionContext

KEY_PREFIX = "letsql_cache-"


class Backend(DataFusionBackend):
    name = "let"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sources = {}

    def register(
        self,
        source: str | Path | pa.Table | pa.RecordBatch | pa.Dataset | pd.DataFrame,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        backend = self
        if isinstance(source, ir.Table) and hasattr(source, "to_pyarrow_batches"):
            table = source.op()
            backend = table.source

            if backend == self:
                original = backend.sources.get(table, self)
                source = original.table(table.name)

        registered_table = super().register(source, table_name=table_name, **kwargs)
        self.sources[registered_table.op()] = backend

        return registered_table

    def execute(self, expr: ir.Expr, **kwargs: Any):
        expr = self._register_and_transform_cache_tables(expr)
        backend = self._get_source(expr)

        res = (
            super().execute(expr, **kwargs)
            if backend == self
            else backend.execute(expr, **kwargs)
        )

        return res

    def do_connect(
        self, config: Mapping[str, str | Path] | SessionContext | None = None
    ) -> None:
        super().do_connect(config=config)

    def _get_source(self, expr: ir.Expr):
        """Find the source"""

        tables = expr.op().find(
            lambda n: (s := getattr(n, "source", None)) and isinstance(s, BaseBackend)
        )

        # assumes only one backend per DB type
        sources = set(
            self.sources.get(table, getattr(table, "source", self)) for table in tables
        )

        if len(sources) != 1:
            return self
        else:
            return sources.pop()

    def _cached(self, expr: ir.Table, storage=None):
        source = self._get_source(expr)
        storage = storage or SourceStorage(source)
        op = CachedNode(
            schema=expr.schema(),
            parent=expr.op(),
            source=source,
            storage=storage,
        )
        return op.to_expr()

    def _register_and_transform_cache_tables(self, expr):
        """This function will sequentially execute any cache node that is not already cached"""

        def fn(node, _, **kwargs):
            if isinstance(node, CachedNode):
                replace_source = replace_source_factory(node.source)
                uncached = node.map_clear(replace_cache_table)
                # datafusion+ParquetStorage requires key have .parquet suffix: maybe push suffix append into ParquetStorage?
                key = KEY_PREFIX + dask.base.tokenize(uncached.to_expr())
                storage = kwargs["storage"]
                if not storage.exists(key):
                    value = uncached
                    storage.put(key, value.replace(replace_source))
                node = storage.get(key)
                table = node.to_expr()
                if node.source != self:
                    self.register(table, table_name=key)
            if hasattr(node, "parent"):
                parent = kwargs.get("parent", node.parent)
                node = node.replace({node.parent: parent})

            return node

        op = expr.op()
        results = op.map(fn)
        out = results[op]

        return out.to_expr()

    def _to_sqlglot(
        self, expr: ir.Expr, *, limit: str | None = None, params=None, **_: Any
    ):
        op = expr.op()
        out = op.map_clear(replace_cache_table)

        return super()._to_sqlglot(out.to_expr(), limit=limit, params=params)

    def sql(
        self,
        query: str,
        schema: SchemaLike | None = None,
        dialect: str | None = None,
    ) -> ir.Table:
        query = self._transpile_sql(query, dialect=self.compiler.dialect)
        catalog = self._extract_catalog(query)
        return sql_to_ibis(query, catalog).as_table()

    def _extract_catalog(self, query):
        tables = parse_one(query).find_all(exp.Table)
        return {table.name: self.table(table.name) for table in tables}


def letsql_cache(self, storage=None):
    current_backend = self._find_backend(use_default=True)
    return current_backend._cached(self, storage=storage)


def do_monkeypatch_Table_cache():
    setattr(letsql_cache, "orig_cache", ir.Table.cache)
    setattr(ir.Table, "cache", letsql_cache)


do_monkeypatch_Table_cache()
