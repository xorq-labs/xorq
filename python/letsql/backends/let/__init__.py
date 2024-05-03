from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Mapping

import dask
import pandas as pd
import pyarrow as pa
import pyarrow_hotfix  # noqa: F401
from letsql.internal import SessionContext
from ibis import BaseBackend
from letsql.backends.datafusion import Backend as DataFusionBackend
from ibis.common.exceptions import IbisError
from ibis.expr import types as ir
import ibis.expr.operations as ops

import letsql.common.utils.dask_normalize  # noqa: F401
from letsql.common.caching import (
    SourceStorage,
)
from letsql.expr.relations import (
    replace_cache_table,
    CachedNode,
    replace_source_factory,
)

KEY_PREFIX = "letsql_cache-"


class CanListConnections(abc.ABC):
    @abc.abstractmethod
    def list_connections(
        self,
        like: str | None = None,
    ) -> list[BaseBackend]:
        pass


class CanCreateConnections(CanListConnections):
    @abc.abstractmethod
    def add_connection(
        self,
        connection: BaseBackend,
        name: str | None = None,
    ) -> None:
        """Add a connection named `name`."""

    @abc.abstractmethod
    def drop_connection(
        self,
        name: str,
    ) -> None:
        """Drop the connection with `name`."""


class Backend(DataFusionBackend, CanCreateConnections):
    name = "let"
    connections = {}
    sources = {}

    def add_connection(self, connection: BaseBackend, name: str | None = None) -> None:
        self.connections[connection.name] = connection

    def drop_connection(self, name: str) -> None:
        self.connections.pop(name)

    def list_connections(self, like: str | None = None) -> list[BaseBackend]:
        return list(self.connections.values()) + [self]

    def register(
        self,
        source: str | Path | pa.Table | pa.RecordBatch | pa.Dataset | pd.DataFrame,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        if isinstance(source, ir.Table) and hasattr(source, "to_pyarrow_batches"):
            table = source.op()
            backend = table.source
            if backend.name == "let":
                if (original := backend.sources.get(table, None)) != backend:
                    if original is not None:
                        conn = backend.connections[original.name]
                        source = conn.table(table.name)
                else:
                    # TODO make a better fix to make the table durable (like a datafusion view)
                    source = original.table(table.name).to_pyarrow_batches()

        return super().register(source, table_name=table_name, **kwargs)

    def execute(self, expr: ir.Expr, **kwargs: Any):
        expr = self._register_and_transform_cache_tables(expr)
        names_to_drop = self._register_remote_sources(expr)
        name = self._get_source_name(expr)

        backend = super() if name == "let" else self.connections[name]
        res = backend.execute(expr, **kwargs)

        for table_name in names_to_drop:
            super().drop_table(table_name)

        return res

    def do_connect(
        self, config: Mapping[str, str | Path] | SessionContext | None = None
    ) -> None:
        super().do_connect(config=config)

    def table(
        self,
        name: str,
        schema: str | None = None,
        database: tuple[str, str] | str | None = None,
    ) -> ir.Table:
        backends = list(self.connections.values())
        backends.append(super())

        for backend in backends:
            try:
                t = backend.table(name, schema=schema)
                original = t.op().source
                override = t.op().copy(source=self)
                self.sources[override] = original
                return override.to_expr()
            except IbisError:
                continue
        else:
            if self.connections:
                raise IbisError(f"Table not found: {name!r}")

    def list_tables(
        self,
        like: str | None = None,
        database: str | None = None,
    ) -> list[str]:
        backends = list(self.connections.values())
        backends.append(super())

        return [t for backend in backends for t in backend.list_tables(like=like)]

    def _get_source_name(self, expr: ir.Expr):
        origin = expr.op()
        while hasattr(origin, "parent"):
            origin = getattr(origin, "parent")

        if hasattr(origin, "table"):
            origin = origin.table

        source = self.sources.get(origin, getattr(origin, "source", self))
        return source.name

    def _cached(self, expr: ir.Table, storage=None):
        name = self._get_source_name(expr)
        if name == "let":
            source = self
        else:
            source = self.connections[name]
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

    def _load_into_cache(self, name, op) -> ir.Table:
        out = op.map_clear(replace_cache_table)
        expr = out.to_expr()
        source_name = self._get_source_name(expr)
        backend = self.connections[source_name]
        data = backend.to_pyarrow(expr)
        temp_table = super().create_table(name, data, temp=True)
        return temp_table

    def _recreate_cache(self, name, data):
        super().create_table(name, data, temp=True)

    def _register_remote_sources(self, expr):
        sources = set(
            self.sources.get(table, None) for table in expr.op().find(ops.DatabaseTable)
        )
        names = []
        if len(sources) > 1:
            for table in expr.op().find(ops.DatabaseTable):
                if (source := self.sources.get(table, None)) != self:
                    if source is not None:
                        conn = self.connections[source.name]

                        # register a new to_pyarrow_batches table
                        super().register(conn.table(table.name), table.name)

                        # store the name to do a cleanup after the end of execution
                        names.append(table.name)
        return names


def letsql_cache(self, storage=None):
    current_backend = self._find_backend(use_default=True)
    return current_backend._cached(self, storage=storage)


def do_monkeypatch_Table_cache():
    setattr(letsql_cache, "orig_cache", ir.Table.cache)
    setattr(ir.Table, "cache", letsql_cache)


do_monkeypatch_Table_cache()
