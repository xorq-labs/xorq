from __future__ import annotations

import abc
import functools
from pathlib import Path
from typing import Any, Mapping

import pyarrow_hotfix  # noqa: F401
from datafusion import SessionContext
from ibis import BaseBackend
from letsql.backends.datafusion import Backend as DataFusionBackend
from ibis.common.exceptions import IbisError
from ibis.expr import types as ir
from ibis.util import gen_name

from letsql.common.caching import (
    ParquetCacheStorage,
)
from letsql.expr.relations import contract_cache_table, DeferredCacheTable

from operator import contains


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
    connections = {}
    sources = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_storage = ParquetCacheStorage(
            populate=self._load_into_cache,
            recreate=self._recreate_cache,
            generate_name=functools.partial(gen_name, "cache"),
        )

    def add_connection(self, connection: BaseBackend, name: str | None = None) -> None:
        self.connections[connection.name] = connection

    def drop_connection(self, name: str) -> None:
        self.connections.pop(name)

    def list_connections(self, like: str | None = None) -> list[BaseBackend]:
        return list(self.connections.values())

    def execute(self, expr: ir.Expr, **kwargs: Any):
        expr = self._register_and_transform_cache_tables(expr)
        name = self._get_source_name(expr)

        if name == "datafusion":
            return super().execute(expr, **kwargs)

        backend = self.connections[name]

        return backend.execute(expr, **kwargs)

    def do_connect(
        self, config: Mapping[str, str | Path] | SessionContext | None = None
    ) -> None:
        super().do_connect(config=config)
        self.cache_storage.try_recreate()  # try to recreate the cache

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

        source = self.sources.get(origin, self)
        return source.name

    def _cached(self, expr: ir.Table):
        op = DeferredCacheTable(
            query="", schema=expr.schema(), source=self, parent=expr.op()
        )
        self.cache_storage.name(op)
        return op.to_expr()

    def _register_and_transform_cache_tables(self, expr):
        """This function will sequentially execute any cache node that is not already cached"""

        is_cached = functools.partial(contains, self.cache_storage)

        def fn(node, _, **kwargs):
            if is_cached(node):
                name = self.cache_storage[node]
                if name not in self.list_tables():
                    self.cache_storage.store(node)
                node = node.replace({node: ir.CachedTable(self.table(name).op()).op()})
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
        results = op.map(contract_cache_table)
        out = results[op]

        return super()._to_sqlglot(out.to_expr(), limit=limit, params=params)

    def _load_into_cache(self, name, op) -> ir.Table:
        results = op.parent.map(contract_cache_table)
        out = results[op.parent]

        expr = out.to_expr()
        source_name = self._get_source_name(expr)
        backend = self.connections[source_name]
        data = backend.to_pyarrow(expr)
        temp_table = super().create_table(name, data, temp=True)
        return temp_table

    def _recreate_cache(self, name, data):
        super().create_table(name, data, temp=True)
