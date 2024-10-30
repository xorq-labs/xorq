from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow_hotfix  # noqa: F401
from ibis import BaseBackend
from ibis.expr import types as ir, schema as sch
from sqlglot import exp, parse_one

import letsql.backends.let.hotfix  # noqa: F401
from letsql.backends.datafusion import Backend as DataFusionBackend
from letsql.common.collections import SourceDict
from letsql.expr.relations import (
    CachedNode,
    replace_cache_table,
    RemoteTableReplacer,
)


def _get_datafusion_table(con, table_name, database="public"):
    default = con.catalog()
    public = default.database(database)
    return public.table(table_name)


def _get_datafusion_dataframe(con, expr, **kwargs):
    con._register_udfs(expr)
    con._register_in_memory_tables(expr)

    table_expr = expr.as_table()
    raw_sql = con.compile(table_expr, **kwargs)

    return con.con.sql(raw_sql)


class Backend(DataFusionBackend):
    name = "let"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sources = SourceDict()

    def register(
        self,
        source: str | Path | pa.Table | pa.RecordBatch | pa.Dataset | pd.DataFrame,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        # FIXME: make sure all paths set the correct backend, table_or_expr pairs
        table_or_expr = None
        if isinstance(source, ir.Expr) and hasattr(source, "to_pyarrow_batches"):
            table_or_expr = source.op()

            backends, has_unbound = source._find_backends()
            backend = None
            if not backends:
                if not has_unbound:
                    # presumes MemoryTable?
                    source = super().execute(source)
                    table_or_expr = None
            elif len(backends) > 1:
                raise ValueError("Multiple backends found for this expression")
            else:
                backend = backends[0]

            if isinstance(backend, Backend):
                if table_or_expr in backend._sources:
                    old_table_expr, table_or_expr = (
                        table_or_expr,
                        backend._sources.get_table_or_op(table_or_expr),
                    )
                    backend = backend._sources.get_backend(old_table_expr)
                source = table_or_expr.to_expr()

            if (
                isinstance(backend, DataFusionBackend)
                or getattr(backend, "name", "") == DataFusionBackend.name
            ):
                source = _get_datafusion_dataframe(backend, source)

        registered_table = super().register(source, table_name=table_name, **kwargs)
        self._sources[registered_table.op()] = table_or_expr or registered_table.op()

        return registered_table

    def read_parquet(
        self, path: str | Path, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        registered_table = super().read_parquet(path, table_name=table_name, **kwargs)
        self._sources[registered_table.op()] = registered_table.op()
        return registered_table

    def read_csv(
        self, path: str | Path, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        registered_table = super().read_csv(path, table_name=table_name, **kwargs)
        self._sources[registered_table.op()] = registered_table.op()
        return registered_table

    def read_json(
        self, path: str | Path, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        registered_table = super().read_json(path, table_name=table_name, **kwargs)
        self._sources[registered_table.op()] = registered_table.op()
        return registered_table

    def read_delta(
        self, source_table: str | Path, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        registered_table = super().read_delta(
            source_table, table_name=table_name, **kwargs
        )
        self._sources[registered_table.op()] = registered_table.op()
        return registered_table

    def read_postgres(
        self, uri: str, *, table_name: str | None = None, database: str = "public"
    ) -> ir.Table:
        """Register a table from a postgres instance into a DuckDB table.

        Parameters
        ----------
        uri
            A postgres URI of the form `postgres://user:password@host:port`
        table_name
            The table to read
        database
            PostgreSQL database (schema) where `table_name` resides

        Returns
        -------
        ir.Table
            The just-registered table.

        """
        from letsql.backends.postgres import Backend

        backend = Backend()
        backend = backend._from_url(uri, database=database)
        table = backend.table(table_name)
        registered_table = super().register_table_provider(table, table_name=table_name)
        self._sources[registered_table.op()] = table.op()
        return registered_table

    def read_sqlite(
        self, path: str | Path, *, table_name: str | None = None
    ) -> ir.Table:
        """Register a table from a SQLite database into a DuckDB table.

        Parameters
        ----------
        path
            The path to the SQLite database
        table_name
            The table to read

        Returns
        -------
        ir.Table
            The just-registered table.

        Examples
        --------
        >>> import letsql as ls
        >>> import sqlite3
        >>> ls.options.interactive = True
        >>> with sqlite3.connect("/tmp/sqlite.db") as con:
        ...     con.execute("DROP TABLE IF EXISTS t")  # doctest: +ELLIPSIS
        ...     con.execute("CREATE TABLE t (a INT, b TEXT)")  # doctest: +ELLIPSIS
        ...     con.execute(
        ...         "INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c')"
        ...     )  # doctest: +ELLIPSIS
        <...>
        >>> t = ls.read_sqlite(path="/tmp/sqlite.db", table_name="t")
        >>> t
        ┏━━━━━━━┳━━━━━━━━┓
        ┃ a     ┃ b      ┃
        ┡━━━━━━━╇━━━━━━━━┩
        │ int64 │ string │
        ├───────┼────────┤
        │     1 │ a      │
        │     2 │ b      │
        │     3 │ c      │
        └───────┴────────┘

        """
        import letsql as ls

        con = ls.sqlite.connect(path)

        table = con.table(table_name)
        registered_table = super().register_table_provider(table, table_name=table_name)
        self._sources[registered_table.op()] = table.op()
        return registered_table

    def create_table(
        self,
        name: str,
        obj: pd.DataFrame | pa.Table | ir.Table | None = None,
        *,
        schema: sch.Schema | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ):
        registered_table = super().create_table(
            name, obj, schema=schema, database=database, temp=temp, overwrite=overwrite
        )
        self._sources[registered_table.op()] = registered_table.op()
        return registered_table

    def _transform_to_native_backend(self, expr):
        native_backend = self._get_source(expr) is not self
        if native_backend:

            def replace_table(node, _, **_kwargs):
                return self._sources.get_table_or_op(node, node.__recreate__(_kwargs))

            expr = expr.op().replace(replace_table).to_expr()

        return expr

    def execute(self, expr: ir.Expr, **kwargs: Any):
        with self._get_backend_and_expr(expr) as resource:
            backend, expr = resource
            result = backend.execute(expr, **kwargs)
        return result

    def to_pyarrow(self, expr: ir.Expr, **kwargs: Any) -> pa.Table:
        with self._get_backend_and_expr(expr) as resource:
            backend, expr = resource
        return backend.to_pyarrow(expr, **kwargs)

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        chunk_size: int = 1_000_000,
        **kwargs: Any,
    ) -> pa.ipc.RecordBatchReader:
        with self._get_backend_and_expr(expr) as resource:
            backend, expr = resource
        return backend.to_pyarrow_batches(expr, chunk_size=chunk_size, **kwargs)

    @contextmanager
    def _get_backend_and_expr(self, expr):
        expr = self._transform_to_native_backend(expr)
        expr, created = self._register_and_transform_remote_tables(expr)
        expr = self._register_and_transform_cache_tables(expr)
        expr, dt_to_read = self._transform_deferred_reads(expr)
        backend = self._get_source(expr)
        if isinstance(backend, self.__class__):
            backend = super(self.__class__, backend)
        yield backend, expr.unbind()
        for table, con in created.items():
            try:
                con.drop_table(table)
            except Exception:
                con.drop_view(table)

    def do_connect(self, config: Mapping[str, str | Path] | None = None) -> None:
        """Creates a connection.

        Parameters
        ----------
        config
            Mapping of table names to files.

        Examples
        --------
        >>> import letsql as ls
        >>> con = ls.connect()

        """
        super().do_connect(config=config)

    def _get_source(self, expr: ir.Expr):
        """Find the source"""

        tables = expr.op().find(
            lambda n: (s := getattr(n, "source", None)) and isinstance(s, BaseBackend)
        )

        # assumes only one backend per DB type
        sources = []
        for table in tables:
            source = self._sources.get_backend(table, getattr(table, "source", self))
            if all(source is not seen for seen in sources):
                sources.append(source)

        if len(sources) != 1:
            return self
        else:
            return sources[0]

    def _transform_deferred_reads(self, expr, **kwargs):
        dt_to_read = {}

        def replace_read(node, _, **_kwargs):
            from letsql.expr.relations import Read

            if isinstance(node, Read):
                if node.source.name == "pandas":
                    # FIXME: pandas read is not lazy, leave it to the pandas executor to do
                    node = dt_to_read[node] = node.make_dt()
                else:
                    node = dt_to_read[node] = node.make_dt()
            else:
                node = node.__recreate__(_kwargs)
            return node

        expr = expr.op().replace(replace_read).to_expr()
        return expr, dt_to_read

    def _register_and_transform_cache_tables(self, expr):
        """This function will sequentially execute any cache node that is not already cached"""

        def fn(node, _, **kwargs):
            node = node.__recreate__(kwargs)
            if isinstance(node, CachedNode):
                uncached, storage = node.parent, node.storage
                node = storage.set_default(uncached.to_expr(), uncached)
                table = node.to_expr()
                if node.source is self:
                    table = _get_datafusion_table(self.con, node.name)
                node = self.register(table, table_name=node.name).op()
            return node

        op = expr.op()
        out = op.replace(fn)

        return out.to_expr()

    def _to_sqlglot(
        self, expr: ir.Expr, *, limit: str | None = None, params=None, **_: Any
    ):
        op = expr.op()
        out = op.map_clear(replace_cache_table)

        return super()._to_sqlglot(out.to_expr(), limit=limit, params=params)

    def _extract_catalog(self, query):
        tables = parse_one(query).find_all(exp.Table)
        return {table.name: self.table(table.name) for table in tables}

    @staticmethod
    def _register_and_transform_remote_tables(expr):
        replacer = RemoteTableReplacer()
        return expr.op().replace(replacer).to_expr(), replacer.created
