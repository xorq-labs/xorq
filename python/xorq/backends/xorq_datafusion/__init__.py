from __future__ import annotations

import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow_hotfix  # noqa: F401
from sqlglot import exp, parse_one

from xorq.backends.xorq_datafusion.datafusion import Backend as DataFusionBackend
from xorq.internal import SessionConfig, WindowUDF
from xorq.vendor.ibis.expr import types as ir


if TYPE_CHECKING:
    import pandas as pd


class Backend(DataFusionBackend):
    name = "xorq_datafusion"

    def register(
        self,
        source: str | Path | pa.Table | pa.RecordBatch | pa.Dataset | pd.DataFrame,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        if isinstance(source, ir.Expr):
            from xorq.expr.relations import into_backend  # noqa: PLC0415

            return into_backend(source, self, table_name)
        return super().register(source, table_name=table_name, **kwargs)

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
        from xorq.backends.postgres import Backend  # noqa: PLC0415

        backend = Backend()
        parsed = urllib.parse.urlparse(uri)
        backend = backend._from_url(parsed, database=database)
        table = backend.table(table_name)
        return super().register_table_provider(table, table_name=table_name)

    def execute(self, expr: ir.Expr, **kwargs: Any):
        batch_reader = self.to_pyarrow_batches(expr, **kwargs)
        return expr.__pandas_result__(
            batch_reader.read_pandas(timestamp_as_object=True)
        )

    def to_pyarrow(self, expr: ir.Expr, **kwargs: Any) -> pa.Table:
        batch_reader = self.to_pyarrow_batches(expr, **kwargs)
        arrow_table = batch_reader.read_all()
        return expr.__pyarrow_result__(arrow_table)

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        chunk_size: int = 1_000_000,
        **kwargs: Any,
    ) -> pa.ipc.RecordBatchReader:
        return super().to_pyarrow_batches(expr, chunk_size=chunk_size, **kwargs)

    def do_connect(self, config: SessionConfig | None = None) -> None:
        """Creates a connection.

        Parameters
        ----------
        config
            Mapping of table names to files.

        Examples
        --------
        >>> import xorq.api as xo
        >>> con = xo.connect()

        """
        super().do_connect(config=config)

    def _extract_catalog(self, query):
        tables = parse_one(query).find_all(exp.Table)
        return {table.name: self.table(table.name) for table in tables}

    def register_udwf(self, func: WindowUDF):
        self.con.register_udwf(func)


def connect(config: SessionConfig | None = None):
    con = Backend()
    con.do_connect(config)
    return con
