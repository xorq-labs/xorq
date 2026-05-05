from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow_hotfix  # noqa: F401
from sqlglot import exp, parse_one

from xorq.backends.xorq_datafusion.datafusion import Backend as DataFusionBackend
from xorq.internal import SessionConfig, WindowUDF
from xorq.vendor.ibis.expr import types as ir


class Backend(DataFusionBackend):
    name = "xorq_datafusion"

    def register(self, source, table_name=None, **kwargs):
        if isinstance(source, ir.Expr) and hasattr(source, "to_pyarrow_batches"):
            backends, has_unbound = source._find_backends()
            if len(backends) > 1:
                raise ValueError("Multiple backends found for this expression")
            elif len(backends) == 1 and isinstance(backends[0], DataFusionBackend):
                # Compile to a native DataFrame to avoid IbisTableProvider, which
                # would cause a nested tokio runtime panic on execute_stream().
                backend = backends[0]
                backend._register_udfs(source)
                backend._register_in_memory_tables(source)
                raw_sql = backend.compile(source.as_table(), **kwargs)
                source = backend.con.sql(raw_sql)
            elif not backends and not has_unbound:
                source = self.execute(source)
        return super().register(source, table_name=table_name, **kwargs)

    def execute(self, expr: ir.Expr, **kwargs: Any):
        batch_reader = self.to_pyarrow_batches(expr, **kwargs)
        return expr.__pandas_result__(
            batch_reader.read_pandas(timestamp_as_object=True)
        )

    def to_pyarrow(self, expr: ir.Expr, **kwargs: Any) -> pa.Table:
        batch_reader = self.to_pyarrow_batches(expr, **kwargs)
        arrow_table = batch_reader.read_all()
        return expr.__pyarrow_result__(arrow_table)

    def _extract_catalog(self, query):
        tables = parse_one(query).find_all(exp.Table)
        return {table.name: self.table(table.name) for table in tables}

    def register_udwf(self, func: WindowUDF):
        self.con.register_udwf(func)


def connect(config: SessionConfig | None = None):
    con = Backend()
    con.do_connect(config)
    return con
