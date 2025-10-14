import functools
import importlib.metadata
from abc import ABC
from typing import Any, Mapping

import ibis
import ibis.expr.operations as ops
import pandas as pd
import pyarrow as pa
from ibis.backends.sql import SQLBackend
from ibis.expr import schema as sch
from ibis.expr import types as ir


class ExecutionBackend(SQLBackend, ABC):
    def _pandas_execute(self, expr: ir.Expr, **kwargs):
        from xorq.expr.api import _transform_expr
        from xorq.expr.relations import FlightExpr, FlightUDXF

        node = expr.op()
        if isinstance(node, (FlightExpr, FlightUDXF)):
            df = node.to_rbr().read_pandas(timestamp_as_object=True)
            return expr.__pandas_result__(df)
        (expr, created) = _transform_expr(expr)

        return super().execute(expr, **kwargs)

    def execute(self, expr, **kwargs) -> Any:
        if self.name == "pandas":
            return self._pandas_execute(expr, **kwargs)

        batch_reader = self.to_pyarrow_batches(expr, **kwargs)
        df = batch_reader.read_pandas(timestamp_as_object=True)

        return expr.__pandas_result__(df)

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        chunk_size: int = 1_000_000,
        **kwargs: Any,
    ):
        from xorq.common.utils.defer_utils import rbr_wrapper
        from xorq.expr.api import _transform_expr
        from xorq.expr.relations import FlightExpr, FlightUDXF

        if isinstance(expr.op(), (FlightExpr, FlightUDXF)):
            return expr.op().to_rbr()
        (expr, created) = _transform_expr(expr)
        reader = super().to_pyarrow_batches(expr, chunk_size=chunk_size, **kwargs)

        def clean_up():
            for table_name, conn in created.items():
                try:
                    conn.drop_table(table_name, force=True)
                except Exception:
                    conn.drop_view(table_name)

        return rbr_wrapper(reader, clean_up)

    def _pandas_to_pyarrow(self, expr, **kwargs):
        from xorq.expr.api import _transform_expr
        from xorq.expr.relations import FlightExpr, FlightUDXF

        node = expr.op()
        if isinstance(node, (FlightExpr, FlightUDXF)):
            df = node.to_rbr().read_pandas(timestamp_as_object=True)
            return expr.__pyarrow_result__(df)
        (expr, created) = _transform_expr(expr)

        return super().to_pyarrow(expr, **kwargs)

    def to_pyarrow(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ):
        if self.name == "pandas":
            return self._pandas_to_pyarrow(expr, **kwargs)

        batch_reader = self.to_pyarrow_batches(expr, **kwargs)
        arrow_table = batch_reader.read_all()
        return expr.__pyarrow_result__(arrow_table)

    @property
    def version(self) -> str:
        return super().version

    def list_tables(
        self, *, like: str | None = None, database: tuple[str, str] | str | None = None
    ) -> list[str]:
        return super().list_tables(like=like, database=database)

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        return super()._get_schema_using_query(query)

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        return super()._register_in_memory_table(op)

    def create_table(
        self,
        name: str,
        /,
        obj: pd.DataFrame | pa.Table | ir.Table | None = None,
        *,
        schema: ibis.Schema | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ) -> ir.Table:
        return super().create_table(
            name,
            obj=obj,
            schema=schema,
            database=database,
            temp=temp,
            overwrite=overwrite,
        )

    def table(
        self, name: str, /, *, database: tuple[str, str] | str | None = None
    ) -> ir.Table:
        return super().table(name, database=database)


@functools.cache
def _get_backend_names(*, exclude: tuple[str] = ()) -> frozenset[str]:
    """Return the set of known backend names.

    Parameters
    ----------
    exclude
        These backend names should be excluded from the result

    Notes
    -----
    This function returns a frozenset to prevent cache pollution.

    If a `set` is used, then any in-place modifications to the set
    are visible to every caller of this function.

    """

    entrypoints = importlib.metadata.entry_points(group="xorq.backends")
    return frozenset(ep.name for ep in entrypoints).difference(exclude)
