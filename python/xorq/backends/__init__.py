import functools
import importlib.metadata
from abc import ABC
from typing import Any, Mapping

from xorq.vendor.ibis import BaseBackend
from xorq.vendor.ibis.expr import types as ir


class ExecutionBackend(BaseBackend, ABC):
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
