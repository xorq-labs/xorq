from __future__ import annotations

import sqlglot as sg
import sqlglot.expressions as sge

import xorq.vendor.ibis.expr.datatypes as dt
from xorq.vendor.ibis.backends.sql.compilers.base import NULL
from xorq.vendor.ibis.backends.sql.compilers.datafusion import (
    DataFusionCompiler as IbisDataFusionCompiler,
)


class DataFusionCompiler(IbisDataFusionCompiler):
    def visit_Quantile(self, op, *, arg, quantile, where):
        # datafusion < 49 has no percentile_cont: index into the sorted non-null values
        cond = arg.is_(sg.not_(NULL, copy=False))
        if where is not None:
            cond = sge.And(this=cond, expression=where)
        values = self.f.array_sort(self.agg.array_agg(arg, where=cond))
        n = self.f.cardinality(values)
        if not op.arg.dtype.is_numeric():
            # discrete: the first value whose cumulative share reaches quantile
            index = self.f.greatest(self.f.ceil(quantile * n), 1)
            return self.f.array_element(values, self.cast(index, dt.int64))
        # continuous: interpolate between the two values straddling the position
        pos = quantile * (n - 1) + 1
        frac = sge.paren(pos - self.f.floor(pos))
        lo, hi = (
            self.f.array_element(values, self.cast(bound, dt.int64))
            for bound in (self.f.floor(pos), self.f.ceil(pos))
        )
        return self.if_(lo.eq(hi), lo, lo * (1 - frac) + hi * frac)


compiler = DataFusionCompiler()
