from __future__ import annotations

import sqlglot as sg
import sqlglot.expressions as sge

import xorq.vendor.ibis.expr.datatypes as dt
import xorq.vendor.ibis.expr.operations as ops
from xorq.backends.xorq_datafusion.compiler import (
    DataFusionCompiler as XorqDataFusionCompiler,
)
from xorq.vendor.ibis.backends.sql.compilers.base import NULL
from xorq.vendor.ibis.backends.sql.compilers.datafusion import (
    DataFusionCompiler as IbisDataFusionCompiler,
)


# xorq_datafusion's rules for these run on the datafusion engine too
PORTED_OPS = frozenset(
    (
        ops.DateDelta,
        ops.IntervalFromInteger,
        ops.Strftime,
        ops.StringToDate,
        ops.StringToTimestamp,
        ops.TimeDelta,
        ops.TimestampBucket,
        ops.TimestampDelta,
    )
)


class DataFusionCompiler(IbisDataFusionCompiler):
    UNSUPPORTED_OPS = tuple(
        op for op in IbisDataFusionCompiler.UNSUPPORTED_OPS if op not in PORTED_OPS
    )

    SIMPLE_OPS = IbisDataFusionCompiler.SIMPLE_OPS | {
        op: XorqDataFusionCompiler.SIMPLE_OPS[op]
        for op in (ops.StringToDate, ops.StringToTimestamp)
    }

    visit_DateDelta = XorqDataFusionCompiler.visit_DateDelta
    visit_IntervalFromInteger = XorqDataFusionCompiler.visit_IntervalFromInteger
    visit_Strftime = XorqDataFusionCompiler.visit_Strftime
    visit_TimestampBucket = XorqDataFusionCompiler.visit_TimestampBucket
    visit_TimestampDelta = XorqDataFusionCompiler.visit_TimestampDelta

    def visit_TimeDelta(self, op, *, part, left, right):
        # this backend stores time as Time64(us), and DataFusion does not
        # coerce it to the Time64(ns) the time_delta UDF is registered with
        left, right = (self.cast(arg, dt.time) for arg in (left, right))
        return XorqDataFusionCompiler.visit_TimeDelta(
            self, op, part=part, left=left, right=right
        )

    def visit_Quantile(self, op, *, arg, quantile, where):
        # percentile_cont rejects non-numeric input: index into the sorted non-null values
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
