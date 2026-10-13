from __future__ import annotations

from functools import partial, reduce

import sqlglot as sg
import sqlglot.expressions as sge
from sqlglot import exp

import xorq.expr.datatypes as dt
import xorq.vendor.ibis.expr.operations as ops
from xorq.expr.datatypes import LargeString
from xorq.vendor.ibis.backends.sql.compilers.base import NULL
from xorq.vendor.ibis.backends.sql.compilers.datafusion import (
    DataFusionCompiler as IbisDataFusionCompiler,
)
from xorq.vendor.ibis.backends.sql.datatypes import (
    DataFusionType as IbisDataFusionType,
)
from xorq.vendor.ibis.backends.sql.dialects import DataFusion as IbisDataFusion


_UNIX_EPOCH = "1970-01-01T00:00:00Z"

typecode = sge.DataType.Type


class DataFusionType(IbisDataFusionType):
    # the vendored mapper parses and renders type strings as "datafusion";
    # this one has always done so as "postgres" (inherited from PostgresType),
    # which renders LargeString as TEXT in `to_string`
    dialect = "postgres"
    unknown_type_strings = {
        "utf8": dt.string,
        "utf8view": dt.string,
        "float64": dt.float64,
        "date32": dt.date,
    }

    @classmethod
    def _from_ibis_LargeString(cls, dtype: LargeString) -> sge.DataType:
        return sge.DataType(this=typecode.LONGTEXT)

    @classmethod
    def _from_sqlglot_LONGTEXT(cls, nullable: bool | None = None) -> LargeString:
        return LargeString(nullable=nullable)


def _int_div_sql(self: DataFusion.Generator, expression: exp.IntDiv) -> str:
    return f"{self.sql(expression, 'this')} / {self.sql(expression, 'expression')}"


class DataFusion(IbisDataFusion):
    class Generator(IbisDataFusion.Generator):
        TRANSFORMS = IbisDataFusion.Generator.TRANSFORMS.copy() | {
            sge.IntDiv: _int_div_sql,
        }

        TYPE_MAPPING = IbisDataFusion.Generator.TYPE_MAPPING.copy() | {
            exp.DataType.Type.LONGTEXT: "'LargeUtf8'",
        }


class DataFusionCompiler(IbisDataFusionCompiler):
    __slots__ = ()

    dialect = DataFusion
    type_mapper = DataFusionType

    UNSUPPORTED_OPS = (
        ops.ArgMax,
        ops.ArgMin,
        ops.ArrayFilter,
        ops.ArrayMap,
        ops.ArrayZip,
        ops.CountDistinctStar,
        ops.RowID,
    )

    SIMPLE_OPS = {
        ops.ApproxQuantile: "approx_percentile_cont",
        ops.ApproxMedian: "approx_median",
        ops.ArrayRemove: "array_remove_all",
        ops.BitAnd: "bit_and",
        ops.BitOr: "bit_or",
        ops.BitXor: "bit_xor",
        ops.Cot: "cot",
        ops.ExtractMicrosecond: "extract_microsecond",
        ops.Median: "median",
        ops.StringLength: "character_length",
        ops.RandomUUID: "uuid",
        ops.RegexSplit: "regex_split",
        ops.EndsWith: "ends_with",
        ops.ArrayIntersect: "array_intersect",
        ops.ArrayUnion: "array_union",
        ops.ArrayFlatten: "flatten",
        ops.IntegerRange: "range",
        ops.ArrayDistinct: "array_distinct",
        ops.Unnest: "unnest",
        ops.StringToDate: "to_date",
        ops.StringToTimestamp: "to_timestamp",
        ops.MapKeys: "map_keys",
        ops.MapValues: "map_values",
    }

    def visit_Cast(self, op, *, arg, to):
        if to.is_interval():
            unit = to.unit.name.lower()
            return sg.cast(
                self.f.concat(self.cast(arg, dt.string), f" {unit}"), "interval"
            )
        if to.is_timestamp():
            return self._to_timestamp(arg, to)
        if to.is_decimal():
            from xorq.vendor.ibis.formats.pyarrow import PyArrowType  # noqa: PLC0415

            return self.f.arrow_cast(arg, f"{PyArrowType.from_ibis(to)}".capitalize())

        if isinstance(to, LargeString):
            return self.f.arrow_cast(arg, "LargeUtf8")

        return self.cast(arg, to)

    def visit_ElementWiseVectorizedUDF(
        self, op, *, func, func_args, input_type, return_type
    ):
        return self.f[op.__func_name__](*func_args)

    def visit_ArraySlice(self, op, *, arg, start, stop):
        array_length = self.f.array_length(arg)
        start = self.cast(self.f.coalesce(start, 0), dt.int64)
        stop = self.cast(self.f.coalesce(stop, array_length + 1), dt.int64)
        return self.f.array_slice(
            arg,
            self.if_(
                start < 0,
                self.if_(self.f.abs(start) >= array_length, 0, start),
                start + 1,
            ),
            self.if_(stop < 0, stop - 1, stop),
        )

    def visit_TimestampNow(self, op):
        return self.f.to_timestamp_micros(self.cast(self.f.now(), dt.string))

    def visit_HexDigest(self, op, *, arg, how):
        return self.f.encode(self.f.digest(arg, how), "hex")

    def visit_TypeOf(self, op, *, arg):
        return self.f.arrow_typeof(arg)

    def visit_BitwiseNot(self, op, *, arg):
        # https://stackoverflow.com/q/69648488/4001592
        return sge.BitwiseXor(this=arg, expression=sg.exp.convert(-1))

    def visit_Clip(self, op, *, arg, lower, upper):
        ifs = []
        if lower is not None:
            lower_case = self.if_(arg < lower, lower)
            ifs.append(lower_case)
        if upper is not None:
            upper_case = self.if_(arg > upper, upper)
            ifs.append(upper_case)

        return sg.exp.Case(ifs=ifs, default=arg)

    def visit_IntervalFromInteger(self, op, *, arg, unit):
        unit = unit.name.lower()
        return sg.cast(self.f.concat(self.cast(arg, dt.string), f" {unit}"), "interval")

    def visit_Greatest(self, op, *, arg):
        return self.f.greatest(*arg)

    def visit_Least(self, op, *, arg):
        return self.f.least(*arg)

    def visit_TimestampBucket(self, op, *, arg, interval, offset):
        if offset is None:
            return self.f.date_bin(interval, arg)
        else:
            # bin boundaries must sit at offset + k * interval, so the origin
            # is the epoch shifted forward by the offset
            offset = (
                self.f.arrow_cast(_UNIX_EPOCH, "Timestamp(Nanosecond, None)") + offset
            )
        return self.f.date_bin(interval, arg, offset)

    def visit_StructField(self, op, *, arg, field):
        return sge.Bracket(this=arg, expressions=[sg.exp.convert(field)])

    def visit_Hash(self, op, *, arg):
        return self.f.hash_int(arg)

    def visit_ArrayAll(self, op, *, arg):
        value_type = op.arg.dtype.value_type
        return self.if_(
            arg.is_(NULL),
            self.cast(NULL, dt.bool),
            self.if_(
                self.f.array_length(arg) > 0,
                self.if_(
                    self.f.array_has_all(
                        self.f.make_array(self.cast(NULL, value_type)), arg
                    ),
                    self.cast(NULL, dt.bool),
                    self.f.array_has_all(
                        self.f.make_array(True, self.cast(NULL, value_type)), arg
                    ),
                ),
                self.cast(NULL, dt.bool),
            ),
        )

    def visit_ArrayAny(self, op, *, arg):
        value_type = op.arg.dtype.value_type
        return self.if_(
            arg.is_(NULL),
            self.cast(NULL, dt.bool),
            self.if_(
                self.f.array_length(arg) > 0,
                self.if_(
                    self.f.array_has_all(
                        self.f.make_array(self.cast(NULL, value_type)), arg
                    ),
                    self.cast(NULL, dt.bool),
                    self.f.array_has_any(self.f.make_array(True), arg),
                ),
                self.cast(NULL, dt.bool),
            ),
        )

    def visit_Strftime(self, op, *, arg, format_str):
        return self.f.date_format(arg, format_str)

    def visit_TimeDelta(self, op, *, part, left, right):
        unit = part.this.lower()
        return self.f.time_delta(unit, right, left)

    def visit_TimestampDelta(self, op, *, part, left, right):
        unit = part.this.lower()
        return self.f.timestamp_delta(unit, right, left)

    def visit_RegexReplace(self, op, *, arg, pattern, replacement):
        return self.f.regexp_replace(arg, pattern, replacement)

    visit_DateDelta = visit_TimestampDelta

    def visit_TableUnnest(
        self,
        op,
        *,
        parent,
        column,
        column_name: str,
        offset: str | None,
        keep_empty: bool,
    ):
        quoted = self.quoted
        table = sg.to_identifier(parent.alias_or_name, quoted=quoted)
        column_alias = sg.to_identifier(column_name, quoted=quoted)
        value_type = op.column.dtype.value_type

        if keep_empty:
            column = self.if_(
                sg.or_(column.is_(NULL), self.f.empty(column)),
                self.f.make_array(self.cast(NULL, value_type)),
                column,
            )

        selcols = tuple(
            sge.Column(this=sg.to_identifier(name, quoted=quoted), table=table)
            if name != column_name
            else sge.Unnest(expressions=[column]).as_(column_alias)
            for name in op.parent.schema.names
        )

        if offset:
            offset_alias = sg.to_identifier(offset, quoted=quoted)
            selcols += (
                sge.Unnest(expressions=[self.f.range(self.f.array_length(column))]).as_(
                    offset_alias
                ),
            )

        res = sg.select(*selcols).from_(parent)
        return res

    def visit_Quantile(self, op, *, arg, quantile, where):
        expr = sge.WithinGroup(
            this=self.f.percentile_cont(quantile),
            expression=sge.Order(expressions=[sge.Ordered(this=arg)]),
        )
        if where is not None:
            expr = sge.Filter(this=expr, expression=sge.Where(this=where))
        return expr

    def visit_ArrayConcat(self, op, *, arg):
        return reduce(
            lambda x, y: self.if_(
                x.is_(NULL).or_(y.is_(NULL)), NULL, self.f.array_cat(x, y)
            ),
            map(partial(self.cast, to=op.dtype), arg),
        )

    def visit_MapGet(self, op, *, arg, key, default):
        if op.dtype.is_null():
            return NULL
        return self.f.coalesce(self.f.map_extract(arg, key)[1], default)

    def visit_MapContains(self, op, *, arg, key):
        return self.f.array_has(self.f.map_keys(arg), key)

    def visit_MapLength(self, op, *, arg):
        return self.f.array_length(self.f.map_keys(arg))


compiler = DataFusionCompiler()
