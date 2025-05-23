from __future__ import annotations

import operator
from functools import reduce

import numpy as np
import pandas as pd
from packaging.version import parse as vparse

import xorq.vendor.ibis.backends.pandas.kernels as pandas_kernels
import xorq.vendor.ibis.expr.operations as ops
from xorq.common.exceptions import (
    OperationNotDefinedError,
    UnboundExpressionError,
    UnsupportedOperationError,
)
from xorq.vendor.ibis.backends.pandas.convert import PandasConverter
from xorq.vendor.ibis.backends.pandas.helpers import (
    GroupedFrame,
    PandasUtils,
    RangeFrame,
    RowsFrame,
    UngroupedFrame,
)
from xorq.vendor.ibis.backends.pandas.rewrites import (
    PandasAggregate,
    PandasAsofJoin,
    PandasJoin,
    PandasLimit,
    PandasRename,
    PandasResetIndex,
    PandasScalarSubquery,
    PandasWindowFrame,
    PandasWindowFunction,
    plan,
)
from xorq.vendor.ibis.formats.pandas import PandasData, PandasType
from xorq.vendor.ibis.util import any_of, gen_name


# ruff: noqa: F811


class PandasExecutor(PandasUtils):
    name = "pandas"
    kernels = pandas_kernels

    @classmethod
    def visit(cls, op: ops.Node, **kwargs):
        raise OperationNotDefinedError(
            f"Operation {op!r} is not implemented for the pandas backend"
        )

    @classmethod
    def visit_Literal(cls, op: ops.Literal, value, dtype):
        if dtype.is_interval():
            value = pd.Timedelta(value, dtype.unit.short)
        elif dtype.is_array():
            value = np.array(value)
        elif dtype.is_date():
            value = pd.Timestamp(value, tz="UTC").tz_localize(None)
        return value

    @classmethod
    def visit_Field(cls, op: ops.Field, rel, name):
        return rel[name]

    @classmethod
    def visit_Alias(cls, op: ops.Alias, arg, name):
        try:
            return arg.rename(name)
        except AttributeError:
            return arg

    @classmethod
    def visit_SortKey(cls, op: ops.SortKey, expr, ascending, nulls_first):
        return expr

    @classmethod
    def visit_Cast(cls, op: ops.Cast, arg, to):
        if arg is None:
            return None
        elif isinstance(arg, pd.Series):
            return PandasConverter.convert_column(arg, to)
        else:
            return PandasConverter.convert_scalar(arg, to)

    @classmethod
    def visit_Greatest(cls, op: ops.Greatest, arg):
        return cls.columnwise(lambda df: df.max(axis=1), arg)

    @classmethod
    def visit_Least(cls, op: ops.Least, arg):
        return cls.columnwise(lambda df: df.min(axis=1), arg)

    @classmethod
    def visit_Coalesce(cls, op: ops.Coalesce, arg):
        return cls.columnwise(lambda df: df.bfill(axis=1).iloc[:, 0], arg)

    @classmethod
    def visit_Value(cls, op: ops.Value, **operands):
        # automatically pick the correct kernel based on the operand types
        typ = type(op)
        name = op.name
        dtype = PandasType.from_ibis(op.dtype)
        kwargs = {"operands": operands, "name": name, "dtype": dtype}

        # decimal operations have special implementations
        if op.dtype.is_decimal():
            func = cls.kernels.elementwise_decimal[typ]
            return cls.elementwise(func, **kwargs)

        # prefer generic implementations if available
        if func := cls.kernels.generic.get(typ):
            return cls.generic(func, **kwargs)

        if len(operands) < 1:
            raise OperationNotDefinedError(
                f"No implementation found for operation {typ}"
            )
        _, *rest = operands.values()
        is_multi_arg = bool(rest)
        is_multi_column = any_of(rest, pd.Series)

        if is_multi_column:
            if func := cls.kernels.columnwise.get(typ):
                return cls.columnwise(func, **kwargs)
            elif func := cls.kernels.rowwise.get(typ):
                return cls.rowwise(func, **kwargs)
            else:
                raise OperationNotDefinedError(
                    "No columnwise or rowwise implementation found for "
                    f"multi-column operation {typ}"
                )
        elif is_multi_arg:
            if func := cls.kernels.columnwise.get(typ):
                return cls.columnwise(func, **kwargs)
            elif func := cls.kernels.serieswise.get(typ):
                return cls.serieswise(func, **kwargs)
            elif func := cls.kernels.rowwise.get(typ):
                return cls.rowwise(func, **kwargs)
            elif func := cls.kernels.elementwise.get(typ):
                return cls.elementwise(func, **kwargs)
            else:
                raise OperationNotDefinedError(
                    "No columnwise, serieswise, rowwise or elementwise "
                    f"implementation found for multi-argument operation {typ}"
                )
        else:  # noqa: PLR5501
            if func := cls.kernels.serieswise.get(typ):
                return cls.serieswise(func, **kwargs)
            elif func := cls.kernels.elementwise.get(typ):
                return cls.elementwise(func, **kwargs)
            else:
                raise OperationNotDefinedError(
                    "No serieswise or elementwise implementation found for "
                    f"single-argument operation {typ}"
                )

    @classmethod
    def visit_IsNan(cls, op: ops.IsNan, arg):
        try:
            return np.isnan(arg)
        except (TypeError, ValueError):
            # if `arg` contains `None` np.isnan will complain
            # so we take advantage of NaN not equaling itself
            # to do the correct thing
            return arg != arg

    @classmethod
    def visit_SearchedCase(
        cls, op: ops.SearchedCase | ops.SimpleCase, cases, results, default, base=None
    ):
        if base is not None:
            cases = tuple(base == case for case in cases)
        cases, _ = cls.asframe(cases, concat=False)
        index = cases[0].index
        results, _ = cls.asframe(results, concat=False)
        out = np.select(cases, results, default)
        return pd.Series(out, index=index)

    @classmethod
    def visit_TimestampTruncate(
        cls, op: ops.TimestampTruncate | ops.DateTruncate, arg, unit
    ):
        # TODO(kszucs): should use serieswise()
        if vparse(pd.__version__) >= vparse("2.2"):
            units = {"m": "min"}
        else:
            units = {"m": "Min", "ms": "L"}

        unit = units.get(unit.short, unit.short)

        if unit in "YQMWD":
            return arg.dt.to_period(unit).dt.to_timestamp()
        try:
            return arg.dt.floor(unit)
        except ValueError:
            return arg.dt.to_period(unit).dt.to_timestamp()

    @classmethod
    def visit_IntervalFromInteger(cls, op: ops.IntervalFromInteger, unit, **kwargs):
        if unit.short in {"Y", "Q", "M", "W"}:
            return cls.elementwise(lambda v: pd.DateOffset(**{unit.plural: v}), kwargs)
        else:
            return cls.serieswise(
                lambda arg: arg.astype(f"timedelta64[{unit.short}]"), kwargs
            )

    @classmethod
    def visit_BetweenTime(cls, op: ops.BetweenTime, arg, lower_bound, upper_bound):
        idx = pd.DatetimeIndex(arg)
        if idx.tz is not None:
            idx = idx.tz_convert(None)  # make naive because times are naive
        indexer = idx.indexer_between_time(lower_bound, upper_bound)
        result = np.zeros(len(arg), dtype=np.bool_)
        result[indexer] = True
        return pd.Series(result)

    @classmethod
    def visit_FindInSet(cls, op: ops.FindInSet, needle, values):
        (needle, *haystack), _ = cls.asframe((needle, *values), concat=False)
        condlist = [needle == col for col in haystack]
        choicelist = [i for i, _ in enumerate(haystack)]
        result = np.select(condlist, choicelist, default=-1)
        return pd.Series(result, name=op.name)

    @classmethod
    def visit_Array(cls, op: ops.Array, exprs):
        return cls.rowwise(lambda row: np.array(row, dtype=object), exprs)

    @classmethod
    def visit_StructColumn(cls, op: ops.StructColumn, names, values):
        return cls.rowwise(lambda row: dict(zip(names, row)), values)

    @classmethod
    def visit_ArrayConcat(cls, op: ops.ArrayConcat, arg):
        return cls.rowwise(lambda row: np.concatenate(row.values), arg)

    @classmethod
    def visit_Unnest(cls, op: ops.Unnest, arg):
        arg = cls.asseries(arg)
        mask = arg.map(lambda v: bool(len(v)), na_action="ignore")
        return arg[mask].explode()

    @classmethod
    def visit_ElementWiseVectorizedUDF(
        cls, op: ops.ElementWiseVectorizedUDF, func, func_args, input_type, return_type
    ):
        """Execute an elementwise UDF."""

        res = func(*func_args)
        if isinstance(res, pd.DataFrame):
            # it is important otherwise it is going to fill up the memory
            res = res.apply(lambda row: row.to_dict(), axis=1)

        return res

    ############################# Reductions ##################################

    @classmethod
    def visit_Reduction(cls, op: ops.Reduction, arg, where, order_by=()):
        if order_by:
            raise UnsupportedOperationError(
                "ordering of order-sensitive aggregations via `order_by` is "
                "not supported for this backend"
            )
        func = cls.kernels.reductions[type(op)]
        return cls.agg(func, arg, where)

    @classmethod
    def visit_CountStar(cls, op: ops.CountStar, arg, where):
        def agg(df):
            if where is None:
                return len(df)
            else:
                return df[where.name].sum()

        return agg

    @classmethod
    def visit_CountDistinctStar(cls, op: ops.CountDistinctStar, arg, where):
        def agg(df):
            if where is None:
                return df.nunique()
            else:
                return df[where.name].nunique()

        return agg

    @classmethod
    def visit_Arbitrary(cls, op: ops.Arbitrary, arg, where):
        return cls.agg(cls.kernels.reductions[ops.Arbitrary], arg, where)

    @classmethod
    def visit_ArgMin(cls, op: ops.ArgMin | ops.ArgMax, arg, key, where):
        func = operator.methodcaller(op.__class__.__name__.lower())

        if where is None:

            def agg(df):
                indices = func(df[key.name])
                return df[arg.name].iloc[indices]
        else:

            def agg(df):
                mask = df[where.name]
                filtered = df[mask]
                indices = func(filtered[key.name])
                return filtered[arg.name].iloc[indices]

        return agg

    @classmethod
    def visit_Variance(cls, op: ops.Variance, arg, where, how):
        ddof = {"pop": 0, "sample": 1}[how]
        return cls.agg(lambda x: x.var(ddof=ddof), arg, where)

    @classmethod
    def visit_StandardDev(cls, op: ops.StandardDev, arg, where, how):
        ddof = {"pop": 0, "sample": 1}[how]
        return cls.agg(lambda x: x.std(ddof=ddof), arg, where)

    @classmethod
    def visit_ArrayCollect(
        cls, op: ops.ArrayCollect, arg, where, order_by, include_null
    ):
        if order_by:
            raise UnsupportedOperationError(
                "ordering of order-sensitive aggregations via `order_by` is "
                "not supported for this backend"
            )
        return cls.agg(
            (lambda x: x.tolist() if include_null else x.dropna().tolist()), arg, where
        )

    @classmethod
    def visit_First(cls, op: ops.First, arg, where, order_by, include_null):
        if order_by:
            raise UnsupportedOperationError(
                "ordering of order-sensitive aggregations via `order_by` is "
                "not supported for this backend"
            )

        def first(arg):
            if not include_null:
                arg = arg.dropna()
            return arg.iat[0] if len(arg) else None

        return cls.agg(first, arg, where)

    @classmethod
    def visit_Last(cls, op: ops.Last, arg, where, order_by, include_null):
        if order_by:
            raise UnsupportedOperationError(
                "ordering of order-sensitive aggregations via `order_by` is "
                "not supported for this backend"
            )

        def last(arg):
            if not include_null:
                arg = arg.dropna()
            return arg.iat[-1] if len(arg) else None

        return cls.agg(last, arg, where)

    @classmethod
    def visit_Correlation(cls, op: ops.Correlation, left, right, where, how):
        if where is None:

            def agg(df):
                return df[left.name].corr(df[right.name])
        else:

            def agg(df):
                mask = df[where.name]
                lhs = df[left.name][mask]
                rhs = df[right.name][mask]
                return lhs.corr(rhs)

        return agg

    @classmethod
    def visit_Covariance(cls, op: ops.Covariance, left, right, where, how):
        ddof = {"pop": 0, "sample": 1}[how]
        if where is None:

            def agg(df):
                return df[left.name].cov(df[right.name], ddof=ddof)
        else:

            def agg(df):
                mask = df[where.name]
                lhs = df[left.name][mask]
                rhs = df[right.name][mask]
                return lhs.cov(rhs, ddof=ddof)

        return agg

    @classmethod
    def visit_GroupConcat(cls, op: ops.GroupConcat, arg, sep, where, order_by):
        if order_by:
            raise UnsupportedOperationError(
                "ordering of order-sensitive aggregations via `order_by` is "
                "not supported for this backend"
            )

        if where is None:

            def agg(df):
                return sep.join(df[arg.name].astype(str))
        else:

            def agg(df):
                mask = df[where.name]
                group = df[arg.name][mask]
                if group.empty:
                    return pd.NA
                return sep.join(group)

        return agg

    @classmethod
    def visit_Quantile(cls, op: ops.Quantile, arg, quantile, where):
        return cls.agg(lambda x: x.quantile(quantile), arg, where)

    @classmethod
    def visit_MultiQuantile(cls, op: ops.MultiQuantile, arg, quantile, where):
        return cls.agg(lambda x: list(x.quantile(quantile)), arg, where)

    @classmethod
    def visit_ReductionVectorizedUDF(
        cls, op: ops.ReductionVectorizedUDF, func, func_args, input_type, return_type
    ):
        def agg(df):
            args = [df[col.name] for col in func_args]
            return func(*args)

        return agg

    ############################# Analytic ####################################

    @classmethod
    def visit_RowNumber(cls, op: ops.RowNumber):
        def agg(df, order_keys):
            return pd.Series(np.arange(len(df)), index=df.index)

        return agg

    @classmethod
    def visit_Lag(cls, op: ops.Lag | ops.Lead, arg, offset, default):
        if isinstance(op, ops.Lag):
            sign = operator.pos
        else:
            sign = operator.neg

        if op.offset is not None and op.offset.dtype.is_interval():

            def agg(df, order_keys):
                df = df.set_index(order_keys)
                col = df[arg.name].shift(freq=sign(offset))
                res = col.reindex(df.index)
                if not pd.isnull(default):
                    res = res.fillna(default)
                return res.reset_index(drop=True)

        else:
            offset = 1 if offset is None else offset

            def agg(df, order_keys):
                res = df[arg.name].shift(sign(offset))
                if not pd.isnull(default):
                    res = res.fillna(default)
                return res

        return agg

    @classmethod
    def visit_MinRank(cls, op: ops.MinRank | ops.DenseRank):
        method = "dense" if isinstance(op, ops.DenseRank) else "min"

        def agg(df, order_keys):
            if len(order_keys) == 0:
                raise ValueError("order_by argument is required for rank functions")
            elif len(order_keys) == 1:
                s = df[order_keys[0]]
            else:
                s = df[order_keys].apply(tuple, axis=1)

            return s.rank(method=method).astype("int64") - 1

        return agg

    @classmethod
    def visit_PercentRank(cls, op: ops.PercentRank):
        def agg(df, order_keys):
            if len(order_keys) == 0:
                raise ValueError("order_by argument is required for rank functions")
            elif len(order_keys) == 1:
                s = df[order_keys[0]]
            else:
                s = df[order_keys].apply(tuple, axis=1)

            return s.rank(method="min").sub(1).div(len(df) - 1)

        return agg

    @classmethod
    def visit_CumeDist(cls, op: ops.CumeDist):
        def agg(df, order_keys):
            if len(order_keys) == 0:
                raise ValueError("order_by argument is required for rank functions")
            elif len(order_keys) == 1:
                s = df[order_keys[0]]
            else:
                s = df[order_keys].apply(tuple, axis=1)

            return s.rank(method="average", pct=True)

        return agg

    @classmethod
    def visit_AnalyticVectorizedUDF(
        cls, op: ops.AnalyticVectorizedUDF, func, func_args, input_type, return_type
    ):
        def agg(df, order_keys):
            args = [df[col.name] for col in func_args]
            res = func(*args)
            if isinstance(res, pd.DataFrame):
                # it is important otherwise it is going to fill up the memory
                res = res.apply(lambda row: row.to_dict(), axis=1)
            return res

        return agg

    ############################ Window functions #############################

    @classmethod
    def visit_WindowBoundary(cls, op: ops.WindowBoundary, value, preceding):
        return value

    @classmethod
    def visit_PandasWindowFrame(
        cls, op: PandasWindowFrame, table, how, start, end, group_by, order_by
    ):
        if start is not None and op.start.preceding:
            start = -start
        if end is not None and op.end.preceding:
            end = -end

        table = table.assign(__start__=start, __end__=end)

        # TODO(kszucs): order by ibis.random() is not supported because it is
        # excluded from the group by keys due to its scalar shape
        group_keys = [group.name for group in op.group_by]
        order_keys = [key.name for key in op.order_by if key.shape.is_columnar()]
        ascending = [key.ascending for key in op.order_by if key.shape.is_columnar()]

        if order_by:
            table = table.sort_values(order_keys, ascending=ascending, kind="mergesort")

        if group_by:
            frame = GroupedFrame(df=table, group_keys=group_keys)
        else:
            frame = UngroupedFrame(df=table)

        if start is None and end is None:
            return frame
        elif how == "rows":
            return RowsFrame(parent=frame)
        elif how == "range":
            if len(order_keys) != 1:
                raise NotImplementedError(
                    "Only single column order by is supported for range window frames"
                )
            return RangeFrame(parent=frame, order_key=order_keys[0])
        else:
            raise NotImplementedError(f"Unsupported window frame type: {how}")

    @classmethod
    def visit_PandasWindowFunction(cls, op: PandasWindowFunction, func, frame):
        if isinstance(op.func, ops.Analytic):
            order_keys = [key.name for key in op.frame.order_by]
            return frame.apply_analytic(func, order_keys=order_keys)
        else:
            return frame.apply_reduction(func)

    ############################ Relational ###################################

    @classmethod
    def visit_DatabaseTable(
        cls, op: ops.DatabaseTable, name, schema, source, namespace
    ):
        try:
            return source.dictionary[name]
        except KeyError:
            raise UnboundExpressionError(
                f"{name} is not a table in the {source.name!r} backend, you "
                "probably tried to execute an expression without a data source"
            )

    @classmethod
    def visit_InMemoryTable(cls, op: ops.InMemoryTable, name, schema, data):
        return data.to_frame()

    @classmethod
    def visit_DummyTable(cls, op: ops.DummyTable, values):
        df, _ = cls.asframe(values)
        return df

    @classmethod
    def visit_Reference(cls, op: ops.Reference, parent, **kwargs):
        return parent

    @classmethod
    def visit_PandasRename(cls, op: PandasRename, parent, mapping):
        return parent.rename(columns=mapping)

    @classmethod
    def visit_PandasLimit(cls, op: PandasLimit, parent, n, offset):
        n = n.iat[0, 0]
        offset = offset.iat[0, 0]
        if n is None:
            return parent.iloc[offset:]
        else:
            return parent.iloc[offset : offset + n]

    @classmethod
    def visit_PandasResetIndex(cls, op: PandasResetIndex, parent):
        return parent.reset_index(drop=True)

    @classmethod
    def visit_Sample(cls, op: ops.Sample, parent, fraction, method, seed):
        return parent.sample(frac=fraction, random_state=seed)

    @classmethod
    def visit_Project(cls, op: ops.Project, parent, values):
        df, all_scalars = cls.asframe(values)
        if all_scalars and len(parent) != len(df):
            df = cls.concat([df] * len(parent))
        return df

    @classmethod
    def visit_Filter(cls, op: ops.Filter, parent, predicates):
        if predicates:
            pred = reduce(operator.and_, predicates)
            if len(pred) != len(parent):
                raise RuntimeError(
                    "Selection predicate length does not match underlying table"
                )
            parent = parent.loc[pred].reset_index(drop=True)
        return parent

    @classmethod
    def visit_Sort(cls, op: ops.Sort, parent, keys):
        # 1. add sort key columns to the dataframe if they are not already present
        # 2. sort the dataframe using those columns
        # 3. drop the sort key columns
        ascending = [key.ascending for key in op.keys]
        nulls_first = [key.nulls_first for key in op.keys]

        if all(nulls_first):
            na_position = "first"
        elif not any(nulls_first):
            na_position = "last"
        else:
            raise ValueError(
                "pandas does not support specifying null ordering for individual columns"
            )

        newcols = {gen_name("sort_key"): col for col in keys}
        names = list(newcols.keys())
        df = parent.assign(**newcols)
        df = df.sort_values(
            by=names,
            ascending=ascending,
            na_position=na_position,
            ignore_index=True,
            kind="mergesort",
        )
        return df.drop(columns=names)

    @classmethod
    def visit_DropColumns(cls, op: ops.DropColumns, parent, columns_to_drop):
        return parent.drop(columns=list(columns_to_drop))

    @classmethod
    def visit_PandasAggregate(cls, op: PandasAggregate, parent, groups, metrics):
        if groups:
            parent = parent.groupby([col.name for col in groups.values()])
            metrics = {k: parent.apply(v) for k, v in metrics.items()}
            result = cls.concat(metrics, axis=1).reset_index()
            renames = {v.name: k for k, v in op.groups.items()}
            return result.rename(columns=renames)
        else:
            results = {k: v(parent) for k, v in metrics.items()}
            combined, _ = cls.asframe(results)
            return combined

    @classmethod
    def visit_PandasJoin(cls, op: PandasJoin, how, left, right, left_on, right_on):
        # broadcast predicates if they are scalar values
        left_on = [cls.asseries(v, like=left) for v in left_on]
        right_on = [cls.asseries(v, like=right) for v in right_on]

        if how == "cross":
            assert not left_on and not right_on
            return cls.merge(left, right, how="cross")
        elif how == "positional":
            assert not left_on and not right_on
            return cls.concat([left, right], axis=1)
        elif how == "anti":
            df = cls.merge(
                left,
                right,
                how="outer",
                left_on=left_on,
                right_on=right_on,
                indicator=True,
            )
            df = df[df["_merge"] == "left_only"]
            return df.drop(columns=["_merge"])
        elif how == "semi":
            mask = cls.asseries(True, like=left)
            for left_pred, right_pred in zip(left_on, right_on):
                mask = mask & left_pred.isin(right_pred)
            return left[mask]
        else:
            left_columns = {gen_name("left"): s for s in left_on}
            right_columns = {gen_name("right"): s for s in right_on}
            left_keys = list(left_columns.keys())
            right_keys = list(right_columns.keys())
            left = left.assign(**left_columns)
            right = right.assign(**right_columns)
            df = left.merge(right, how=how, left_on=left_keys, right_on=right_keys)
            return df

    @classmethod
    def visit_PandasAsofJoin(
        cls,
        op: PandasAsofJoin,
        how,
        left,
        right,
        left_on,
        right_on,
        left_by,
        right_by,
        operator,
    ):
        # broadcast predicates if they are scalar values
        left_on = [cls.asseries(v, like=left) for v in left_on]
        left_by = [cls.asseries(v, like=left) for v in left_by]
        right_on = [cls.asseries(v, like=right) for v in right_on]
        right_by = [cls.asseries(v, like=right) for v in right_by]

        # merge_asof only works with column names not with series
        left_on = {gen_name("left"): s for s in left_on}
        left_by = {gen_name("left"): s for s in left_by}
        right_on = {gen_name("right"): s for s in right_on}
        right_by = {gen_name("right"): s for s in right_by}

        left = left.assign(**left_on, **left_by)
        right = right.assign(**right_on, **right_by)

        # construct the appropriate flags for merge_asof
        if operator == ops.LessEqual:
            direction = "forward"
            allow_exact_matches = True
        elif operator == ops.GreaterEqual:
            direction = "backward"
            allow_exact_matches = True
        elif operator == ops.Less:
            direction = "forward"
            allow_exact_matches = False
        elif operator == ops.Greater:
            direction = "backward"
            allow_exact_matches = False
        elif operator == ops.Equals:
            direction = "nearest"
            allow_exact_matches = True
        else:
            raise NotImplementedError(
                f"Operator {operator} not supported for asof join"
            )

        # merge_asof requires the left side to be sorted by the join keys
        left = left.sort_values(by=list(left_on.keys()))
        df = cls.merge_asof(
            left,
            right,
            left_on=list(left_on.keys()),
            right_on=list(right_on.keys()),
            left_by=list(left_by.keys()) or None,
            right_by=list(right_by.keys()) or None,
            direction=direction,
            allow_exact_matches=allow_exact_matches,
        )
        return df

    @classmethod
    def visit_Union(cls, op: ops.Union, left, right, distinct):
        result = cls.concat([left, right], axis=0)
        return result.drop_duplicates() if distinct else result

    @classmethod
    def visit_Intersection(cls, op: ops.Intersection, left, right, distinct):
        if not distinct:
            raise NotImplementedError(
                "`distinct=False` is not supported by the pandas backend"
            )
        return left.merge(right, on=list(left.columns), how="inner")

    @classmethod
    def visit_Difference(cls, op: ops.Difference, left, right, distinct):
        if not distinct:
            raise NotImplementedError(
                "`distinct=False` is not supported by the pandas backend"
            )
        merged = left.merge(right, on=list(left.columns), how="outer", indicator=True)
        result = merged[merged["_merge"] == "left_only"].drop("_merge", axis=1)
        return result

    @classmethod
    def visit_Distinct(cls, op: ops.Distinct, parent):
        return parent.drop_duplicates()

    @classmethod
    def visit_DropNull(cls, op: ops.DropNull, parent, how, subset):
        if op.subset is not None:
            subset = [col.name for col in op.subset]
        else:
            subset = None
        return parent.dropna(how=how, subset=subset)

    @classmethod
    def visit_FillNull(cls, op: ops.FillNull, parent, replacements):
        return parent.fillna(replacements)

    @classmethod
    def visit_InValues(cls, op: ops.InValues, value, options):
        if isinstance(value, pd.Series):
            return value.isin(options)
        else:
            return value in options

    @classmethod
    def visit_InSubquery(cls, op: ops.InSubquery, rel, needle):
        first_column = rel.iloc[:, 0]
        if isinstance(needle, pd.Series):
            return needle.isin(first_column)
        else:
            return needle in first_column

    @classmethod
    def visit_PandasScalarSubquery(cls, op: PandasScalarSubquery, rel):
        return rel.iat[0, 0]

    @classmethod
    def execute(cls, node, backend, params):
        def fn(node, _, **kwargs):
            if isinstance(node, ops.Reduction):
                return cls.visit_Reduction(node, **kwargs)
            else:
                method = getattr(cls, f"visit_{type(node).__name__}", None)
                if method is not None:
                    return method(node, **kwargs)
                elif isinstance(node, ops.Value):
                    return cls.visit_Value(node, **kwargs)
                else:
                    raise OperationNotDefinedError(
                        f"No translation rule for {type(node).__name__}"
                    )

            # return cls.visit(node, **kwargs)

        original = node
        node = node.to_expr().as_table().op()
        node = plan(node, backend=backend, params=params)
        df = node.map_clear(fn)

        # TODO(kszucs): add a flag to disable this conversion because it can be
        # expensive for columns with object dtype
        df = PandasData.convert_table(df, node.schema)
        if isinstance(original, ops.Value):
            if original.shape.is_scalar():
                return df.iloc[0, 0]
            elif original.shape.is_columnar():
                return df.iloc[:, 0]
            else:
                raise TypeError(f"Unexpected shape: {original.shape}")
        else:
            return df
