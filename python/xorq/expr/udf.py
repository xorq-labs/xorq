import functools
from typing import Any

import cloudpickle
import pandas as pd
import pyarrow as pa
import toolz

import xorq.expr.datatypes as dt
import xorq.vendor.ibis.expr.rules as rlz
import xorq.vendor.ibis.expr.types as ir
from xorq.vendor.ibis.common.annotations import Argument
from xorq.vendor.ibis.common.collections import FrozenDict
from xorq.vendor.ibis.expr.operations import Namespace
from xorq.vendor.ibis.expr.operations.udf import (
    AggUDF,
    InputType,
    ScalarUDF,
    _make_udf_name,
    _wrap,
    scalar,
)
from xorq.vendor.ibis.expr.operations.udf import (
    agg as _agg,
)


def property_wrap_fn(fn):
    return property(fget=lambda _, fn=fn: fn)


def arrays_to_df(names, *arrays):
    return pd.DataFrame(
        {name: array.to_pandas() for (name, array) in zip(names, arrays)}
    )


def make_pyarrow_array(return_type, series):
    return pa.Array.from_pandas(series, type=return_type.to_pyarrow())


def make_dunder_func(fn, schema, return_type=None):
    def fn_from_arrays(*arrays):
        df = arrays_to_df(schema, *arrays)
        value = fn(df)
        if return_type is not None:
            value = make_pyarrow_array(return_type, value)
        return value

    return property_wrap_fn(fn_from_arrays)


def make_expr_scalar_udf_dunder_func(fn, schema, return_type):
    def fn_from_arrays(*arrays, computed_arg=None, **kwargs):
        if computed_arg is None:
            raise ValueError(
                "Caller must bind computed_arg to the output of computed_kwargs_expr"
            )
        df = arrays_to_df(schema, *arrays)
        value = fn(computed_arg, df, **kwargs)
        return make_pyarrow_array(
            return_type,
            value,
        )

    return property_wrap_fn(fn_from_arrays)


@toolz.curry
def wrap_model(value, model_key="model"):
    return cloudpickle.dumps({model_key: value})


unwrap_model = cloudpickle.loads


class ExprScalarUDF(ScalarUDF):
    @property
    def computed_kwargs_expr(self):
        # must push the expr into __config__ so that it doesn't get turned into a window function
        return self.__config__["computed_kwargs_expr"]

    @property
    def post_process_fn(self):
        return self.__config__["post_process_fn"]

    @property
    def schema(self):
        return self.__config__["schema"]

    def on_expr(self, e, **kwargs):
        # rebind deferred_model (computed_kwargs_expr) to a new expr
        return type(self)(*(e[c] for c in self.schema), **kwargs)


@toolz.curry
def make_pandas_expr_udf(
    computed_kwargs_expr,
    fn,
    schema,
    return_type=dt.binary,
    database=None,
    catalog=None,
    name=None,
    *,
    post_process_fn=unwrap_model,
    **kwargs,
):
    name = name if name is not None else _make_udf_name(fn)
    bases = (ExprScalarUDF,)
    fields = {
        arg_name: Argument(pattern=rlz.ValueOf(typ), typehint=typ)
        for (arg_name, typ) in schema.items()
    }
    meta = {
        "dtype": return_type,
        "__input_type__": InputType.PYARROW,
        "__func__": make_expr_scalar_udf_dunder_func(fn, schema, return_type),
        # valid config keys: computed_kwargs_expr, post_process_fn, volatility
        "__config__": FrozenDict(
            computed_kwargs_expr=computed_kwargs_expr,
            post_process_fn=post_process_fn,
            schema=schema,
            fn=fn,
            **kwargs,
        ),
        "__udf_namespace__": Namespace(database=database, catalog=catalog),
        "__module__": fn.__module__,
        # FIXME: determine why this fails with case mismatch by default
        "__func_name__": name,
    }
    kwds = {
        **fields,
        **meta,
    }

    node = type(
        name,
        bases,
        kwds,
    )

    # FIXME: enable working with deferred like _wrap enables
    @functools.wraps(fn)
    def construct(*args: Any, **kwargs: Any) -> ir.Value:
        return node(*args, **kwargs).to_expr()

    def on_expr(e, **kwargs):
        return construct(*(e[c] for c in schema), **kwargs)

    construct.on_expr = on_expr
    construct.fn = fn
    return construct


@toolz.curry
def make_pandas_udf(
    fn, schema, return_type, database=None, catalog=None, name=None, **kwargs
):
    from xorq.vendor.ibis.expr.operations.udf import ScalarUDF

    name = name if name is not None else _make_udf_name(fn)
    bases = (ScalarUDF,)
    fields = {
        arg_name: Argument(pattern=rlz.ValueOf(typ), typehint=typ)
        for (arg_name, typ) in schema.items()
    }
    meta = {
        "dtype": return_type,
        "__input_type__": InputType.PYARROW,
        "__func__": make_dunder_func(fn, schema, return_type),
        # valid config keys: volatility
        "__config__": FrozenDict(**kwargs),
        "__udf_namespace__": Namespace(database=database, catalog=catalog),
        "__module__": fn.__module__,
        # FIXME: determine why this fails with case mismatch by default
        "__func_name__": name,
    }
    kwds = {
        **fields,
        **meta,
    }

    node = type(
        name,
        bases,
        kwds,
    )

    # FIXME: enable working with deferred like _wrap enables
    @functools.wraps(fn)
    def construct(*args: Any, **kwargs: Any) -> ir.Value:
        return node(*args, **kwargs).to_expr()

    def on_expr(e, **kwargs):
        return construct(*(e[c] for c in schema), **kwargs)

    construct.on_expr = on_expr
    construct.fn = fn
    return construct


class agg(_agg):
    __slots__ = ()

    _base = AggUDF

    @classmethod
    def pyarrow(
        cls,
        fn=None,
        name=None,
        signature=None,
        **kwargs,
    ):
        result = _wrap(
            cls._make_wrapper,
            InputType.PYARROW,
            fn,
            name=name,
            signature=signature,
            **kwargs,
        )
        return result

    @classmethod
    @toolz.curry
    def pandas_df(
        cls,
        fn,
        schema,
        return_type,
        database=None,
        catalog=None,
        name=None,
        **kwargs,
    ):
        name = name if name is not None else _make_udf_name(fn)
        bases = (cls._base,)
        fields = {
            arg_name: Argument(pattern=rlz.ValueOf(typ), typehint=typ)
            for (arg_name, typ) in schema.items()
        }
        meta = {
            "dtype": return_type,
            "__input_type__": InputType.PYARROW,
            "__func__": make_dunder_func(fn, schema),
            # valid config keys: volatility
            "__config__": FrozenDict(fn=fn, **kwargs),
            "__udf_namespace__": Namespace(database=database, catalog=catalog),
            "__module__": fn.__module__,
            # FIXME: determine why this fails with case mismatch by default
            "__func_name__": name,
        }
        kwds = {
            **fields,
            **meta,
        }

        node = type(
            name,
            bases,
            kwds,
        )

        # FIXME: enable working with deferred like _wrap enables
        @functools.wraps(fn)
        def construct(*args: Any, **kwargs: Any) -> ir.Value:
            return node(*args, **kwargs).to_expr()

        def on_expr(e, **kwargs):
            return construct(*(e[c] for c in schema), **kwargs)

        construct.on_expr = on_expr
        construct.fn = fn
        return construct


def arbitrate_evaluate(
    uses_window_frame=False,
    supports_bounded_execution=False,
    include_rank=False,
    **config_kwargs,
):
    match (uses_window_frame, supports_bounded_execution, include_rank):
        case (False, False, False):
            return "evaluate_all"
        case (False, True, False):
            return "evaluate"
        case (False, _, True):
            return "evaluate_all_with_rank"
        case (True, _, _):
            return "evaluate"
        case _:
            raise RuntimeError


@toolz.curry
def pyarrow_udwf(
    fn,
    schema,
    return_type,
    name=None,
    namespace=Namespace(database=None, catalog=None),
    base=AggUDF,
    **config_kwargs,
):
    fields = {
        arg_name: Argument(pattern=rlz.ValueOf(typ), typehint=typ)
        for (arg_name, typ) in schema.items()
    }
    # which_evaluate = arbitrate_evaluate(**config_kwargs)
    name = name or fn.__name__
    meta = {
        "dtype": return_type,
        "__input_type__": InputType.PYARROW,
        "__func__": property(fget=toolz.functoolz.return_none),
        "__config__": FrozenDict(
            input_types=tuple(datatype for datatype in schema.fields.values()),
            return_type=return_type,
            name=name,
            **config_kwargs,
            # assert which_evaluate in ("evaluate", "evaluate_all", "evaluate_all_with_rank")
            # **{which_evaluate: fn},
            **{
                which_evaluate: fn
                for which_evaluate in (
                    "evaluate",
                    "evaluate_all",
                    "evaluate_all_with_rank",
                )
            },
        ),
        "__udf_namespace__": namespace,
        "__module__": __name__,
        "__func_name__": name,
    }
    node = type(
        name,
        (base,),
        {
            **fields,
            **meta,
        },
    )

    def construct(*args: Any, **kwargs: Any) -> ir.Value:
        return node(*args, **kwargs).to_expr()

    def on_expr(e, **kwargs):
        return construct(*(e[c] for c in schema), **kwargs)

    construct.on_expr = on_expr
    return construct


__all__ = [
    "pyarrow_udwf",
    "make_pandas_expr_udf",
    "make_pandas_udf",
    "scalar",
    "agg",
]
