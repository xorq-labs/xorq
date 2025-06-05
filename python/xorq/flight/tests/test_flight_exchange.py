import operator

import pandas as pd
import pytest
import toolz

import xorq as xo
from xorq.expr.udf import make_pandas_udf


field_name = "price_per_carat"
schema = xo.schema({"sum_price": "float64", "sum_carat": "float64"})
return_type = xo.dtype("float64")


@pytest.fixture(scope="function")
def con():
    return xo.connect()


@pytest.fixture(scope="function")
def diamonds(con):
    return xo.examples.diamonds.fetch(backend=con)


@pytest.fixture(scope="function")
def baseline(diamonds):
    expr = diamonds.pipe(do_agg).mutate(my_udf_on_expr).order_by("cut")
    return expr.execute()


@make_pandas_udf(
    schema=schema,
    return_type=return_type,
    name="my_udf",
)
def my_udf(df):
    return df["sum_carat"].div(df["sum_price"])


def do_agg(expr):
    return (
        expr.group_by("cut")
        .agg(
            xo._.price.sum().name("sum_price"),
            xo._.carat.sum().name("sum_carat"),
            xo._.color.nunique().name("nunique_color"),
        )
        .cast({"sum_price": "float64"})
    )


my_udf_on_expr = toolz.compose(
    operator.methodcaller("name", field_name), my_udf.on_expr
)


def test_flight_expr(con, diamonds, baseline):
    unbound_expr = (
        xo.table(diamonds.schema()).pipe(do_agg).mutate(my_udf_on_expr).order_by("cut")
    )
    expr = xo.expr.relations.flight_expr(
        diamonds,
        unbound_expr,
        inner_name="flight-expr",
        name="remote-expr",
        con=con,
    )
    df = expr.execute()
    pd.testing.assert_frame_equal(
        baseline.sort_values("cut", ignore_index=True),
        df.sort_values("cut", ignore_index=True),
        check_exact=False,
    )


def test_flight_udxf(con, diamonds, baseline):
    input_expr = diamonds.pipe(do_agg)
    process_df = operator.methodcaller("assign", **{field_name: my_udf.fn})
    maybe_schema_in = input_expr.schema()
    maybe_schema_out = xo.schema(input_expr.schema() | {field_name: return_type})
    expr = xo.expr.relations.flight_udxf(
        input_expr,
        process_df=process_df,
        maybe_schema_in=maybe_schema_in,
        maybe_schema_out=maybe_schema_out,
        con=con,
        # operator.methodcaller doesn't have name, so must explicitly pass
        make_udxf_kwargs={"name": my_udf.__name__},
    ).order_by("cut")
    df = expr.execute()
    actual = df.sort_values("cut", ignore_index=True)
    expected = baseline.sort_values("cut", ignore_index=True)
    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_exact=False,
    )
