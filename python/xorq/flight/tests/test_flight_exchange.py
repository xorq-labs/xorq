import functools
import operator

import pandas as pd
import toolz

import xorq as xo
from xorq.expr.udf import make_pandas_udf


name = "price_per_carat"


@make_pandas_udf(
    schema=xo.schema({"sum_price": "float64", "sum_carat": "float64"}),
    return_type=xo.dtype("float64"),
    name="my_udf",
)
def my_udf(df):
    return df["sum_carat"].div(df["sum_price"])


def do_agg(expr):
    return expr.group_by("cut").agg(
        xo._.price.sum().name("sum_price"),
        xo._.carat.sum().name("sum_carat"),
        xo._.color.nunique().name("nunique_color"),
    )


my_udf_on_expr = toolz.compose(operator.methodcaller("name", name), my_udf.on_expr)


@functools.cache
def calc_baseline():
    con = xo.connect()
    input_expr = xo.examples.diamonds.fetch(backend=con)
    expr = input_expr.pipe(do_agg).mutate(my_udf_on_expr).order_by("cut")
    return expr.execute()


def test_flight_operator():
    con = xo.connect()
    input_expr = diamonds = xo.examples.diamonds.fetch(backend=con)
    unbound_expr = (
        xo.table(diamonds.schema()).pipe(do_agg).mutate(my_udf_on_expr).order_by("cut")
    )
    expr = xo.expr.relations.flight_operator(
        input_expr,
        unbound_expr,
        inner_name="flight-expr",
        name="remote-expr",
        con=con,
    )
    df = expr.execute()
    pd.testing.assert_frame_equal(
        calc_baseline().sort_values("cut", ignore_index=True),
        df.sort_values("cut", ignore_index=True),
        check_exact=False,
    )


def test_flight_udxf():
    con = xo.connect()
    input_expr = xo.examples.diamonds.fetch(backend=con).pipe(do_agg)
    expr = xo.expr.relations.flight_udxf(
        input_expr,
        my_udf,
        col_name=name,
        inner_name="flight-expr",
        name="remote-expr",
        con=con,
        do_instrument_reader=True,
    ).order_by("cut")
    df = expr.execute()
    actual = df.sort_values("cut", ignore_index=True)
    expected = calc_baseline().sort_values("cut", ignore_index=True)
    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_exact=False,
    )
