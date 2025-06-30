import operator

import pytest
import toolz

import xorq as xo
from xorq.expr.udf import make_pandas_udf


field_name = "price_per_carat"
schema = xo.schema({"sum_price": "float64", "sum_carat": "float64"})
return_type = xo.dtype("float64")


@pytest.fixture(scope="session")
def tls_kwargs():
    return {
        "ca_kwargs": {
            "common_name": "root_cert",
        },
        "server_kwargs": {
            "common_name": "server",
            "sans": ("localhost",),
        },
    }


@pytest.fixture(scope="session")
def mtls_kwargs(tls_kwargs):
    return tls_kwargs | {
        "client_kwargs": {
            "common_name": "client",
        },
    }


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
