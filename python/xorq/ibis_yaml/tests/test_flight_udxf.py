import operator

import pandas as pd
import pytest
import toolz

import xorq as xo
from xorq.caching import ParquetStorage
from xorq.expr.udf import make_pandas_udf
from xorq.ibis_yaml.compiler import YamlExpressionTranslator
from xorq.tests.util import assert_frame_equal


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
    compiler = YamlExpressionTranslator()
    yaml_dict = compiler.to_yaml(expr)

    diamonds_con = diamonds._find_backend()

    profiles = {
        con._profile.hash_name: con,
        diamonds_con._profile.hash_name: diamonds_con,
    }
    roundtrip_expr = compiler.from_yaml(yaml_dict, profiles)

    df = expr.execute()
    roundtrip_df = roundtrip_expr.execute()
    pd.testing.assert_frame_equal(
        df,
        roundtrip_df,
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
        make_udxf_kwargs={"name": my_udf.__name__},
    ).order_by("cut")

    compiler = YamlExpressionTranslator()
    yaml_dict = compiler.to_yaml(expr)

    diamonds_con = diamonds._find_backend()

    profiles = {
        con._profile.hash_name: con,
        diamonds_con._profile.hash_name: diamonds_con,
    }
    roundtrip_expr = compiler.from_yaml(yaml_dict, profiles)

    df = expr.execute()

    roundtrip_df = roundtrip_expr.execute()

    pd.testing.assert_frame_equal(
        df,
        roundtrip_df,
        check_exact=False,
    )


def test_flight_udxf_cached(con, diamonds, baseline):
    from xorq.common.utils.graph_utils import find_all_sources

    input_expr = diamonds.pipe(do_agg)
    process_df = operator.methodcaller("assign", **{field_name: my_udf.fn})
    maybe_schema_in = input_expr.schema()
    maybe_schema_out = xo.schema(input_expr.schema() | {field_name: return_type})

    udxf = xo.expr.relations.flight_udxf(
        process_df=process_df,
        maybe_schema_in=maybe_schema_in,
        maybe_schema_out=maybe_schema_out,
        con=con,
        make_udxf_kwargs={"name": my_udf.__name__},
    )

    raw_expr = input_expr.pipe(udxf)

    ddb_con = xo.duckdb.connect()
    expr = (
        raw_expr.filter(xo._.cut.notnull())
        .cache(storage=ParquetStorage(ddb_con))
        .filter(~xo._.cut.contains("ERROR"))
        .order_by("cut")
    )

    compiler = YamlExpressionTranslator()
    yaml_dict = compiler.to_yaml(expr)

    profiles = {con._profile.hash_name: con for con in find_all_sources(expr)}
    roundtrip_expr = compiler.from_yaml(yaml_dict, profiles)

    expected = expr.execute()

    actual = roundtrip_expr.execute()

    assert_frame_equal(
        expected,
        actual,
        check_exact=False,
    )
