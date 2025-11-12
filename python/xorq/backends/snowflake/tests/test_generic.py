import hashlib

import pytest

import xorq.api as xo
import xorq.expr.datatypes as dt
from xorq.backends.snowflake.tests.conftest import inside_temp_schema
from xorq.tests.util import assert_series_equal
from xorq.vendor.ibis import literal as L


def calc_hexdigest(string, how):
    return hashlib.new(how, string.encode()).hexdigest()


@pytest.mark.snowflake
def test_create_table_from_expr_success(sf_con, temp_catalog, temp_db, parquet_dir):
    with inside_temp_schema(sf_con, temp_catalog, temp_db):
        name = "functional_alltypes"
        col_name, by = "string_col", "id"
        alltypes = (
            xo.deferred_read_parquet(
                parquet_dir.joinpath(f"{name}.parquet"), table_name=name
            )
            .limit(1000)
            .into_backend(sf_con)
        )

        expr = alltypes.order_by(by).limit(10)[col_name]
        for how in ("sha256", "sha512", "md5"):
            h1 = expr.hexdigest(how=how).execute()
            h2 = expr.execute().apply(calc_hexdigest, how=how).rename(h1.name)
            assert_series_equal(h1, h2)


@pytest.mark.parametrize(
    ("raw_value", "expected"), [("a", 0), ("b", 1), ("d", -1), (None, 3)]
)
def test_find_in_set(sf_con, raw_value, expected):
    value = L(raw_value, dt.string)
    haystack = ["a", "b", "c", None]
    expr = value.find_in_set(haystack)
    assert sf_con.execute(expr) == expected


def test_string_column_find_in_set(sf_con, temp_catalog, temp_db, parquet_dir):
    with inside_temp_schema(sf_con, temp_catalog, temp_db):
        name = "functional_alltypes"
        alltypes = (
            xo.deferred_read_parquet(
                parquet_dir.joinpath(f"{name}.parquet"), table_name=name
            )
            .limit(1000)
            .into_backend(sf_con)
        )

        s = alltypes.string_col
        vals = "312"

        actual = s.find_in_set(list(vals)).execute().to_list()
        expected = [vals.find(v) for v in s.execute()]

        assert actual == expected
