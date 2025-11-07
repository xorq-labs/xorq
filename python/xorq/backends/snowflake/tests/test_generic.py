import pytest

import xorq.api as xo
import xorq.expr.datatypes as dt
from xorq.backends.snowflake.tests.conftest import inside_temp_schema
from xorq.tests.util import assert_series_equal
from xorq.vendor.ibis import literal as L


@pytest.mark.snowflake
def test_create_table_from_expr_success(sf_con, temp_catalog, temp_db, parquet_dir):
    with inside_temp_schema(sf_con, temp_catalog, temp_db):
        name = "functional_alltypes"
        alltypes = (
            xo.deferred_read_parquet(
                parquet_dir.joinpath(f"{name}.parquet"), table_name=name
            )
            .limit(1000)
            .into_backend(sf_con)
        )

        import hashlib

        for how in ("sha256", "sha512", "md5"):

            def hashing(col):
                return getattr(hashlib, how)(col.encode()).hexdigest()

            h1 = alltypes.order_by("id").string_col.hexdigest(how=how).execute(limit=10)
            df = alltypes.order_by("id").execute(limit=10)
            h2 = df["string_col"].apply(hashing).rename("HexDigest(string_col)")

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
        vals = list("abc")

        expr = s.find_in_set(vals)
        assert not expr.execute().empty
