import pytest

import xorq.api as xo
from xorq.backends.snowflake.tests.conftest import inside_temp_schema
from xorq.tests.util import assert_series_equal


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
