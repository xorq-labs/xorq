import pytest

import xorq.api as xo
from xorq.backends._lazy import LazyBackend
from xorq.backends.snowflake.tests.conftest import inside_temp_schema
from xorq.vendor.ibis.backends.profiles import Profile


SU = pytest.importorskip("xorq.common.utils.snowflake_utils")


@pytest.mark.snowflake
def test_into_backend_from_xorq_lazy_snowflake(
    sf_con, temp_catalog, temp_db, parquet_dir
):
    """into_backend to a lazy snowflake backend works end-to-end.

    Uses inside_temp_schema so writes go to a writable schema rather than the
    read-only SNOWFLAKE_SAMPLE_DATA shared database.
    """
    source_con = xo.connect()
    parquet_path = parquet_dir / "awards_players.parquet"
    source_expr = source_con.read_parquet(parquet_path, table_name="awards_players")

    profile = Profile.from_con(sf_con)
    lazy_con = profile.get_con(lazy=True)

    assert isinstance(lazy_con, LazyBackend)
    assert not lazy_con.is_connected

    with inside_temp_schema(lazy_con, temp_catalog, temp_db):
        result = source_expr.into_backend(lazy_con).execute()

    assert lazy_con.is_connected
    assert len(result) > 0
