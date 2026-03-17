import pytest

import xorq.api as xo
from xorq.backends._lazy import LazyBackend
from xorq.vendor.ibis.backends.profiles import Profile


SU = pytest.importorskip("xorq.common.utils.snowflake_utils")


def test_into_backend_from_xorq_lazy_snowflake(parquet_dir):
    """into_backend to a lazy snowflake backend works end-to-end."""
    source_con = xo.connect()
    parquet_path = parquet_dir / "awards_players.parquet"
    source_expr = source_con.read_parquet(parquet_path, table_name="awards_players")

    sf_con = xo.snowflake.connect_env_keypair(
        database=SU.default_database,
        schema=SU.default_schema,
        create_object_udfs=False,
    )
    profile = Profile.from_con(sf_con)
    lazy_con = profile.get_con(lazy=True)

    assert isinstance(lazy_con, LazyBackend)
    assert not lazy_con.is_connected

    result = source_expr.into_backend(lazy_con).execute()

    assert lazy_con.is_connected
    assert len(result) > 0
