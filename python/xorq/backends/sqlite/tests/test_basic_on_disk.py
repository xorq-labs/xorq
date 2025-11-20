import xorq.api as xo
from xorq.caching import SourceStorage
from xorq.tests.util import assert_frame_equal


def test_read_parquet(persistent_sqlite_con, astronauts_parquet_path):
    t = persistent_sqlite_con.read_parquet(astronauts_parquet_path)
    assert not t.execute().empty


def test_can_be_cached(persistent_sqlite_con, astronauts_parquet_path):
    astronauts = persistent_sqlite_con.read_parquet(astronauts_parquet_path)
    expr = (
        astronauts.cache(SourceStorage(xo.duckdb.connect()))
        .filter(xo._.number == 104)
        .select(xo._.id, xo._.number, xo._.nationwide_number, xo._.name)
    )

    actual = expr.execute()
    expected = (
        astronauts.filter(xo._.number == 104)
        .select(xo._.id, xo._.number, xo._.nationwide_number, xo._.name)
        .execute()
    )

    assert_frame_equal(expected, actual)
