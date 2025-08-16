import pytest

import xorq as xo
from xorq.caching import SourceStorage


@pytest.mark.parametrize(
    "name",
    (
        "diamonds",
        "astronauts",
    ),
)
def test_source_caching(name, pg, parquet_dir):
    con = xo.connect()
    example = xo.deferred_read_parquet(parquet_dir / f"{name}.parquet", con)
    expr = example.cache(storage=SourceStorage(pg))
    assert not expr.ls.exists()
    actual = expr.execute()
    expected = example.execute()
    cached = pg.table(expr.ls.get_key()).execute()
    assert actual.equals(expected)
    assert actual.equals(cached)
    assert expr.ls.exists()
    pg.drop_table(expr.ls.get_key())
    assert not expr.ls.exists()
