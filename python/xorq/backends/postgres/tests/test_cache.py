import pytest

import xorq as xo
from xorq.caching import SourceStorage


@pytest.mark.parametrize(
    "name",
    (
        "diamonds",
        "hn-data-small.parquet",
    ),
)
def test_source_caching(name, pg):
    con = xo.connect()
    example = xo.examples.get_table_from_name(name, con)
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
