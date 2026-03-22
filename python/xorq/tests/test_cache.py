import pandas as pd
import pytest

import xorq.api as xo
from xorq.caching import ParquetCache, SourceCache
from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.relations import RemoteTable
from xorq.tests.util import assert_frame_equal


pytestmark = pytest.mark.postgres


def test_cross_source_cache(pg):
    name = "astronauts"
    expr = (
        xo.duckdb.connect()
        .create_table(name, pg.table(name).to_pyarrow())[lambda t: t.number > 22]
        .cache(cache=SourceCache.from_kwargs(source=pg))
    )
    expr.execute()


def test_caching_of_registered_arbitrary_expression(con, pg, tmp_path):
    table_name = "batting"
    t = pg.table(table_name)

    expr = t.filter(t.playerID == "allisar01")[
        ["playerID", "yearID", "stint", "teamID", "lgID"]
    ]
    expected = expr.execute()

    result = expr.cache(
        cache=ParquetCache.from_kwargs(source=con, relative_path=tmp_path)
    ).execute()

    assert result is not None
    assert_frame_equal(result, expected, check_like=True)


def test_cache_two_same_type_backends_creates_remote_table():
    con1 = xo.connect()
    con2 = xo.connect()
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    t = con1.create_table("t", df, overwrite=True)
    cached = t.cache(cache=SourceCache.from_kwargs(source=con2))
    assert any(walk_nodes(RemoteTable, cached))


def test_cache_inmemory_table_no_remote_table():
    ddb_con = xo.duckdb.connect()
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    t = xo.memtable(df)
    expr = t.filter(t.a > 1).select("a")
    cached = expr.cache(cache=SourceCache.from_kwargs(source=ddb_con))
    assert not any(walk_nodes(RemoteTable, cached))


def test_cache_record_batch_provider_exec(pg):
    batches = pg.table("batting").to_pyarrow_batches()
    t = (ls_con := xo.connect()).register(batches, table_name="batting_batches")
    cache = SourceCache.from_kwargs(source=ls_con)

    assert cache.calc_key(t) is not None
