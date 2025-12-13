import dask
import pandas as pd

import xorq.api as xo
from xorq.backends.conftest import KEY_PREFIX, get_cache_uncached
from xorq.caching import SourceCache, SourceSnapshotCache
from xorq.expr.relations import into_backend
from xorq.vendor import ibis


def test_pandas_snapshot(xo_con, alltypes_df):
    group_by = "year"
    name = ibis.util.gen_name("tmp_table")

    # create a temp table we can mutate
    pd_con = xo.pandas.connect()
    table = pd_con.create_table(name, alltypes_df)

    cached_expr = (
        table.group_by(group_by)
        .agg({f"count_{col}": table[col].count() for col in table.columns})
        .pipe(into_backend, xo_con)
        .cache(cache=SourceSnapshotCache.from_kwargs(source=xo_con))
    )
    (cache, uncached) = get_cache_uncached(cached_expr)

    # test preconditions
    assert not cache.exists(uncached)

    # test cache creation
    executed0 = cached_expr.execute()

    with cache.strategy.normalization_context(uncached):
        normalized0 = dask.base.normalize_token(uncached)
    assert cache.exists(uncached)

    # test cache use
    executed1 = cached_expr.execute()
    assert executed0.equals(executed1)

    # test NO cache invalidation
    pd_con.reconnect()
    table2 = pd_con.create_table(name, pd.concat((alltypes_df, alltypes_df)))

    cached_expr = (
        table2.group_by(group_by)
        .agg({f"count_{col}": table2[col].count() for col in table2.columns})
        .pipe(into_backend, xo_con)
        .cache(cache)
    )
    (cache, uncached) = get_cache_uncached(cached_expr)
    with cache.strategy.normalization_context(uncached):
        normalized1 = dask.base.normalize_token(uncached)

    # everything else is stable, despite the different data
    assert normalized0[1][1] == normalized1[1][1]
    assert cache.exists(uncached)
    assert cache.calc_key(uncached).count(KEY_PREFIX) == 1
    executed2 = cached_expr.ls.uncached.execute()
    assert not executed0.equals(executed2)


def test_caching_pandas(csv_dir):
    diamonds_path = csv_dir / "diamonds.csv"
    pandas_con = xo.pandas.connect()
    cache = SourceCache.from_kwargs(source=pandas_con)
    t = pandas_con.read_csv(diamonds_path).cache(cache=cache)
    assert t.execute() is not None
