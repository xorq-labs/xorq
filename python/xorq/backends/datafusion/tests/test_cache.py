import pytest

import xorq.api as xo
from xorq.backends.conftest import get_storage_uncached
from xorq.caching import (
    ParquetSnapshotStorage,
    ParquetStorage,
    SourceSnapshotStorage,
    SourceStorage,
)
from xorq.vendor import ibis


KEY_PREFIX = xo.config.options.cache.key_prefix


@pytest.mark.parametrize(
    "get_expr",
    [
        lambda t: t,
        lambda t: t.group_by("playerID").agg(t.stint.max().name("n-stints")),
    ],
)
def test_register_with_different_name_and_cache(csv_dir, get_expr):
    batting_path = csv_dir.joinpath("batting.csv")
    table_name = "batting"

    datafusion_con = xo.datafusion.connect()
    xorq_table_name = f"{datafusion_con.name}_{table_name}"
    t = datafusion_con.read_csv(
        batting_path, table_name=table_name, schema_infer_max_records=50_000
    )
    expr = t.pipe(get_expr).cache()

    assert table_name != xorq_table_name
    assert expr.execute() is not None


def test_datafusion_snapshot(ls_con, alltypes_df):
    group_by = "year"
    name = ibis.util.gen_name("tmp_table")

    # create a temp table we can mutate
    df_con = xo.datafusion.connect()
    table = df_con.create_table(name, alltypes_df)

    cached_expr = (
        table.group_by(group_by)
        .agg({f"count_{col}": table[col].count() for col in table.columns})
        .cache(storage=SourceSnapshotStorage(source=ls_con))
    )
    (storage, uncached) = get_storage_uncached(cached_expr)

    # test preconditions
    assert not storage.exists(uncached)

    # test cache creation
    executed0 = cached_expr.execute()
    assert storage.exists(uncached)

    # test cache use
    executed1 = cached_expr.execute()
    assert executed0.equals(executed1)

    # test NO cache invalidation
    df_con.insert(name, alltypes_df)
    executed2 = cached_expr.execute()
    executed3 = cached_expr.ls.uncached.execute()
    assert executed0.equals(executed2)
    assert not executed0.equals(executed3)
    assert storage.get_key(uncached).count(KEY_PREFIX) == 1


def test_cross_source_snapshot(ls_con, alltypes_df):
    expr_con = xo.datafusion.connect()
    group_by = "year"
    name = ibis.util.gen_name("tmp_table")

    # create a temp table we can mutate
    table = expr_con.create_table(name, alltypes_df)

    storage = ParquetSnapshotStorage(source=ls_con)

    expr = table.group_by(group_by).agg(
        {f"count_{col}": table[col].count() for col in table.columns}
    )

    cached_expr = expr.cache(storage=storage)

    # test preconditions
    assert not storage.exists(expr)  # the expr is not cached
    assert storage.source is not expr_con  # the cache is cross source

    # test cache creation
    df = cached_expr.execute()

    assert not df.empty
    assert storage.exists(expr)

    # test cache use
    executed1 = cached_expr.execute()
    assert df.equals(executed1)

    # test NO cache invalidation
    expr_con.insert(name, alltypes_df)
    executed2 = cached_expr.execute()
    executed3 = cached_expr.ls.uncached.execute()
    assert df.equals(executed2)
    assert not df.equals(executed3)


@pytest.mark.parametrize(
    "cls",
    [ParquetSnapshotStorage, ParquetStorage, SourceSnapshotStorage, SourceStorage],
)
def test_cache_find_backend(cls, parquet_dir):
    con = xo.datafusion.connect()
    astronauts_path = parquet_dir / "astronauts.parquet"
    storage = cls(source=con)
    expr = con.read_parquet(astronauts_path).cache(storage=storage)
    assert expr._find_backend()._profile == con._profile
