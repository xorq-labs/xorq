import pytest

import xorq.api as xo
from xorq.caching import (
    ParquetSnapshotStorage,
    ParquetStorage,
    SourceSnapshotStorage,
    SourceStorage,
)


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


def test_datafusion_snapshot(con_snapshot):
    con_snapshot(xo.datafusion.connect())


def test_cross_source_snapshot(con_cross_source_snapshot):
    con_cross_source_snapshot(xo.datafusion.connect())


@pytest.mark.parametrize(
    "cls",
    [ParquetSnapshotStorage, ParquetStorage, SourceSnapshotStorage, SourceStorage],
)
def test_cache_find_backend(cls, parquet_dir, con_cache_find_backend):
    con_cache_find_backend(cls, xo.datafusion.connect())
