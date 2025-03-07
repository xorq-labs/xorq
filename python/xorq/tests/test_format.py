import re

import pytest

import xorq as xo
from xorq import _
from xorq.caching import (
    ParquetSnapshotStorage,
    ParquetStorage,
    SourceSnapshotStorage,
    SourceStorage,
)
from xorq.common.utils.defer_utils import deferred_read_csv


def test_into_backend(batting):
    xo.options.interactive = False

    ddb_con = xo.duckdb.connect()

    t = batting.filter(batting.yearID == 2015).into_backend(ddb_con, "ls_batting")

    expr = (
        t.join(t, "playerID")
        .limit(15)
        .select(player_id="playerID", year_id="yearID_right")
    )

    assert "RemoteTable[r1, name=ls_batting]" in repr(expr)


@pytest.mark.parametrize(
    "storage, strategy, parquet",
    [
        pytest.param(ParquetStorage(), "modification_time", True, id="parquet_storage"),
        pytest.param(
            ParquetSnapshotStorage(), "snapshot", True, id="parquet_snapshot_storage"
        ),
        pytest.param(SourceStorage(), "modification_time", False, id="source_storage"),
        pytest.param(
            SourceSnapshotStorage(), "snapshot", False, id="source_snapshot_storage"
        ),
    ],
)
def test_cache(batting, storage, strategy, parquet):
    xo.options.interactive = False

    expr = (
        batting.join(batting, "playerID")
        .limit(15)
        .select(player_id="playerID", year_id="yearID_right")
        .cache(storage)
    )

    pattern = rf"CachedNode\[r\d+, strategy={strategy}, parquet={parquet}, source"

    assert re.search(pattern, repr(expr))


def test_read():
    xo.options.interactive = False

    csv_name = "iris"
    csv_path = xo.options.pins.get_path(csv_name)

    # we can work with a pandas expr without having read it yet
    pd_con = xo.pandas.connect()

    expr = deferred_read_csv(con=pd_con, path=csv_path, table_name=csv_name).filter(
        _.sepal_length > 6
    )

    pattern = r"Read\[name=iris, method_name=read_csv, source=pandas-\d+\]"

    assert re.search(pattern, repr(expr))
