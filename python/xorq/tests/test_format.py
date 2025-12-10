import re

import pytest

import xorq.api as xo
from xorq.caching import (
    ParquetCache,
    ParquetSnapshotCache,
    SourceCache,
    SourceSnapshotCache,
)


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
        pytest.param(ParquetCache(), "modification_time", True, id="parquet_storage"),
        pytest.param(
            ParquetSnapshotCache(), "snapshot", True, id="parquet_snapshot_storage"
        ),
        pytest.param(SourceCache(), "modification_time", False, id="source_storage"),
        pytest.param(
            SourceSnapshotCache(), "snapshot", False, id="source_snapshot_storage"
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
