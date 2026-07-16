from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

import xorq.api as xo
from xorq.caching import (
    ParquetCache,
    ParquetSnapshotCache,
    SourceCache,
    SourceSnapshotCache,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from xorq.backends.bigquery import Backend


# the google client libraries are an optional (`--extra bigquery`) dependency
pytest.importorskip("google.cloud.bigquery")


@pytest.mark.bigquery
@pytest.mark.parametrize(
    "cls",
    (
        pytest.param(ParquetSnapshotCache, id="parquet-snapshot"),
        pytest.param(ParquetCache, id="parquet"),
        pytest.param(SourceSnapshotCache, id="source-snapshot"),
        pytest.param(SourceCache, id="source"),
    ),
)
def test_cache_find_backend(
    cls: type, con: Backend, con_cache_find_backend: Callable[..., None]
) -> None:
    con_cache_find_backend(cls, con)


@pytest.mark.bigquery
def test_source_snapshot_cache_roundtrip(con: Backend) -> None:
    # exercises the full tokenize path (the bigquery __TABLES__ normalizer)
    # plus a cross-source cache round-trip into duckdb
    df = pd.DataFrame({"key": list("abc"), "value": [1, 2, 3]})
    name = "cache_roundtrip_src"
    con.create_table(name, obj=df, overwrite=True)
    try:
        table = con.table(name)
        uncached = table.group_by("key").agg(total=table.value.sum())
        cached_expr = uncached.cache(
            SourceSnapshotCache.from_kwargs(source=xo.duckdb.connect())
        )

        (cache, uncached_op) = (cached_expr.ls.cache, cached_expr.ls.uncached_one)
        assert not cache.exists(uncached_op)

        # cache creation
        executed0 = cached_expr.execute()
        assert not executed0.empty
        assert cache.exists(uncached_op)

        # cache use
        executed1 = cached_expr.execute()
        assert executed0.equals(executed1)
    finally:
        con.drop_table(name, force=True)
