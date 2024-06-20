import pathlib
import re

import dask
import ibis
import pytest

import letsql.common.utils.dask_normalize  # noqa: F401
from letsql.common.caching import (
    SnapshotStorage,
)
from letsql.common.utils.dask_normalize import (
    patch_normalize_token,
)


def test_ensure_deterministic():
    assert dask.config.get("tokenize.ensure-deterministic")


def test_unregistered_rasies():
    class Unregistered:
        pass

    with pytest.raises(RuntimeError, match="cannot be deterministically hashed"):
        dask.base.tokenize(Unregistered())


def test_tokenize_datafusion_memory_expr(alltypes_df):
    con = ibis.datafusion.connect()
    typ = type(con)
    t = con.register(alltypes_df, "t")
    with patch_normalize_token(type(con)) as mocks:
        actual = dask.base.tokenize(t)
    mocks[typ].assert_called_once()
    expected = "c5c75c85e9998c1d8f2ed60c829f2f43"
    assert actual == expected


def test_tokenize_datafusion_parquet_expr(alltypes_df, tmp_path):
    path = pathlib.Path(tmp_path).joinpath("data.parquet")
    alltypes_df.to_parquet(path)
    con = ibis.datafusion.connect()
    t = con.register(path, "t")
    # work around tmp_path variation
    (prefix, suffix) = (
        re.escape(part)
        for part in (
            r"file_groups={1 group: [[",
            r"]]",
        )
    )
    to_hash = re.sub(
        prefix + f".*?/{path.name}" + suffix,
        prefix + f"/{path.name}" + suffix,
        str(tuple(dask.base.normalize_token(t))),
    )
    actual = dask.base._md5(to_hash.encode()).hexdigest()
    expected = "7685a5a58f3da4e8afd41b6a4a4f5790"
    assert actual == expected


def test_tokenize_pandas_expr(alltypes_df):
    con = ibis.pandas.connect()
    typ = type(con)
    t = con.create_table("t", alltypes_df)
    with patch_normalize_token(type(t.op().source)) as mocks:
        actual = dask.base.tokenize(t)
    mocks[typ].assert_called_once()
    expected = "bfa5bb55e47f2da9094d7aed9cee6130"
    assert actual == expected


def test_tokenize_duckdb_expr(batting):
    con = ibis.duckdb.connect()
    typ = type(con)
    t = con.register(batting.to_pyarrow(), "dashed-name")
    with patch_normalize_token(type(con)) as mocks:
        actual = dask.base.tokenize(t)
    mocks[typ].assert_called_once()
    expected = "26d37818c847a542b65d3e1455501e04"
    assert actual == expected


def test_pandas_snapshot_key(alltypes_df):
    con = ibis.pandas.connect()
    t = con.create_table("t", alltypes_df)
    storage = SnapshotStorage(source=con)
    actual = storage.get_key(t)
    expected = "snapshot-e8bbc4feaa5271ea2470140185beae96"
    assert actual == expected


def test_duckdb_snapshot_key(batting):
    con = ibis.duckdb.connect()
    t = con.register(batting.to_pyarrow(), "dashed-name")
    storage = SnapshotStorage(source=con)
    actual = storage.get_key(t)
    expected = "snapshot-e645278e370b6a79b62fd1865a77fff5"
    assert actual == expected
