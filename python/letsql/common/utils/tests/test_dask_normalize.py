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


def test_unregistered_raises():
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
    mocks[typ].assert_not_called()
    expected = "534bc9e796f1c4e196948dbfb1105b5f"
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
    expected = "418032406d49e33664758f6bdc1806dd"
    assert actual == expected


def test_tokenize_pandas_expr(alltypes_df):
    con = ibis.pandas.connect()
    typ = type(con)
    t = con.create_table("t", alltypes_df)
    with patch_normalize_token(type(t.op().source)) as mocks:
        actual = dask.base.tokenize(t)
    mocks[typ].assert_not_called()
    expected = "79d57cb9ea641bbde700fef50b63a178"
    assert actual == expected


def test_tokenize_duckdb_expr(batting):
    con = letsql.duckdb.connect()
    typ = type(con)
    t = con.register(batting.to_pyarrow(), "dashed-name")
    with patch_normalize_token(type(con)) as mocks:
        actual = dask.base.tokenize(t)
    mocks[typ].assert_not_called()
    expected = "b03c2dc252e61b8a2273b6708e37bade"
    assert actual == expected


def test_pandas_snapshot_key(alltypes_df):
    con = ibis.pandas.connect()
    t = con.create_table("t", alltypes_df)
    storage = SnapshotStorage(source=con)
    actual = storage.get_key(t)
    expected = "snapshot-458282738391db238cd7b3462d40b13e"
    assert actual == expected


def test_duckdb_snapshot_key(batting):
    con = letsql.duckdb.connect()
    t = con.register(batting.to_pyarrow(), "dashed-name")
    storage = SnapshotStorage(source=con)
    actual = storage.get_key(t)
    expected = "snapshot-614c0fc08df80bf954862a79a63bc4ea"
    assert actual == expected
