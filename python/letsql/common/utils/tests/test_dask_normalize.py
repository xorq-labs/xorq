import pathlib
import re

import dask
import ibis
import letsql.common.utils.dask_normalize  # noqa: F401
import pytest


def test_ensure_deterministic():
    assert dask.config.get("tokenize.ensure-deterministic")


def test_unregistered_rasies():
    class Unregistered:
        pass

    with pytest.raises(RuntimeError, match="cannot be deterministically hashed"):
        dask.base.tokenize(Unregistered())


def test_tokenize_datafusion_memory_expr(alltypes_df):
    con = ibis.datafusion.connect()
    t = con.register(alltypes_df, "t")
    actual = dask.base.tokenize(t)
    expected = "e5feefe7661d275607da0e0a089e2c3e"
    assert actual == expected


def test_tokenize_datafusion_parquet_expr(alltypes_df, tmp_path):
    path = pathlib.Path(tmp_path).joinpath("data.parquet")
    alltypes_df.to_parquet(path)
    t = ibis.datafusion.connect({"t": path}).table("t")
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
    t = ibis.pandas.connect({"t": alltypes_df})
    actual = dask.base.tokenize(t)
    expected = "7b0019049171a3ef78ecbd5f463ac728"
    assert actual == expected
