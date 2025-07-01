import hashlib
import operator
import pathlib
import re

import dask
import pytest
import toolz

import xorq as xo
import xorq.common.utils.dask_normalize  # noqa: F401
from xorq.caching import (
    SourceSnapshotStorage,
)
from xorq.common.utils.dask_normalize import (
    get_normalize_token_subset,
)
from xorq.common.utils.dask_normalize.dask_normalize_utils import (
    file_digest,
    gen_batches,
    manual_file_digest,
    patch_normalize_token,
    walk_normalized,
)


def test_ensure_deterministic():
    assert dask.config.get("tokenize.ensure-deterministic")


def test_unregistered_raises():
    class Unregistered:
        pass

    with pytest.raises(ValueError, match="cannot be deterministically hashed"):
        dask.base.tokenize(Unregistered())


@pytest.mark.snapshot_check
def test_tokenize_datafusion_memory_expr(alltypes_df, snapshot):
    con = xo.datafusion.connect()
    typ = type(con)
    t = con.register(alltypes_df, "t")
    with patch_normalize_token(type(con)) as mocks:
        actual = dask.base.tokenize(t)
    mocks[typ].assert_not_called()
    snapshot.assert_match(actual, "datafusion_memory_key.txt")


@pytest.mark.snapshot_check
def test_tokenize_datafusion_parquet_expr(alltypes_df, tmp_path, snapshot):
    path = pathlib.Path(tmp_path).joinpath("data.parquet")
    alltypes_df.to_parquet(path)
    con = xo.datafusion.connect()
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
    actual = hashlib.md5(to_hash.encode(), usedforsecurity=False).hexdigest()
    snapshot.assert_match(actual, "datafusion_key.txt")


@pytest.mark.snapshot_check
def test_tokenize_pandas_expr(alltypes_df, snapshot):
    con = xo.pandas.connect()
    typ = type(con)
    t = con.create_table("t", alltypes_df)
    with patch_normalize_token(type(t.op().source)) as mocks:
        actual = dask.base.tokenize(t)
    mocks[typ].assert_not_called()
    snapshot.assert_match(actual, "pandas_key.txt")


@pytest.mark.snapshot_check
def test_tokenize_duckdb_expr(batting, snapshot):
    con = xo.duckdb.connect()
    typ = type(con)
    t = con.register(batting.to_pyarrow(), "dashed-name")
    with patch_normalize_token(type(con)) as mocks:
        actual = dask.base.tokenize(t)
    mocks[typ].assert_not_called()

    snapshot.assert_match(actual, "duckdb_key.txt")


@pytest.mark.snapshot_check
def test_pandas_snapshot_key(alltypes_df, snapshot):
    con = xo.pandas.connect()
    t = con.create_table("t", alltypes_df)
    storage = SourceSnapshotStorage(source=con)
    actual = storage.get_key(t)
    snapshot.assert_match(actual, "pandas_snapshot_key.txt")


@pytest.mark.snapshot_check
def test_duckdb_snapshot_key(batting, snapshot):
    con = xo.duckdb.connect()
    t = con.register(batting.to_pyarrow(), "dashed-name")
    storage = SourceSnapshotStorage(source=con)
    actual = storage.get_key(t)
    snapshot.assert_match(actual, "duckdb_snapshot_key.txt")


@pytest.mark.parametrize(
    "target, expected",
    (
        (1, 1),
        (1.0, 1),
        ("two", 2),
        # bytes are converted to text
        (b"three", 0),
        ("three", 3),
    ),
)
def test_walk_normalized(target, expected):
    normalized = (
        [
            (
                b"three",
                b"three",
                1,
                [
                    "two",
                ],
            ),
        ],
        (
            2,
            (
                3,
                b"three",
            ),
            "two",
        ),
    )
    f = toolz.curry(operator.eq, target)
    actual = sum(walk_normalized(f, normalized))
    assert actual == expected


def test_normalize_token_lookup():
    assert not get_normalize_token_subset()


def test_dask_tokenize_object():
    class MissingDaskTokenize:
        pass

    tokenize_value = 1

    class HasDaskTokenize:
        def __dask_tokenize__(self):
            return tokenize_value

    assert dask.base.normalize_token(HasDaskTokenize()) == tokenize_value
    with pytest.raises(ValueError):
        dask.base.tokenize(MissingDaskTokenize())


def test_partitioning():
    path = pathlib.Path(xo.config.options.pins.get_path("batting"))
    assert len(tuple(gen_batches(path))) > 1
    content = b"".join(gen_batches(path))
    assert content == path.read_bytes()


def test_file_digest():
    path = pathlib.Path(xo.config.options.pins.get_path("batting"))
    actual = manual_file_digest(path)
    expected = file_digest(path)
    assert actual == expected
