from __future__ import annotations

import inspect
import pathlib
import uuid

import pyarrow as pa
import pyarrow.compute as pc
import pytest
import toolz

import xorq.api as xo
import xorq.expr.datatypes as dt
from xorq.caching import (
    ParquetSnapshotStorage,
    ParquetStorage,
    SourceSnapshotStorage,
    SourceStorage,
)
from xorq.common.utils.inspect_utils import get_python_version_no_dot
from xorq.expr.udf import (
    agg,
)
from xorq.tests.util import (
    assert_frame_equal,
)
from xorq.vendor.ibis import _
from xorq.vendor.ibis.expr.types.relations import CACHED_NODE_NAME_PLACEHOLDER


KEY_PREFIX = xo.config.options.cache.key_prefix


@pytest.fixture
def cached_two(ls_con, batting, tmp_path):
    parquet_storage = ParquetStorage(source=ls_con, relative_path=tmp_path)
    return (
        batting[lambda t: t.yearID > 2014]
        .cache()[lambda t: t.stint == 1]
        .cache(storage=parquet_storage)
    )


@pytest.fixture(scope="function")
def pg_alltypes(pg):
    return pg.table("functional_alltypes")


def test_cache_simple(con, alltypes, alltypes_df):
    initial_tables = con.list_tables()

    expr = alltypes.select(
        alltypes.smallint_col, alltypes.int_col, alltypes.float_col
    ).filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )
    cached = expr.cache(storage=SourceStorage(source=con))
    tables_after_caching = con.list_tables()

    expected = alltypes_df[
        (alltypes_df["float_col"] > 0)
        & (alltypes_df["smallint_col"] == 9)
        & (alltypes_df["int_col"] < alltypes_df["float_col"] * 2)
    ][["smallint_col", "int_col", "float_col"]]

    executed = cached.execute()
    tables_after_executing = con.list_tables()

    assert_frame_equal(executed, expected)
    assert not any(
        table_name.startswith(KEY_PREFIX)
        for table_name in set(tables_after_caching).difference(initial_tables)
    )
    assert any(
        table_name.startswith(KEY_PREFIX)
        for table_name in set(tables_after_executing).difference(initial_tables)
    )


def test_cache_multiple_times(con, alltypes, alltypes_df):
    expr = alltypes.select(
        alltypes.smallint_col, alltypes.int_col, alltypes.float_col
    ).filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )
    cached = expr.cache()

    # reassign the expression
    expr = alltypes.select(
        alltypes.smallint_col, alltypes.int_col, alltypes.float_col
    ).filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )

    re_cached = expr.cache()

    first = cached.execute()
    tables_after_first_caching = con.list_tables()

    second = re_cached.execute()
    tables_after_second_caching = con.list_tables()

    expected = alltypes_df[
        (alltypes_df["float_col"] > 0)
        & (alltypes_df["smallint_col"] == 9)
        & (alltypes_df["int_col"] < alltypes_df["float_col"] * 2)
    ][["smallint_col", "int_col", "float_col"]]

    assert_frame_equal(first, expected)
    assert_frame_equal(second, expected)

    first_tables = [t for t in tables_after_first_caching if t.startswith(KEY_PREFIX)]
    second_tables = [t for t in tables_after_second_caching if t.startswith(KEY_PREFIX)]

    assert sorted(first_tables) == sorted(second_tables)


def test_cache_to_sql(alltypes):
    expr = alltypes.select(
        alltypes.smallint_col, alltypes.int_col, alltypes.float_col
    ).filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )
    cached = expr.cache()

    assert CACHED_NODE_NAME_PLACEHOLDER in xo.to_sql(cached)


def test_op_after_cache(alltypes):
    expr = alltypes.select(alltypes.smallint_col, alltypes.int_col, alltypes.float_col)
    cached = expr.cache()
    cached = cached.filter(
        [
            _.float_col > 0,
            _.smallint_col == 9,
            _.int_col < _.float_col * 2,
        ]
    )

    full_expr = expr.filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )

    actual = cached.execute()
    expected = full_expr.execute()

    assert_frame_equal(actual, expected)

    assert CACHED_NODE_NAME_PLACEHOLDER in xo.to_sql(cached)


def test_cache_recreate(alltypes):
    def make_expr(alltypes):
        return alltypes.select(
            alltypes.smallint_col, alltypes.int_col, alltypes.float_col
        ).filter(
            [
                alltypes.float_col > 0,
                alltypes.smallint_col == 9,
                alltypes.int_col < alltypes.float_col * 2,
            ]
        )

    alltypes_df = alltypes.execute()
    cons = (con0, con1) = xo.connect(), xo.connect()
    ts = tuple(con.create_table("alltypes", alltypes_df) for con in cons)
    exprs = tuple(make_expr(t) for t in ts)

    for con, expr in zip(cons, exprs):
        # FIXME: execute one, simply check the other returns true for `expr.ls.exists()`
        expr.cache(storage=SourceStorage(source=con)).execute()

    (con_cached_tables0, con_cached_tables1) = (
        set(
            table_name
            for table_name in con.list_tables()
            if table_name.startswith(KEY_PREFIX)
        )
        for con in cons
    )

    assert con_cached_tables0
    assert con_cached_tables0 == con_cached_tables1
    for table_name in con_cached_tables1:
        assert_frame_equal(
            con0.table(table_name).to_pandas(), con1.table(table_name).to_pandas()
        )


def test_cache_execution(alltypes):
    cached = (
        alltypes.select(alltypes.smallint_col, alltypes.int_col, alltypes.float_col)
        .cache()
        .filter(
            [
                _.float_col > 0,
                _.smallint_col == 9,
                _.int_col < _.float_col * 2,
            ]
        )
        .select(_.int_col * 4)
        .cache()
    )

    actual = cached.execute()

    expected = (
        alltypes.select(alltypes.smallint_col, alltypes.int_col, alltypes.float_col)
        .filter(
            [
                alltypes.float_col > 0,
                alltypes.smallint_col == 9,
                alltypes.int_col < alltypes.float_col * 2,
            ]
        )
        .select(alltypes.int_col * 4)
        .execute()
    )

    assert_frame_equal(actual, expected)


def test_parquet_cache_storage(tmp_path, alltypes_df):
    tmp_path = pathlib.Path(tmp_path)
    path = tmp_path.joinpath("to-delete.parquet")

    con = xo.connect()
    alltypes_df.to_parquet(path)
    t = con.read_parquet(path, "t")
    cols = ["id", "bool_col", "float_col", "string_col"]
    expr = t[cols]
    expected = alltypes_df[cols]
    source = expr._find_backend()
    storage = ParquetStorage(
        source=source,
        relative_path=tmp_path.joinpath("parquet-cache-storage"),
    )
    cached = expr.cache(storage=storage)
    actual = cached.execute()
    assert_frame_equal(actual, expected)

    # the file must exist and have the same schema
    alltypes_df.head(1).to_parquet(path)
    actual = cached.execute()
    assert_frame_equal(actual, expected)

    path.unlink()
    pattern = "".join(
        (
            "Object Store error: Object at location",
            ".*",
            "not found: No such file or directory",
        )
    )
    with pytest.raises(Exception, match=pattern):
        # if the file doesn't exist, we get a failure, even for cached
        cached.execute()


def test_parquet_remote_to_local(con, alltypes, tmp_path):
    tmp_path = pathlib.Path(tmp_path)

    expr = alltypes.select(
        alltypes.smallint_col, alltypes.int_col, alltypes.float_col
    ).filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )
    storage = ParquetStorage(
        source=con,
        relative_path=tmp_path.joinpath("parquet-cache-storage"),
    )
    cached = expr.cache(storage=storage)
    expected = expr.execute()
    actual = cached.execute()
    assert_frame_equal(actual, expected)


def test_read_parquet_and_cache(con, parquet_dir, tmp_path):
    batting_path = parquet_dir / "batting.parquet"
    t = con.read_parquet(batting_path, table_name=f"parquet_batting-{uuid.uuid4()}")
    expr = t.cache(storage=ParquetStorage(source=con, relative_path=tmp_path))
    assert expr.execute() is not None


def test_read_parquet_and_cache_xorq(ls_con, parquet_dir, tmp_path):
    batting_path = parquet_dir / "batting.parquet"
    t = ls_con.read_parquet(batting_path, table_name=f"parquet_batting-{uuid.uuid4()}")
    expr = t.cache(storage=ParquetStorage(source=ls_con, relative_path=tmp_path))
    assert expr.execute() is not None


def test_read_parquet_compute_and_cache(con, parquet_dir, tmp_path):
    batting_path = parquet_dir / "batting.parquet"
    t = con.read_parquet(batting_path, table_name=f"parquet_batting-{uuid.uuid4()}")
    expr = (
        t[t.yearID == 2015]
        .cache(storage=ParquetStorage(source=con, relative_path=tmp_path))
        .cache()
    )
    assert expr.execute() is not None


def test_read_csv_and_cache(ls_con, csv_dir, tmp_path):
    batting_path = csv_dir / "batting.csv"
    t = ls_con.read_csv(batting_path, table_name=f"csv_batting-{uuid.uuid4()}")
    expr = t.cache(storage=ParquetStorage(source=ls_con, relative_path=tmp_path))
    assert expr.execute() is not None


def test_read_csv_compute_and_cache(ls_con, csv_dir, tmp_path):
    batting_path = csv_dir / "batting.csv"
    t = ls_con.read_csv(
        batting_path,
        table_name=f"csv_batting-{uuid.uuid4()}",
        schema_infer_max_records=50_000,
    )
    expr = (
        t[t.yearID == 2015]
        .cache(storage=ParquetStorage(source=ls_con, relative_path=tmp_path))
        .cache()
    )
    assert expr.execute() is not None


def test_repeated_cache(con, ls_con, tmp_path):
    storage = ParquetStorage(
        source=ls_con,
        relative_path=tmp_path,
    )
    t = (
        con.table("batting")[lambda t: t.yearID > 2014]
        .cache(storage=storage)[lambda t: t.stint == 1]
        .cache(storage=storage)
    )

    actual = t.execute()
    expected = con.table("batting").filter([_.yearID > 2014, _.stint == 1]).execute()

    assert_frame_equal(actual, expected)


def test_cache_default_path_set(batting, ls_con, tmp_path):
    xo.options.cache.default_relative_path = tmp_path

    storage = ParquetStorage(
        source=ls_con,
    )

    expr = batting[lambda t: t.yearID > 2014].limit(1).cache(storage=storage)

    result = expr.execute()

    cache_files = list(
        path
        for path in tmp_path.iterdir()
        if path.is_file()
        and path.name.startswith(KEY_PREFIX)
        and path.name.endswith(".parquet")
    )

    assert result is not None
    assert cache_files


@pytest.mark.snapshot_check
@pytest.mark.xfail
def test_udf_caching(ls_con, alltypes_df, snapshot):
    @xo.udf.scalar.pyarrow
    def my_mul(tinyint_col: dt.int16, smallint_col: dt.int16) -> dt.int16:
        return pc.multiply(tinyint_col, smallint_col)

    @toolz.curry
    def wrapper(f, t):
        # here: map pandas into pyarrow
        # goal: map pyarrow into pandas: so users can define functions in pandas land
        inner = f.__wrapped__
        kwargs = {k: pa.array(v) for k, v in t.items()}
        return inner(**kwargs)

    cols = list(inspect.signature(my_mul).parameters)

    expr = (
        ls_con.create_table("alltypes", alltypes_df)[cols]
        .pipe(lambda t: t.mutate(mulled=my_mul(*(t[col] for col in cols))))
        .cache()
    )
    from_ls = expr.execute()
    from_pandas = alltypes_df[cols].assign(mulled=wrapper(my_mul))
    assert from_ls.equals(from_pandas)

    py_version = f"py{get_python_version_no_dot()}"
    snapshot.assert_match(expr.ls.get_key(), f"{py_version}_udf_caching.txt")


@pytest.mark.snapshot_check
@pytest.mark.xfail
def test_udaf_caching(ls_con, alltypes_df, snapshot):
    def my_mul_sum(df):
        return df.sum().sum()

    cols = ["tinyint_col", "smallint_col"]
    ibis_output_type = dt.infer(alltypes_df[cols].sum().sum())
    by = "bool_col"
    name = "my_mul_sum"

    t = ls_con.create_table("alltypes", alltypes_df)
    agg_udf = agg.pandas_df(
        my_mul_sum,
        t[cols].schema(),
        ibis_output_type,
        name=name,
    )
    from_pandas = (
        alltypes_df.groupby(by)[cols]
        .apply(my_mul_sum)
        .rename(name)
        .reset_index()
        .sort_values(by)
    )
    expr = (
        t.group_by(by)
        .agg(**{name: agg_udf(*(t[col] for col in cols))})
        .order_by(by)
        .cache()
    )
    on_expr = t.group_by(by).agg(**{name: agg_udf.on_expr(t)}).order_by(by).cache()
    assert not expr.ls.exists()
    assert not on_expr.ls.exists()

    from_ls = expr.execute().sort_values(by="bool_col")
    assert_frame_equal(from_ls, from_pandas.sort_values(by="bool_col"))
    assert_frame_equal(from_ls, on_expr.execute().sort_values(by="bool_col"))
    assert expr.ls.exists()
    assert on_expr.ls.exists()

    py_version = f"py{get_python_version_no_dot()}"
    snapshot.assert_match(expr.ls.get_key(), f"{py_version}_test_udaf_caching.txt")
    snapshot.assert_match(on_expr.ls.get_key(), f"{py_version}_test_udaf_caching.txt")


def test_storage_exists_doesnt_materialize(cached_two):
    storage = cached_two.ls.storage
    assert not storage.exists(cached_two)
    assert not storage.exists(cached_two)
    assert not cached_two.ls.exists()
    cached_two.count().execute()
    assert cached_two.ls.exists()
    assert storage.exists(cached_two)


def test_ls_exists_doesnt_materialize(cached_two):
    storage = cached_two.ls.storage
    assert not cached_two.ls.exists()
    assert not cached_two.ls.exists()
    assert not storage.exists(cached_two)
    cached_two.count().execute()
    assert cached_two.ls.exists()
    assert storage.exists(cached_two)


@pytest.mark.parametrize(
    "cls",
    [ParquetSnapshotStorage, ParquetStorage, SourceSnapshotStorage, SourceStorage],
)
def test_cache_find_embedded_backend(cls, parquet_dir):
    con = xo.connect()
    astronauts_path = parquet_dir / "astronauts.parquet"
    storage = cls(source=con)
    expr = con.read_parquet(astronauts_path).cache(storage=storage)
    assert expr._find_backend()._profile == con._profile
