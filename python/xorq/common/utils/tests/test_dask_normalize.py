import hashlib
import operator
import pathlib
import pickle
import re
from unittest.mock import (
    Mock,
    patch,
)

import cloudpickle
import dask
import pandas as pd
import pyarrow.compute as pc
import pytest
import toolz

import xorq.api as xo
import xorq.common.utils.dask_normalize  # noqa: F401
import xorq.expr.datatypes as dt
from xorq.caching import (
    ParquetCache,
    SourceSnapshotCache,
)
from xorq.common.utils.dask_normalize import (
    get_normalize_token_subset,
)
from xorq.common.utils.dask_normalize.dask_normalize_expr import (
    _extract_duckdb_file_paths,
    _extract_plan_file_paths,
)
from xorq.common.utils.dask_normalize.dask_normalize_utils import (
    file_digest,
    gen_batches,
    manual_file_digest,
    patch_normalize_token,
    walk_normalized,
)
from xorq.expr.udf import agg, make_pandas_expr_udf
from xorq.ibis_yaml.compiler import build_expr


def test_ensure_deterministic():
    assert dask.config.get("tokenize.ensure-deterministic")


def test_unregistered_raises():
    class Unregistered:
        pass

    with pytest.raises(ValueError, match="cannot be deterministically hashed"):
        dask.base.tokenize(Unregistered())


@pytest.fixture(scope="function")
def alltypes_df(pg):
    return pg.table("functional_alltypes").execute()


@pytest.fixture(scope="function")
def batting(pg):
    return pg.table("batting")


@pytest.mark.snapshot_check
def test_tokenize_datafusion_memory_expr(alltypes_df, snapshot):
    con = xo.datafusion.connect()
    t = con.register(alltypes_df, "t")
    f = Mock(side_effect=toolz.functoolz.return_none)
    with patch_normalize_token(type(con), f=f):
        actual = dask.base.tokenize(t)
    f.assert_not_called()
    snapshot.assert_match(actual, "datafusion_memory_key.txt")


@pytest.mark.snapshot_check
def test_tokenize_datafusion_parquet_expr(parquet_dir, snapshot):
    path = parquet_dir.joinpath("functional_alltypes.parquet")
    con = xo.datafusion.connect()
    t = con.read_parquet(path, table_name="t")
    # DataFusion strips the leading "/" when rendering the plan, so the path
    # in the normalized token has no leading slash. Strip both forms to make
    # the snapshot stable across runs.
    parent = str(path.parent)
    # Mock _normalize_path_stat so mtime/size/ino don't vary between machines.
    _fixed_stat = (("st_mtime", 0), ("st_size", 0), ("st_ino", 0))
    with patch(
        "xorq.common.utils.dask_normalize.dask_normalize_expr._normalize_path_stat",
        return_value=_fixed_stat,
    ):
        to_hash = (
            str(tuple(dask.base.normalize_token(t)))
            .replace(parent + "/", "")
            .replace(parent.lstrip("/") + "/", "")
        )
    actual = hashlib.md5(to_hash.encode(), usedforsecurity=False).hexdigest()
    snapshot.assert_match(actual, "datafusion_key.txt")


@pytest.mark.parametrize(
    "ep_str, expected",
    [
        (
            "DataSourceExec: partitions=1, partition_sizes=[1]",
            (),
        ),
        (
            "DataSourceExec: file_groups={1 group: [[tmp/path/file.parquet]]}, file_type=parquet",
            ("/tmp/path/file.parquet",),
        ),
        (
            "DataSourceExec: file_groups={2 groups: [[tmp/a.parquet], [tmp/b.parquet]]}, file_type=parquet",
            ("/tmp/a.parquet", "/tmp/b.parquet"),
        ),
        (
            "DataSourceExec: file_groups={1 group: [[tmp/a.parquet, tmp/b.parquet]]}, file_type=parquet",
            ("/tmp/a.parquet", "/tmp/b.parquet"),
        ),
        (
            "DataSourceExec: file_groups={1 group: [[/tmp/already/absolute.parquet]]}, file_type=parquet",
            ("/tmp/already/absolute.parquet",),
        ),
        (
            "DataSourceExec: file_groups={1 group: [[tmp/data.csv]]}, file_type=csv, has_header=true",
            ("/tmp/data.csv",),
        ),
        (
            "DataSourceExec: file_groups={1 group: [[https://example.com/data.parquet]]}, file_type=parquet",
            ("https://example.com/data.parquet",),
        ),
    ],
)
def test_extract_plan_file_paths(ep_str, expected):
    assert _extract_plan_file_paths(ep_str) == expected


@pytest.mark.parametrize(
    "ddl, expected",
    [
        (
            "CREATE VIEW v AS SELECT * FROM read_parquet('/tmp/file.parquet')",
            ("/tmp/file.parquet",),
        ),
        (
            "CREATE VIEW v AS SELECT * FROM read_parquet(['/tmp/a.parquet', '/tmp/b.parquet'])",
            ("/tmp/a.parquet", "/tmp/b.parquet"),
        ),
        (
            "CREATE VIEW v AS SELECT * FROM read_csv('/tmp/file.csv')",
            ("/tmp/file.csv",),
        ),
        (
            "CREATE TABLE t (a BIGINT, b DOUBLE)",
            (),
        ),
        (
            "CREATE VIEW v AS SELECT * FROM read_parquet('https://example.com/data.parquet')",
            ("https://example.com/data.parquet",),
        ),
    ],
)
def test_extract_duckdb_file_paths(ddl, expected):
    assert _extract_duckdb_file_paths(ddl) == expected


def test_duckdb_remote_http_token_is_url_based():
    """DuckDB: extractor preserves the URL string; token is stable across identical URLs."""
    batting_url = "https://storage.googleapis.com/letsql-pins/batting/20240711T171118Z-431ef/batting.parquet"
    ddl = f"CREATE VIEW batting AS SELECT * FROM read_parquet('{batting_url}')"

    # Extractor now returns the raw URL string, not a mangled local path
    (extracted,) = _extract_duckdb_file_paths(ddl)
    assert extracted == batting_url

    # With a real connection, re-registering the same URL produces the same token
    con = xo.duckdb.connect()
    con.raw_sql(f"CREATE VIEW batting AS SELECT * FROM read_parquet('{batting_url}')")
    t = con.table("batting")
    token = dask.base.tokenize(t)

    con2 = xo.duckdb.connect()
    con2.raw_sql(f"CREATE VIEW batting AS SELECT * FROM read_parquet('{batting_url}')")
    t2 = con2.table("batting")
    assert token == dask.base.tokenize(t2)


def test_datafusion_remote_http_token_is_ep_str_based():
    """DataFusion: extractor preserves the URL string for HTTP-backed tables."""
    batting_url = "https://storage.googleapis.com/letsql-pins/batting/20240711T171118Z-431ef/batting.parquet"

    ep_str = (
        f"DataSourceExec: file_groups={{1 group: [[{batting_url}]]}}, "
        "file_type=parquet, projection=[playerID, yearID]"
    )

    # Extractor now returns the raw URL string
    (extracted,) = _extract_plan_file_paths(ep_str)
    assert extracted == batting_url


_ASTRONAUTS_CSV_URL = "https://raw.githubusercontent.com/ibis-project/testing-data/refs/heads/master/csv/astronauts.csv"


def duckdb_http_csv_table():
    con = xo.duckdb.connect()
    con.read_csv(_ASTRONAUTS_CSV_URL, table_name="t")
    return con.table("t")


def xorq_http_csv_table():
    return xo.connect().read_csv(_ASTRONAUTS_CSV_URL, table_name="t")


@pytest.mark.parametrize(
    "make_table",
    (
        pytest.param(
            duckdb_http_csv_table,
            id="duckdb",
        ),
        pytest.param(
            xorq_http_csv_table,
            id="xorq",
            marks=pytest.mark.xfail(
                raises=NotImplementedError,
                strict=True,
                reason="DataFusion strips scheme+host from HTTP URLs, rendering them as local-looking paths that _normalize_path_stat cannot resolve",
            ),
        ),
    ),
)
def test_http_csv_token_is_stable(make_table):
    """Token for an HTTP-backed CSV table is deterministic across repeated tokenization."""
    assert dask.base.tokenize(make_table()) == dask.base.tokenize(make_table())


def _datafusion_parquet_table(path):
    return xo.datafusion.connect().read_parquet(path, table_name="t")


def _datafusion_csv_table(path):
    return xo.datafusion.connect().read_csv(path, table_name="t")


def _duckdb_parquet_table(path):
    return xo.duckdb.connect().read_parquet(path, table_name="t")


def _duckdb_csv_table(path):
    return xo.duckdb.connect().read_csv(path, table_name="t")


def _xorq_parquet_table(path):
    return xo.connect().read_parquet(path, table_name="t")


def _xorq_csv_table(path):
    return xo.connect().read_csv(path, table_name="t")


@pytest.mark.parametrize(
    "make_table",
    [_datafusion_parquet_table, _duckdb_parquet_table, _xorq_parquet_table],
    ids=["datafusion", "duckdb", "xorq"],
)
def test_parquet_invalidates_on_file_change(tmp_path, make_table):
    """Cache token changes when the backing parquet file is overwritten."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1.0, 2.0, 3.0, 4.0, 5.0]})
    path = tmp_path / "data.parquet"
    df.to_parquet(path)
    token_before = dask.base.tokenize(make_table(path))
    df.iloc[:2].to_parquet(path)
    token_after = dask.base.tokenize(make_table(path))
    assert token_before != token_after


def test_duckdb_multi_path_cache_key_invalidates_on_file_change(tmp_path):
    """ParquetCache key changes when one of multiple parquet files backing a view changes."""
    import pandas as pd  # noqa: PLC0415

    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    path1 = tmp_path / "part1.parquet"
    path2 = tmp_path / "part2.parquet"
    df.to_parquet(path1)
    df.to_parquet(path2)

    con = xo.duckdb.connect()
    con.raw_sql(
        f"CREATE VIEW test_view AS SELECT * FROM read_parquet(['{path1}', '{path2}'])"
    )
    cache = ParquetCache.from_kwargs(
        source=con, relative_path="cache", base_path=tmp_path
    )
    key_before = cache.calc_key(con.table("test_view"))

    # Modify only path1 — cache key must change even though path2 is unchanged
    df.iloc[:1].to_parquet(path1)

    con2 = xo.duckdb.connect()
    con2.raw_sql(
        f"CREATE VIEW test_view AS SELECT * FROM read_parquet(['{path1}', '{path2}'])"
    )
    cache2 = ParquetCache.from_kwargs(
        source=con2, relative_path="cache", base_path=tmp_path
    )
    key_after = cache2.calc_key(con2.table("test_view"))

    assert key_before != key_after


def test_xorq_multi_csv_path_cache_key_invalidates_on_file_change(tmp_path):
    """ParquetCache key changes when one of multiple CSV files backing an xorq table changes."""
    import pandas as pd  # noqa: PLC0415

    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    path1 = tmp_path / "part1.csv"
    path2 = tmp_path / "part2.csv"
    df.to_csv(path1, index=False)
    df.to_csv(path2, index=False)

    con = xo.connect()
    con.read_csv([path1, path2], table_name="t")
    cache = ParquetCache.from_kwargs(
        source=con, relative_path="cache", base_path=tmp_path
    )
    key_before = cache.calc_key(con.table("t"))

    # Modify only path1 — cache key must change even though path2 is unchanged
    df.iloc[:1].to_csv(path1, index=False)

    con2 = xo.connect()
    con2.read_csv([path1, path2], table_name="t")
    cache2 = ParquetCache.from_kwargs(
        source=con2, relative_path="cache", base_path=tmp_path
    )
    key_after = cache2.calc_key(con2.table("t"))

    assert key_before != key_after


def write_parquet(df, path):
    df.to_parquet(path)


def write_csv(df, path):
    df.to_csv(path, index=False)


@pytest.mark.parametrize(
    "write_file, suffix, make_table",
    [
        (write_parquet, ".parquet", _datafusion_parquet_table),
        (write_csv, ".csv", _datafusion_csv_table),
        (write_parquet, ".parquet", _duckdb_parquet_table),
        (write_csv, ".csv", _duckdb_csv_table),
        (write_parquet, ".parquet", _xorq_parquet_table),
        (write_csv, ".csv", _xorq_csv_table),
    ],
    ids=[
        "datafusion-parquet",
        "datafusion-csv",
        "duckdb-parquet",
        "duckdb-csv",
        "xorq-parquet",
        "xorq-csv",
    ],
)
def test_token_stable_for_same_file(tmp_path, write_file, suffix, make_table):
    """Two connections backed by the same file produce the same token."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    path = tmp_path / f"data{suffix}"
    write_file(df, path)
    assert dask.base.tokenize(make_table(path)) == dask.base.tokenize(make_table(path))


def _datafusion_parquet_tables(path1, path2):
    con = xo.datafusion.connect()
    return con.read_parquet(path1, table_name="t1"), con.read_parquet(
        path2, table_name="t2"
    )


def _duckdb_parquet_tables(path1, path2):
    con = xo.duckdb.connect()
    con.raw_sql(f"CREATE VIEW t1 AS SELECT * FROM read_parquet('{path1}')")
    con.raw_sql(f"CREATE VIEW t2 AS SELECT * FROM read_parquet('{path2}')")
    return con.table("t1"), con.table("t2")


def _xorq_parquet_tables(path1, path2):
    con = xo.connect()
    return con.read_parquet(path1, table_name="t1"), con.read_parquet(
        path2, table_name="t2"
    )


@pytest.mark.parametrize(
    "make_tables",
    [_datafusion_parquet_tables, _duckdb_parquet_tables, _xorq_parquet_tables],
    ids=["datafusion", "duckdb", "xorq"],
)
def test_parquet_different_files_produce_different_tokens(tmp_path, make_tables):
    """Tables backed by different parquet files produce different tokens."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    path1, path2 = tmp_path / "a.parquet", tmp_path / "b.parquet"
    df.to_parquet(path1)
    df.iloc[:1].to_parquet(path2)

    t1, t2 = make_tables(path1, path2)
    assert dask.base.tokenize(t1) != dask.base.tokenize(t2)


@pytest.mark.parametrize(
    "make_tables",
    [_datafusion_parquet_tables, _duckdb_parquet_tables, _xorq_parquet_tables],
    ids=["datafusion", "duckdb", "xorq"],
)
def test_parquet_same_content_different_path_produces_different_token(
    tmp_path, make_tables
):
    """Same file content at different paths produces different tokens (path is in token)."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    path1, path2 = tmp_path / "a.parquet", tmp_path / "b.parquet"
    df.to_parquet(path1)
    df.to_parquet(path2)

    t1, t2 = make_tables(path1, path2)
    assert dask.base.tokenize(t1) != dask.base.tokenize(t2)


def test_datafusion_parquet_different_schema_produces_different_token(tmp_path):
    """Same file registered with different column projections produces different tokens."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": ["x", "y", "z"]})
    path = tmp_path / "data.parquet"
    df.to_parquet(path)

    con = xo.datafusion.connect()
    t_full = con.read_parquet(path, table_name="t_full")
    # register same file again but select only two columns
    t_narrow = con.read_parquet(path, table_name="t_narrow")[["a", "b"]]
    assert dask.base.tokenize(t_full) != dask.base.tokenize(t_narrow)


@pytest.mark.snapshot_check
def test_tokenize_pandas_expr(alltypes_df, snapshot):
    con = xo.pandas.connect()
    t = con.create_table("t", alltypes_df)
    f = Mock(side_effect=toolz.functoolz.return_none)
    with patch_normalize_token(type(t.op().source), f=f):
        actual = dask.base.tokenize(t)
    f.assert_not_called()
    snapshot.assert_match(actual, "pandas_key.txt")


@pytest.mark.snapshot_check
def test_tokenize_duckdb_expr(batting, snapshot):
    con = xo.duckdb.connect()
    t = con.create_table("dashed-name", batting.to_pyarrow())
    f = Mock(side_effect=toolz.functoolz.return_none)
    with patch_normalize_token(type(con), f=f):
        actual = dask.base.tokenize(t)
    f.assert_not_called()

    snapshot.assert_match(actual, "duckdb_key.txt")


@pytest.mark.snapshot_check
def test_pandas_snapshot_key(alltypes_df, snapshot):
    con = xo.pandas.connect()
    t = con.create_table("t", alltypes_df)
    cache = SourceSnapshotCache.from_kwargs(source=con)
    actual = cache.strategy.calc_key(t)
    snapshot.assert_match(actual, "pandas_snapshot_key.txt")


@pytest.mark.snapshot_check
def test_duckdb_snapshot_key(batting, snapshot):
    con = xo.duckdb.connect()
    t = con.create_table("dashed-name", batting.to_pyarrow())
    cache = SourceSnapshotCache.from_kwargs(source=con)
    actual = cache.strategy.calc_key(t)
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


def test_patch_normalize_token():
    def make_type(name):
        cls = type(name, (), {})
        cls.__module__ = name
        return cls

    names = ("to_retain", "to_drop")
    name_to_cls = {name: make_type(name) for name in names}
    name_to_mock = {
        cls.__module__: Mock(
            side_effect=toolz.partial(
                dask.base.normalize_token.register,
                cls,
                toolz.functoolz.return_none,
            ),
        )
        for cls in name_to_cls.values()
    }

    assert not any(
        cls.__module__ in dask.base.normalize_token._lazy
        for cls in name_to_cls.values()
    )
    assert not any(
        cls in dask.base.normalize_token._lookup for cls in name_to_cls.values()
    )
    values = {name: name_to_mock[name] for name in ("to_retain",)}
    with patch.dict(
        dask.base.normalize_token._lazy,
        values=values,
    ):
        with patch_normalize_token(
            name_to_cls["to_drop"],
            f=toolz.functoolz.return_none,
        ):
            assert name_to_cls["to_retain"] not in dask.base.normalize_token._lookup
            for name in names:
                dask.base.tokenize(name_to_cls[name]())
            for cls in name_to_cls.values():
                assert cls in dask.base.normalize_token._lookup
        assert name_to_cls["to_drop"] not in dask.base.normalize_token._lookup
        assert name_to_cls["to_retain"] in dask.base.normalize_token._lookup

    assert "to_retain" not in dask.base.normalize_token._lookup
    assert name_to_cls["to_retain"] in dask.base.normalize_token._lookup


def test_parquet_cache_tokenize_stable_across_cloudpickle():
    con = xo.connect()
    cache = ParquetCache.from_kwargs(source=con)
    token_before = dask.base.tokenize(cache)
    cache2 = cloudpickle.loads(cloudpickle.dumps(cache))
    token_after = dask.base.tokenize(cache2)
    assert token_before == token_after


def test_loaded_parquet_dt_has_stable_token(tmp_path):
    """Two loads of the same build zip produce equal `.ls.tokenized`.

    Regression: `normalize_datafusion_databasetable` used to hash the
    execution plan string, which contains the extract dir path — fresh
    per load — so tokens diverged across loads.
    """
    import pandas as pd  # noqa: PLC0415

    from xorq.catalog.expr_utils import (  # noqa: PLC0415
        build_expr_context_zip,
        load_expr_from_zip,
    )

    df = pd.DataFrame({"x": [1, 2, 3]})
    parquet_path = tmp_path / "data.parquet"
    df.to_parquet(parquet_path)
    expr = xo.deferred_read_parquet(parquet_path, xo.connect(), "t")
    with build_expr_context_zip(expr) as zip_path:
        a = load_expr_from_zip(zip_path)
        b = load_expr_from_zip(zip_path)
        assert a.ls.tokenized == b.ls.tokenized


def test_different_cache_types_produce_different_hashes():
    t = xo.memtable({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    c0 = t.cache()
    c1 = t.cache(ParquetCache.from_kwargs())
    b0 = build_expr(c0)
    b1 = build_expr(c1)
    assert b0 != b1


def test_scalar_udf_token_stable_across_udf_counter_states():
    """Token of a ScalarUDF with computed_kwargs_expr must not depend on
    the process-global UDF name counter."""

    def train(df):
        return pickle.dumps({"trained": True})

    def predict(model, df):
        return [0.0] * len(df)

    def _build_scalar_udf():
        t = xo.memtable({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        schema = t[("a", "b")].schema()
        model_udaf = agg.pandas_df(
            fn=train,
            schema=schema,
            return_type=dt.binary,
            name="mymodel",
        )
        predict_udf = make_pandas_expr_udf(
            computed_kwargs_expr=model_udaf.on_expr(t),
            fn=predict,
            schema=schema,
            return_type=dt.float64,
            name="mypredict",
        )
        return predict_udf(t.a, t.b).op()

    udf_op_1 = _build_scalar_udf()
    token_1 = dask.base.tokenize(udf_op_1)

    # Bump the global counter by creating throwaway AggUDFs,
    # simulating a different import order or parallel worker.
    for _ in range(5):
        agg.pandas_df(
            fn=train,
            schema=xo.memtable({"x": [1]}).schema(),
            return_type=dt.binary,
        )

    udf_op_2 = _build_scalar_udf()
    token_2 = dask.base.tokenize(udf_op_2)

    assert token_1 == token_2, (
        f"ScalarUDF token changed with UDF counter state: {token_1} != {token_2}"
    )


@pytest.mark.snapshot_check
def test_tokenize_named_scalar_parameter_float64(snapshot):
    """Token for a float64 NamedScalarParameter is stable across sessions."""
    p = xo.param("threshold", "float64", default=1.5)
    actual = dask.base.tokenize(p.op())
    snapshot.assert_match(actual, "named_scalar_param_float64.txt")


@pytest.mark.snapshot_check
def test_tokenize_named_scalar_parameter_string(snapshot):
    """Token for a string NamedScalarParameter is stable across sessions."""
    p = xo.param("prefix", "string")
    actual = dask.base.tokenize(p.op())
    snapshot.assert_match(actual, "named_scalar_param_string.txt")


@pytest.mark.snapshot_check
def test_tokenize_expr_with_named_param(snapshot):
    """Token for an expression containing a NamedScalarParameter is stable.

    Uses an unbound table with a fixed name so the SQL is deterministic
    across sessions (InMemoryTable names are process-local random strings).
    """
    threshold = xo.param("threshold", "float64", default=1.5)
    t = xo.table([("x", "float64")], name="t")
    expr = t.filter(t.x > threshold)
    actual = dask.base.tokenize(expr)
    snapshot.assert_match(actual, "expr_with_named_param.txt")


@pytest.mark.snapshot_check
def test_tokenize_expr_two_named_params_positional(snapshot):
    """Tokens for two exprs with swapped params are stable and distinct."""
    a = xo.param("a", "float64", default=1.0)
    b = xo.param("b", "float64", default=2.0)
    t = xo.table([("x", "float64"), ("y", "float64")], name="t")
    expr_ab = t.filter(t.x > a, t.y > b)
    expr_ba = t.filter(t.x > b, t.y > a)

    tokenize_ab = dask.base.tokenize(expr_ab)
    tokenize_ba = dask.base.tokenize(expr_ba)
    assert tokenize_ab != tokenize_ba
    snapshot.assert_match(tokenize_ab, "expr_params_ab.txt")
    snapshot.assert_match(tokenize_ba, "expr_params_ba.txt")


def test_udf_sql_name_uses_func_name_not_class_name():
    """Compiled SQL must use __func_name__ (stable) not type().__name__ (counter-suffixed).

    When multiple UDFs are created, ibis appends a sequential counter to the
    generated class name (e.g. my_add_0, my_add_3).  The SQL compiler should
    use __func_name__ instead so that the emitted SQL is deterministic.
    """

    def _make_udf_expr():
        @xo.udf.scalar.pyarrow
        def my_add(x: dt.float64, y: dt.float64) -> dt.float64:
            return pc.add(x, y)

        t = xo.memtable({"a": [1.0], "b": [2.0]})
        return t.mutate(c=my_add(t.a, t.b))

    expr_1 = _make_udf_expr()
    con = xo.duckdb.connect()
    sql_1 = con.compile(expr_1)

    # Bump the global UDF counter by creating throwaway UDFs
    for _ in range(5):

        @xo.udf.scalar.pyarrow
        def _throwaway(x: dt.float64) -> dt.float64:
            return x

    expr_2 = _make_udf_expr()
    sql_2 = con.compile(expr_2)

    # Strip memtable names (they differ per instance) and compare the rest
    normalize_memtable = re.compile(r"ibis_pandas_memtable_\w+")
    assert normalize_memtable.sub("MEMTABLE", sql_1) == normalize_memtable.sub(
        "MEMTABLE", sql_2
    ), f"SQL changed with UDF counter state:\n  {sql_1}\n  {sql_2}"
    # The user-given name must appear; the counter-suffixed class name must not
    assert "my_add(" in sql_1.lower()
    assert "my_add_" not in sql_1.lower()
