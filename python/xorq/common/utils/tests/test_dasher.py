"""Behavioral tests for ``xorq.common.utils.dasher``.

Covers the invariants the legacy ``test_dask_normalize.py`` suite pinned that
still apply to the dasher-backed cache-key subsystem:

- File-change invalidation (DataFusion + DuckDB + xorq) — a parquet rewrite at
  the same path changes the token.
- Same-content stability — repeated construction of the same table on the same
  file produces the same token.
- Different files at different paths produce different tokens.
- Schema sensitivity — same file registered with different projections.
- UDF counter independence — two ``ScalarUDF`` ops built from the same Python
  function produce the same token even if the process-global UDF name counter
  ticked between them.
- Cache-type sensitivity — wrapping the same expression in different cache
  classes produces different build hashes.
- ``Read``-op contract — build-relative path branch, multi-path error,
  nonexistent-path error.
- ``NamedScalarParameter`` handling — stable token across sessions, and
  expressions containing parameters tokenize without raising
  ``OperationNotDefinedError`` during SQL compilation.
"""

from __future__ import annotations

import pickle

import pandas as pd
import pytest

import xorq.api as xo
import xorq.expr.datatypes as dt
import xorq.expr.relations as rel
from xorq.caching import ParquetCache
from xorq.common.utils.dasher import HASHER, tokenize
from xorq.common.utils.defer_utils import normalize_read_path_stat
from xorq.expr.udf import agg, make_pandas_expr_udf

# --- file-change invalidation ---------------------------------------------


def _datafusion_table(path, table_name="t"):
    con = xo.datafusion.connect()
    return con.read_parquet(path, table_name=table_name)


def _duckdb_table(path, table_name="t"):
    con = xo.duckdb.connect()
    con.raw_sql(f"CREATE VIEW {table_name} AS SELECT * FROM read_parquet('{path}')")
    return con.table(table_name)


def _xorq_table(path, table_name="t"):
    return xo.connect().read_parquet(path, table_name=table_name)


_PARQUET_BACKENDS = pytest.mark.parametrize(
    "make_table",
    (
        pytest.param(_datafusion_table, id="datafusion"),
        pytest.param(_duckdb_table, id="duckdb"),
        pytest.param(_xorq_table, id="xorq"),
    ),
)


@_PARQUET_BACKENDS
def test_parquet_invalidates_on_file_change(tmp_path, make_table):
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1.0, 2.0, 3.0, 4.0, 5.0]})
    path = tmp_path / "data.parquet"
    df.to_parquet(path)
    token_before = tokenize(make_table(path))
    df.iloc[:2].to_parquet(path)
    token_after = tokenize(make_table(path))
    assert token_before != token_after


@_PARQUET_BACKENDS
def test_parquet_same_file_token_stable(tmp_path, make_table):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    path = tmp_path / "data.parquet"
    df.to_parquet(path)
    assert tokenize(make_table(path)) == tokenize(make_table(path))


@_PARQUET_BACKENDS
def test_parquet_different_files_produce_different_tokens(tmp_path, make_table):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    p1, p2 = tmp_path / "a.parquet", tmp_path / "b.parquet"
    df.to_parquet(p1)
    df.iloc[:1].to_parquet(p2)
    assert tokenize(make_table(p1, "t1")) != tokenize(make_table(p2, "t2"))


@_PARQUET_BACKENDS
def test_parquet_same_content_different_path_produces_different_token(
    tmp_path, make_table
):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    p1, p2 = tmp_path / "a.parquet", tmp_path / "b.parquet"
    df.to_parquet(p1)
    df.to_parquet(p2)
    assert tokenize(make_table(p1, "t1")) != tokenize(make_table(p2, "t2"))


def test_datafusion_parquet_different_schema_produces_different_token(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": ["x", "y", "z"]})
    path = tmp_path / "data.parquet"
    df.to_parquet(path)
    con = xo.datafusion.connect()
    t_full = con.read_parquet(path, table_name="t_full")
    t_proj = t_full.select("a", "b")
    assert tokenize(t_full) != tokenize(t_proj)


# --- UDF counter independence ---------------------------------------------


def test_scalar_udf_token_stable_across_udf_counter_states():
    def train(df):
        return pickle.dumps({"trained": True})

    def predict(model, df):
        return [0.0] * len(df)

    def _build():
        t = xo.memtable({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        schema = t[("a", "b")].schema()
        model_udaf = agg.pandas_df(
            fn=train, schema=schema, return_type=dt.binary, name="mymodel"
        )
        predict_udf = make_pandas_expr_udf(
            computed_kwargs_expr=model_udaf.on_expr(t),
            fn=predict,
            schema=schema,
            return_type=dt.float64,
            name="mypredict",
        )
        return predict_udf(t.a, t.b).op()

    op_1 = _build()
    # Build a throwaway UDF to advance the process-global UDF name counter.
    _build()
    op_2 = _build()
    assert tokenize(op_1) == tokenize(op_2)


# --- cache-type sensitivity ----------------------------------------------


def test_different_cache_types_produce_different_hashes():
    t = xo.memtable({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    c0 = t.cache()
    c1 = t.cache(ParquetCache.from_kwargs())
    assert tokenize(c0) != tokenize(c1)


# --- Read normalizer contract --------------------------------------------


def _make_read(read_kwargs):
    return rel.Read(
        method_name="read_parquet",
        name="test_table",
        schema=xo.schema({"a": "int64"}),
        source=xo.connect(),
        read_kwargs=read_kwargs,
        normalize_method=normalize_read_path_stat,
    )


def test_normalize_read_build_relative_path():
    """Build-bundled Read: relative hash_path == read_path is tokenized as a
    content-addressed build-relative path (no filesystem stat)."""
    read = _make_read(
        (
            ("hash_path", "builds/abc123.parquet"),
            ("read_path", "builds/abc123.parquet"),
        )
    )
    flat = str(HASHER.normalize(read))
    assert "build-relative-path" in flat
    assert "builds/abc123.parquet" in flat


def test_normalize_read_multi_path_raises():
    read = _make_read(
        (("hash_path", ("file1.parquet", "file2.parquet")),),
    )
    with pytest.raises(NotImplementedError, match="Don't know how to deal with path"):
        HASHER.normalize(read)


def test_normalize_read_nonexistent_absolute_path_raises():
    read = _make_read(
        (("hash_path", "/nonexistent/path/to/data.parquet"),),
    )
    with pytest.raises(NotImplementedError, match="Don't know how to deal with path"):
        HASHER.normalize(read)


# --- NamedScalarParameter ------------------------------------------------


def test_named_scalar_parameter_token_stable():
    a = xo.param("threshold", "float64", default=1.5)
    b = xo.param("threshold", "float64", default=1.5)
    assert tokenize(a.op()) == tokenize(b.op())


def test_named_scalar_parameter_default_distinguishes():
    a = xo.param("threshold", "float64", default=1.5)
    b = xo.param("threshold", "float64", default=2.5)
    assert tokenize(a.op()) != tokenize(b.op())


def test_named_scalar_parameter_dtype_distinguishes():
    a = xo.param("threshold", "int64")
    b = xo.param("threshold", "string")
    assert tokenize(a.op()) != tokenize(b.op())


def test_expr_with_named_param_tokenizes_without_raising():
    """Regression: NamedScalarParameter must be replaced with a literal in
    the opaque-placeholder pass before SQL compilation, otherwise the
    ibis SQL compiler raises ``OperationNotDefinedError`` for it."""
    threshold = xo.param("threshold", "float64", default=1.5)
    t = xo.table([("x", "float64")], name="t")
    expr = t.filter(t.x > threshold)
    # The token doesn't need a specific value; it must just compute.
    tok = tokenize(expr)
    assert isinstance(tok, str) and len(tok) > 0


# --- _stat_or_canonical: catalog-extract path canonicalization ------------


def test_canonicalize_catalog_path_strips_tempdir_prefix():
    from xorq.common.utils.dasher import _canonicalize_catalog_path  # noqa: PLC0415

    raw = "/var/tmp/xorq-catalog-abc123def/build_xxx/data.parquet"
    canonical, did_strip = _canonicalize_catalog_path(raw)
    assert did_strip
    assert canonical == "build_xxx/data.parquet"


def test_canonicalize_catalog_path_passes_through_non_catalog():
    from xorq.common.utils.dasher import _canonicalize_catalog_path  # noqa: PLC0415

    raw = "/home/user/data/foo.parquet"
    canonical, did_strip = _canonicalize_catalog_path(raw)
    assert not did_strip
    assert canonical == raw


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
def test_extract_datafusion_plan_paths(ep_str, expected):
    from xorq.common.utils.dasher import _extract_datafusion_plan_paths  # noqa: PLC0415

    assert _extract_datafusion_plan_paths(ep_str) == expected


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
    from xorq.common.utils.dasher import _extract_duckdb_file_paths  # noqa: PLC0415

    assert _extract_duckdb_file_paths(ddl) == expected


# --- multi-path cache-key invalidation -------------------------------------


def test_duckdb_multi_path_cache_key_invalidates_on_file_change(tmp_path):
    """ParquetCache key changes when one of multiple parquet files backing a view changes."""
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

    # Modify only path1 — cache key must change even though path2 is unchanged.
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

    df.iloc[:1].to_csv(path1, index=False)

    con2 = xo.connect()
    con2.read_csv([path1, path2], table_name="t")
    cache2 = ParquetCache.from_kwargs(
        source=con2, relative_path="cache", base_path=tmp_path
    )
    key_after = cache2.calc_key(con2.table("t"))

    assert key_before != key_after


# --- HTTP-backed table stability -------------------------------------------

_ASTRONAUTS_CSV_URL = (
    "https://raw.githubusercontent.com/ibis-project/testing-data/"
    "refs/heads/master/csv/astronauts.csv"
)


def _duckdb_http_csv_table():
    con = xo.duckdb.connect()
    con.read_csv(_ASTRONAUTS_CSV_URL, table_name="t")
    return con.table("t")


@pytest.mark.parametrize("make_table", [_duckdb_http_csv_table], ids=["duckdb"])
def test_http_csv_token_is_stable(make_table):
    """Token for an HTTP-backed CSV table is deterministic across repeated tokenization.

    Only DuckDB is exercised here: xorq/DataFusion strips scheme+host from HTTP
    URLs, rendering them as local-looking paths that ``_normalize_path_stat``
    cannot resolve (covered by the older xfail test in test_dask_normalize).
    """
    assert tokenize(make_table()) == tokenize(make_table())
