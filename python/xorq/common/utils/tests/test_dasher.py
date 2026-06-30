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

import functools
import numbers
import operator
import pickle
import tempfile
import types

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import toolz
from xorq_dasher import fqn

import xorq.api as xo
import xorq.common.utils.dasher as dasher
import xorq.expr.datatypes as dt
import xorq.expr.relations as rel
from xorq.caching import ParquetCache
from xorq.common.utils import graph_utils
from xorq.common.utils.dasher import (
    _EXTRA_RULES,
    HASHER,
    _canonicalize_catalog_path,
    _extract_datafusion_plan_paths,
    _extract_duckdb_file_paths,
    compute_expr_token,
    expr_metadata,
    tokenize,
)
from xorq.common.utils.dasher._gap_rules import (
    normalize_ibis_schema,
    normalize_methodcaller,
    normalize_pandas_dataframe,
    normalize_pandas_series,
    normalize_sklearn_constraint,
    normalize_sklearn_hidden,
    normalize_slice,
)
from xorq.common.utils.dasher._opaque import (
    _normalize_computed_kwargs_expr,
    _parent_token,
)
from xorq.common.utils.dasher._relations import _databasetable_dispatcher
from xorq.common.utils.file_utils import normalize_read_path_stat
from xorq.common.utils.tests._test_helpers import BombHasher, MockOp, Probe
from xorq.common.utils.toolz_utils import curry as xo_curry
from xorq.expr import api
from xorq.expr.ml.metrics import MetricComputation, deferred_sklearn_metric
from xorq.expr.udf import agg, make_pandas_expr_udf
from xorq.ibis_yaml.compiler import build_expr
from xorq.vendor.ibis.expr.operations.generic import Cast
from xorq.vendor.ibis.expr.operations.relations import DatabaseTable, Schema
from xorq.vendor.ibis.expr.operations.udf import ScalarUDF
from xorq.vendor.ibis.expr.types import Expr


try:
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.utils._param_validation import HasMethods as _SklearnHasMethods
    from sklearn.utils._param_validation import Hidden as _SklearnHidden
    from sklearn.utils._param_validation import Interval as _SklearnInterval
    from sklearn.utils._param_validation import (
        MissingValues as _SklearnMissingValues,
    )
    from sklearn.utils._param_validation import StrOptions as _SklearnStrOptions
    from sklearn.utils._param_validation import _Constraint as _SklearnConstraint

    _has_sklearn = True
except ImportError:
    _has_sklearn = False

requires_sklearn = pytest.mark.skipif(not _has_sklearn, reason="sklearn not installed")


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


def _decimal_parquet(path, precision, scale):
    table = pa.table(
        {"x": pa.array(["1.23", "4.56"]).cast(pa.decimal128(precision, scale))}
    )
    import pyarrow.parquet as pq  # noqa: PLC0415

    pq.write_table(table, path)


@pytest.mark.parametrize(
    "make_table",
    (
        pytest.param(_datafusion_table, id="datafusion"),
        pytest.param(_duckdb_table, id="duckdb"),
    ),
)
def test_file_table_decimal_precision_distinguishes_schema_component(
    tmp_path, make_table
):
    # Regression guard for xorq-labs/xorq#1973: the file-table DT normalizers
    # must route the schema through ``normalize_ibis_schema`` (which keeps
    # ``str(dtype)``), not ``dt.schema.to_pandas()`` (which collapses every
    # decimal to ``dtype('O')``).  Same column name, different decimal
    # precision must produce distinct schema components in the token.
    pa_path, pb_path = tmp_path / "a.parquet", tmp_path / "b.parquet"
    _decimal_parquet(pa_path, 18, 3)
    _decimal_parquet(pb_path, 9, 2)
    na = _databasetable_dispatcher(make_table(pa_path, "ta").op())
    nb = _databasetable_dispatcher(make_table(pb_path, "tb").op())
    schema_a, schema_b = na[1], nb[1]
    assert schema_a == ("ibis.Schema", (("x", "decimal(18, 3)"),))
    assert schema_b == ("ibis.Schema", (("x", "decimal(9, 2)"),))
    assert schema_a != schema_b


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


def test_normalize_read_nonexistent_multi_path_raises():
    read = _make_read(
        (("hash_path", ("file1.parquet", "file2.parquet")),),
    )
    with pytest.raises(FileNotFoundError, match="local path does not exist"):
        HASHER.normalize(read)


def test_normalize_read_nonexistent_absolute_path_raises():
    read = _make_read(
        (("hash_path", "/nonexistent/path/to/data.parquet"),),
    )
    with pytest.raises(FileNotFoundError, match="local path does not exist"):
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


def test_bare_param_as_project_value_tokenizes():
    """Regression (#2037): a NamedScalarParameter as a direct Project column
    value must not be wrapped in Alias — Project forbids Alias values."""
    expr = xo.memtable({"_": [0]}).select(
        year_months=xo.param("year_months", "string", default="2025_11,2025_12")
    )
    tok = tokenize(expr)
    assert isinstance(tok, str) and len(tok) > 0


def test_two_params_same_dtype_produce_distinct_tokens():
    """Two NamedScalarParameters of the same dtype in one expression must not
    collapse to the same placeholder — their tokens must differ."""
    t = xo.memtable({"_": [0]})
    p1 = xo.param("start", "int64")
    p2 = xo.param("end", "int64")

    tok_both = tokenize(t.select(a=p1, b=p2))
    tok_same = tokenize(t.select(a=p1, b=p1))
    assert tok_both != tok_same


def test_two_params_same_dtype_swapped_positions_produce_distinct_tokens():
    """Swapping two same-dtype params across select positions must produce
    a different token, even though the SQL after NULL substitution is identical."""
    t = xo.memtable({"_": [0]})
    p1 = xo.param("start", "int64")
    p2 = xo.param("end", "int64")

    tok_ab = tokenize(t.select(a=p1, b=p2))
    tok_ba = tokenize(t.select(a=p2, b=p1))
    assert tok_ab != tok_ba


def test_two_params_same_dtype_in_filter_produce_distinct_tokens():
    """Same-dtype params used in filter position must also be distinguishable."""
    t = xo.memtable({"x": [1, 2, 3]})
    p1 = xo.param("lo", "int64")
    p2 = xo.param("hi", "int64")

    tok_two = tokenize(t.filter(t.x > p1).filter(t.x < p2))
    tok_one = tokenize(t.filter(t.x > p1).filter(t.x < p1))
    assert tok_two != tok_one


# --- _stat_or_canonical: catalog-extract path canonicalization ------------


def test_canonicalize_catalog_path_strips_tempdir_prefix():
    raw = "/var/tmp/xorq-catalog-abc123def/build_xxx/data.parquet"
    canonical, did_strip = _canonicalize_catalog_path(raw)
    assert did_strip
    assert canonical == "build_xxx/data.parquet"


def test_canonicalize_catalog_path_passes_through_non_catalog():
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


def test_dasher_tokenize_dunder_is_invoked():
    """xorq_dasher's fallback rule must consult ``__dasher_tokenize__``.

    Cache, ParquetStorage, SourceStorage, GCStorage and others rely on it
    for their cache keys.
    """

    assert tokenize(Probe("same")) == tokenize(Probe("same"))
    assert tokenize(Probe("same")) != tokenize(Probe("different"))


def test_opaque_placeholders_are_content_addressed(tmp_path):
    """Each opaque arm in ``_xorq_opaque_to_placeholder`` should produce a
    stable, schema-derived placeholder — re-constructing the same expr in a
    fresh process must produce the same token (no ``id()`` leakage)."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    path = tmp_path / "data.parquet"
    df.to_parquet(path)

    def build_cached():
        return (
            xo.connect().read_parquet(path, table_name="t").filter(xo._.a > 0).cache()
        )

    def build_remote():
        src = xo.connect()
        src.create_table("t", df)
        return src.table("t").into_backend(xo.connect(), name="t_remote")

    def build_param():
        threshold = xo.param("threshold", "float64", default=1.5)
        return xo.table([("x", "float64")], name="t").filter(xo._.x > threshold)

    for build in (build_cached, build_remote, build_param):
        assert tokenize(build()) == tokenize(build())


def test_normalize_computed_kwargs_expr_is_data_free():
    """``_normalize_computed_kwargs_expr`` is data-free per ADR-0010 — two
    cked expressions with the same shape but different ``InMemoryTable``
    contents must produce identical helper output."""
    cke_a = xo.memtable(pd.DataFrame({"x": [1, 2, 3]})).filter(xo._.x > 0)
    cke_b = xo.memtable(pd.DataFrame({"x": [10, 20, 30]})).filter(xo._.x > 0)
    assert _normalize_computed_kwargs_expr(cke_a) == _normalize_computed_kwargs_expr(
        cke_b
    )


def test_replace_nodes_raises_on_unhandled_opaque(monkeypatch):
    """A future addition to ``opaque_ops`` without a corresponding ``case``
    arm in ``replace_nodes.process_node`` must raise loudly rather than
    silently producing a wrong hash."""
    monkeypatch.setattr(graph_utils, "opaque_ops", graph_utils.opaque_ops + (Cast,))

    con = xo.connect()
    con.create_table("t", pd.DataFrame({"a": [1, 2, 3]}))
    expr = con.table("t").mutate(a_float=xo._.a.cast("float64"))

    with pytest.raises(ValueError, match="unhandled opaque op"):
        graph_utils.replace_nodes(lambda op, _kw: op, expr.op())


# --- _gap_rules normalizers ------------------------------------------------


def test_normalize_methodcaller_no_kwargs():
    mc = operator.methodcaller("upper")
    result = normalize_methodcaller(mc)
    assert result == ("operator.methodcaller", "upper", (), {})


def test_normalize_methodcaller_positional_args_only():
    mc = operator.methodcaller("startswith", "foo", 1)
    result = normalize_methodcaller(mc)
    assert result == ("operator.methodcaller", "startswith", ("foo", 1), {})


def test_normalize_methodcaller_with_kwargs():
    mc = operator.methodcaller("encode", encoding="utf-8")
    result = normalize_methodcaller(mc)
    assert result[0] == "operator.methodcaller"
    assert result[1] == "encode"
    assert result[2] == ()
    assert result[3] == {"encoding": "utf-8"}


def test_normalize_methodcaller_portable():
    mc = operator.methodcaller("upper")
    result = normalize_methodcaller(mc)
    assert result == ("operator.methodcaller", "upper", (), {})


def test_methodcaller_reduce_shape_no_args():
    """Guard: __reduce__ for methodcaller(name) returns (cls, (name,)).

    _extract_methodcaller_fields relies on this shape.  If a Python upgrade
    changes it, this test fails loudly instead of silently mis-extracting fields.
    """
    mc = operator.methodcaller("upper")
    reduced = mc.__reduce__()
    assert len(reduced) == 2
    constructor, args = reduced
    assert constructor is operator.methodcaller
    assert args == ("upper",)


def test_methodcaller_reduce_shape_positional_args():
    """Guard: __reduce__ for methodcaller(name, *args) returns (cls, (name, *args))."""
    mc = operator.methodcaller("startswith", "foo", 1)
    reduced = mc.__reduce__()
    assert len(reduced) == 2
    constructor, args = reduced
    assert constructor is operator.methodcaller
    assert args == ("startswith", "foo", 1)


def test_methodcaller_reduce_shape_kwargs():
    """Guard: __reduce__ for methodcaller(name, **kw) wraps constructor in functools.partial."""
    mc = operator.methodcaller("encode", encoding="utf-8")
    reduced = mc.__reduce__()
    assert len(reduced) == 2
    constructor, args = reduced
    assert isinstance(constructor, functools.partial)
    assert constructor.func is operator.methodcaller
    assert constructor.args == ("encode",)
    assert constructor.keywords == {"encoding": "utf-8"}
    assert args == ()


def test_normalize_pandas_series_delegates_to_dataframe():
    series = pd.Series([1, 2, 3], name="x")
    result = normalize_pandas_series(series)
    assert result[0] == "pandas.Series"
    assert result[1] == "x"
    inner = result[2]
    assert inner[0] == "pandas.DataFrame"


def test_normalize_pandas_series_same_data_same_hash():
    s1 = pd.Series([1, 2, 3], name="x")
    s2 = pd.Series([1, 2, 3], name="x")
    assert normalize_pandas_series(s1) == normalize_pandas_series(s2)


def test_normalize_pandas_series_different_data_different_hash():
    s1 = pd.Series([1, 2, 3], name="x")
    s2 = pd.Series([1, 2, 4], name="x")
    assert normalize_pandas_series(s1) != normalize_pandas_series(s2)


def test_normalize_slice_full():
    assert normalize_slice(slice(1, 10, 2)) == ("slice", 1, 10, 2)


def test_normalize_slice_stop_only():
    assert normalize_slice(slice(5)) == ("slice", None, 5, None)


def test_normalize_slice_all_none():
    assert normalize_slice(slice(None)) == ("slice", None, None, None)


def test_normalize_slice_negative_indices():
    assert normalize_slice(slice(-3, -1)) == ("slice", -3, -1, None)


def test_normalize_ibis_schema_empty():
    assert normalize_ibis_schema(xo.schema({})) == ("ibis.Schema", ())


def test_normalize_ibis_schema_simple():
    result = normalize_ibis_schema(xo.schema({"a": "int64", "b": "string"}))
    assert result == ("ibis.Schema", (("a", "int64"), ("b", "string")))


def test_normalize_ibis_schema_preserves_column_order():
    result = normalize_ibis_schema(xo.schema({"z": "float64", "a": "int64"}))
    assert [name for name, _ in result[1]] == ["z", "a"]


def test_normalize_ibis_schema_complex_types_not_collapsed():
    # pandas to_pandas() collapses both to dtype('O'), but str(dtype) preserves info
    result_decimal = normalize_ibis_schema(xo.schema({"x": dt.Decimal(12, 3)}))
    result_array = normalize_ibis_schema(xo.schema({"x": dt.Array(dt.int64)}))
    assert result_decimal != result_array


def test_normalize_ibis_schema_decimal_precision_distinguishes():
    result_a = normalize_ibis_schema(xo.schema({"x": dt.Decimal(10, 2)}))
    result_b = normalize_ibis_schema(xo.schema({"x": dt.Decimal(12, 3)}))
    assert result_a != result_b


def test_normalize_pandas_dataframe_returns_pa_table():
    """normalize_pandas_dataframe returns a raw pa.Table for dasher's
    normalize_pyarrow_table rule to hash."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = normalize_pandas_dataframe(df)
    assert result[0] == "pandas.DataFrame"
    assert isinstance(result[3], pa.Table)


# --- _parent_token fallback ------------------------------------------------


def test_parent_token_fallback_is_reproducible(monkeypatch):
    """The RecursionError fallback in _parent_token must produce the same
    token for semantically identical objects regardless of memory address.
    """
    monkeypatch.setattr(dasher, "HASHER", BombHasher())

    tok1 = _parent_token(MockOp())
    tok2 = _parent_token(MockOp())

    assert tok1 == tok2
    assert isinstance(tok1, str) and len(tok1) > 0


@pytest.mark.xfail(
    reason="fallback hashes only type+schema — same-type, same-schema ops collide",
    strict=True,
)
def test_parent_token_fallback_distinguishes_same_type_different_ops(monkeypatch):
    """KNOWN LIMITATION: the RecursionError fallback hashes
    ``{module}.{qualname}|{schema}``.  Two distinct ops of the *same* class
    with the *same* schema produce identical fallback tokens.
    """
    monkeypatch.setattr(dasher, "HASHER", BombHasher())

    op_a = MockOp()
    op_a.schema = "shared-schema"
    op_b = MockOp()
    op_b.schema = "shared-schema"

    tok_a = _parent_token(op_a)
    tok_b = _parent_token(op_b)

    assert tok_a != tok_b


# --- expr_metadata / compute_expr_token ------------------------------------


def test_expr_metadata_round_trip_memtable():
    """tokenize(expr) == compute_expr_token(structural_hash, slot_hashes)."""
    t = xo.memtable({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    expr = t.filter(t.x > 1)
    token = tokenize(expr)
    meta = expr_metadata(expr)

    assert meta["version"] == 4
    assert isinstance(meta["structural_hash"], str)
    assert len(meta["slots"]) >= 1

    recomputed = compute_expr_token(
        meta["structural_hash"], tuple(s["hash"] for s in meta["slots"])
    )
    assert recomputed == token


def test_expr_metadata_round_trip_parquet(tmp_path):
    """Round-trip works for file-backed expressions (Read + DatabaseTable)."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    path = tmp_path / "data.parquet"
    df.to_parquet(path)

    con = xo.connect()
    t = con.read_parquet(path, table_name="rt_test")
    expr = t.filter(t.a > 1)

    token = tokenize(expr)
    meta = expr_metadata(expr)
    recomputed = compute_expr_token(
        meta["structural_hash"], tuple(s["hash"] for s in meta["slots"])
    )
    assert recomputed == token


def test_expr_metadata_round_trip_deferred_read(tmp_path):
    """Round-trip works for deferred reads (Read slots)."""
    df = pd.DataFrame({"a": [10, 20, 30]})
    path = tmp_path / "deferred.parquet"
    df.to_parquet(path)

    con = xo.connect()
    t = xo.deferred_read_parquet(str(path), con, table_name="dr_test")

    token = tokenize(t)
    meta = expr_metadata(t)
    assert any(s["kind"] == "Read" for s in meta["slots"])

    recomputed = compute_expr_token(
        meta["structural_hash"], tuple(s["hash"] for s in meta["slots"])
    )
    assert recomputed == token


def test_expr_metadata_structural_hash_stable_across_data(parquet_dir):
    """Same query shape on identically-named tables → same structural hash."""
    con = xo.connect()
    t1 = con.read_parquet(
        parquet_dir / "astronauts.parquet",
        table_name="data",
    )
    t2 = con.read_parquet(
        parquet_dir / "batting.parquet",
        table_name="data",
    )
    meta1 = expr_metadata(t1.head(10))
    meta2 = expr_metadata(t2.head(10))
    assert meta1["structural_hash"] == meta2["structural_hash"]
    assert meta1["slots"][0]["hash"] != meta2["slots"][0]["hash"]


def test_expr_metadata_slot_hash_changes_on_file_edit(tmp_path):
    """Editing the backing file changes the slot hash but not the structural hash."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    path = tmp_path / "data.parquet"
    df.to_parquet(path)

    con = xo.connect()
    meta_before = expr_metadata(con.read_parquet(path, table_name="t"))

    df.iloc[:1].to_parquet(path)
    con2 = xo.connect()
    meta_after = expr_metadata(con2.read_parquet(path, table_name="t"))

    assert meta_before["structural_hash"] == meta_after["structural_hash"]
    assert meta_before["slots"][0]["hash"] != meta_after["slots"][0]["hash"]


def test_expr_metadata_structural_hash_changes_on_udf_edit():
    """Changing a UDF's function body changes structural_hash but not slot hashes."""

    def train(df):
        return pickle.dumps({"trained": True})

    def predict_v1(model, df):
        return [0.0] * len(df)

    def predict_v2(model, df):
        return [1.0] * len(df)

    def _build(predict_fn):
        t = xo.memtable({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        schema = t[("a", "b")].schema()
        model_udaf = agg.pandas_df(
            fn=train, schema=schema, return_type=dt.binary, name="mymodel"
        )
        predict_udf = make_pandas_expr_udf(
            computed_kwargs_expr=model_udaf.on_expr(t),
            fn=predict_fn,
            schema=schema,
            return_type=dt.float64,
            name="mypredict",
        )
        return predict_udf(t.a, t.b)

    meta_v1 = expr_metadata(_build(predict_v1))
    meta_v2 = expr_metadata(_build(predict_v2))

    assert meta_v1["structural_hash"] != meta_v2["structural_hash"]
    v1_slot_hashes = [s["hash"] for s in meta_v1["slots"]]
    v2_slot_hashes = [s["hash"] for s in meta_v2["slots"]]
    assert v1_slot_hashes == v2_slot_hashes


def test_expr_metadata_schema_validation():
    """Returned metadata has the expected schema."""
    t = xo.memtable({"x": [1]})
    meta = expr_metadata(t)

    assert set(meta.keys()) == {"version", "structural_hash", "slots"}
    for slot in meta["slots"]:
        assert set(slot.keys()) == {"index", "kind", "name", "hash"}
        assert isinstance(slot["index"], int)
        assert slot["kind"] in ("Read", "DatabaseTable", "InMemoryTable")
        assert isinstance(slot["name"], str)
        assert isinstance(slot["hash"], str) and len(slot["hash"]) == 32


def test_compute_expr_token_minimal_env():
    """compute_expr_token works with only xxhash + struct (no xorq import)."""
    from xorq.common.utils.dasher._recompute import (  # noqa: PLC0415
        compute_expr_token as _standalone,
    )

    structural = "a" * 32
    slots = ("b" * 32, "c" * 32)
    result = _standalone(structural, slots)
    assert isinstance(result, str) and len(result) == 32

    # Deterministic
    assert _standalone(structural, slots) == result

    # Different inputs → different output
    assert _standalone(structural, ("d" * 32, "c" * 32)) != result


def test_recompute_encode_is_dasher_core():
    """When xorq_dasher is installed, _recompute._encode IS the dasher import."""
    from xorq_dasher.core import _encode as _dasher_encode  # noqa: PLC0415

    from xorq.common.utils.dasher._recompute import (  # noqa: PLC0415
        _encode as _recompute_encode,
    )

    assert _recompute_encode is _dasher_encode


def test_recompute_fallback_encode_matches_dasher_core():
    """_encode_fallback must match xorq_dasher.core._encode for all primitive types."""
    from xorq_dasher.core import _encode as _dasher_encode  # noqa: PLC0415

    from xorq.common.utils.dasher._recompute import (  # noqa: PLC0415
        _encode_fallback,
    )

    cases = [
        None,
        True,
        False,
        0,
        42,
        -7,
        3.14,
        float("nan"),
        float("inf"),
        float("-inf"),
        "",
        "hello",
        b"bytes",
        (),
        ("a", 1, None, (True, -3, 2.0, b"\xff")),
        ("ibis.Expr.v4", "a" * 32, "b" * 32, "c" * 32),
    ]
    for obj in cases:
        assert _encode_fallback(obj) == _dasher_encode(obj), f"mismatch for {obj!r}"


def test_expr_metadata_multi_path_read_slot_name(tmp_path):
    """Read node with a tuple-valued read_path produces a comma-joined slot name."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    path1 = tmp_path / "part1.csv"
    path2 = tmp_path / "part2.csv"
    df.to_csv(path1, index=False)
    df.to_csv(path2, index=False)

    read = _make_read(
        (
            ("hash_path", (str(path1), str(path2))),
            ("read_path", (str(path1), str(path2))),
        )
    )
    meta = expr_metadata(read.to_expr())

    read_slots = [s for s in meta["slots"] if s["kind"] == "Read"]
    assert len(read_slots) == 1
    name = read_slots[0]["name"]
    assert str(path1) in name
    assert str(path2) in name
    assert ", " in name


def test_expr_metadata_zero_slot_scalar():
    """Pure scalar expr has no data leaves → slots == [], round-trip holds."""
    expr = (xo.literal(1) + 2).name("x")
    token = tokenize(expr)
    meta = expr_metadata(expr)

    assert meta["version"] == 4
    assert meta["slots"] == []
    assert isinstance(meta["structural_hash"], str)
    assert compute_expr_token(meta["structural_hash"], ()) == token


@requires_sklearn
def test_extra_rules_fqn_strings() -> None:
    """Guard against class relocations silently breaking hardcoded FQN strings."""
    expected = {
        "functools._lru_cache_wrapper": functools._lru_cache_wrapper,
        "functools.partial": functools.partial,
        "builtins.builtin_function_or_method": types.BuiltinFunctionType,
        "builtins.slice": slice,
        "builtins.property": property,
        "toolz.functoolz.Compose": toolz.functoolz.Compose,
        "toolz.functoolz.curry": toolz.curry,
        "toolz.functoolz.excepts": toolz.functoolz.excepts,
        "operator.methodcaller": operator.methodcaller,
        "xorq.vendor.ibis.expr.operations.relations.DatabaseTable": DatabaseTable,
        "xorq.expr.relations.Read": rel.Read,
        "xorq.vendor.ibis.expr.types.core.Expr": Expr,
        "xorq.vendor.ibis.expr.schema.Schema": Schema,
        "xorq.vendor.ibis.expr.operations.udf.ScalarUDF": ScalarUDF,
        "numpy.dtype": np.dtype,
        "pandas.core.series.Series": pd.Series,
        "pandas.core.frame.DataFrame": pd.DataFrame,
        "xorq.common.utils.toolz_utils.curry": xo_curry,
        "sklearn.utils._param_validation._Constraint": _SklearnConstraint,
        "sklearn.utils._param_validation.Hidden": _SklearnHidden,
    }
    for literal, cls in expected.items():
        assert fqn(cls) == literal, (
            f"FQN drift: {cls!r} moved from {literal!r} to {fqn(cls)!r}; "
            f"update the string in _EXTRA_RULES"
        )

    production_fqns = {fqn_str for fqn_str, _ in _EXTRA_RULES}
    assert production_fqns == set(expected), (
        f"test/production mismatch: {production_fqns.symmetric_difference(set(expected))}"
    )


# --- sklearn constraint normalizers -----------------------------------------


@requires_sklearn
def test_normalize_sklearn_constraint_stroptions() -> None:
    obj = _SklearnStrOptions({"micro", "macro", "weighted"})
    result = normalize_sklearn_constraint(obj)
    assert result[0] == "sklearn._Constraint"
    assert "StrOptions" in result[1]
    items_dict = dict(result[2])
    assert items_dict["options"] == tuple(sorted({"micro", "macro", "weighted"}))


@requires_sklearn
def test_normalize_sklearn_constraint_interval() -> None:
    obj = _SklearnInterval(type=numbers.Real, left=0, right=1, closed="both")
    result = normalize_sklearn_constraint(obj)
    assert result[0] == "sklearn._Constraint"
    assert "Interval" in result[1]


@requires_sklearn
def test_normalize_sklearn_hidden() -> None:
    obj = _SklearnHidden(constraint="array-like")
    result = normalize_sklearn_hidden(obj)
    assert result == ("sklearn.Hidden", "array-like")


@requires_sklearn
def test_normalize_sklearn_constraint_hasmethods() -> None:
    result = normalize_sklearn_constraint(_SklearnHasMethods(["fit", "predict"]))
    assert result[0] == "sklearn._Constraint"
    assert "HasMethods" in result[1]


@requires_sklearn
def test_normalize_sklearn_constraint_missingvalues() -> None:
    """MissingValues has nested _Constraint objects in __dict__ — dasher must recurse."""
    obj = _SklearnMissingValues()
    tok = tokenize(obj)
    assert isinstance(tok, str) and len(tok) > 0
    assert tok == tokenize(_SklearnMissingValues())


@requires_sklearn
def test_sklearn_constraint_set_order_stable() -> None:
    """StrOptions with the same elements in any insertion order must normalize identically."""
    a = normalize_sklearn_constraint(_SklearnStrOptions({"z", "a", "m"}))
    b = normalize_sklearn_constraint(_SklearnStrOptions({"a", "m", "z"}))
    assert a == b


# --- MetricComputation tokenization -----------------------------------------


@requires_sklearn
def test_metric_computation_tokenize_stable() -> None:
    mc1 = MetricComputation(target="y", pred="pred", metric_fn=roc_auc_score)
    mc2 = MetricComputation(target="y", pred="pred", metric_fn=roc_auc_score)
    assert tokenize(mc1) == tokenize(mc2)


@requires_sklearn
def test_metric_computation_different_fn_different_token() -> None:
    mc_auc = MetricComputation(target="y", pred="pred", metric_fn=roc_auc_score)
    mc_acc = MetricComputation(target="y", pred="pred", metric_fn=accuracy_score)
    assert tokenize(mc_auc) != tokenize(mc_acc)


@requires_sklearn
def test_deferred_sklearn_metric_build_expr() -> None:
    """End-to-end: deferred_sklearn_metric expression tokenizes via build_expr."""
    preds = api.register(
        pd.DataFrame({"y": [0, 1, 0, 1], "pred": [0.1, 0.9, 0.25, 0.8]}),
        "preds",
    )
    expr = deferred_sklearn_metric(
        expr=preds, target="y", pred="pred", metric=roc_auc_score
    )
    result = build_expr(expr, builds_dir=tempfile.mkdtemp())
    assert result is not None
