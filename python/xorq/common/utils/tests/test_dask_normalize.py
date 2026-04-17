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


def test_datafusion_parquet_token_stable_across_registered_path(tmp_path):
    """Registering the same parquet content at a different path yields the
    same token — content, not path, is what the normalizer keys on."""
    import pandas as pd  # noqa: PLC0415

    df = pd.DataFrame({"x": [1, 2, 3]})
    p1 = tmp_path / "a" / "data.parquet"
    p2 = tmp_path / "b" / "data.parquet"
    for p in (p1, p2):
        p.parent.mkdir(parents=True)
        df.to_parquet(p)

    con1 = xo.datafusion.connect()
    con2 = xo.datafusion.connect()
    t1 = con1.read_parquet(p1, "t")
    t2 = con2.read_parquet(p2, "t")
    assert dask.base.tokenize(t1) == dask.base.tokenize(t2)


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
