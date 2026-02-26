"""Tests for YAML build serialization of sklearn pipeline expressions."""

import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import xorq.api as xo
import xorq.ibis_yaml.translate as translate_mod
from xorq.expr.ml.cross_validation import _make_folds_from_sklearn
from xorq.ibis_yaml.common import FROM_YAML_HANDLERS
from xorq.ibis_yaml.compiler import build_expr, load_expr


sklearn = pytest.importorskip("sklearn")

from sklearn.datasets import make_regression  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402


@pytest.fixture(scope="module")
def lasso_timeseries_fold_expr(tmp_path_factory):
    """Build a TimeSeriesSplit fold expression using deferred_read_parquet.

    Writes regression data to a parquet file so the expression graph is
    YAML-serializable via the Read op handler in the ibis_yaml compiler.
    _make_folds_from_sklearn is purely deferred — no data execution required.
    """
    X, y = make_regression(
        n_samples=200, n_features=4, n_informative=3, random_state=42
    )
    df = pd.DataFrame(X, columns=["f0", "f1", "f2", "f3"]).assign(
        target=y, t=range(200)
    )
    parquet_dir = tmp_path_factory.mktemp("data")
    parquet_path = parquet_dir / "regression_data.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), parquet_path)

    expr = xo.deferred_read_parquet(parquet_path)
    features = ("f0", "f1", "f2", "f3")
    target = "target"

    return _make_folds_from_sklearn(
        expr,
        TimeSeriesSplit(n_splits=3),
        features,
        target,
        random_seed=42,
        order_by="t",
    )


def test_lasso_timeseries_cv_bytes_to_yaml_invoked(
    lasso_timeseries_fold_expr, tmp_path, monkeypatch
):
    """_bytes_to_yaml is called when building a TimeSeriesSplit fold expression.

    _make_folds_from_sklearn calls _make_fold_udwf, which pickles the splitter as
    cv_bytes and stores it in the UDWF config FrozenDict. Building the fold
    expression to YAML must invoke _bytes_to_yaml at least once for that bytes value.
    """
    original = translate_mod._bytes_to_yaml
    call_count = {"n": 0}

    def spy(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    # Register spy into the singledispatch registry; restore original in finally.
    translate_mod.translate_to_yaml.register(bytes)(spy)
    try:
        build_expr(lasso_timeseries_fold_expr, builds_dir=tmp_path)
    finally:
        translate_mod.translate_to_yaml.register(bytes)(original)

    assert call_count["n"] > 0, (
        "_bytes_to_yaml was never called during YAML serialization. "
        "Check that it is registered via @translate_to_yaml.register(bytes)."
    )


def test_lasso_timeseries_cv_bytes_from_yaml_invoked(
    lasso_timeseries_fold_expr, tmp_path, monkeypatch
):
    """_bytes_from_yaml is called when loading a TimeSeriesSplit fold expression.

    After building the fold expression to YAML, loading it must invoke
    _bytes_from_yaml at least once to reconstruct the pickled splitter bytes.
    """
    expr_path = build_expr(lasso_timeseries_fold_expr, builds_dir=tmp_path)

    original = translate_mod._bytes_from_yaml
    call_count = {"n": 0}

    def spy(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setitem(FROM_YAML_HANDLERS, "bytes", spy)

    load_expr(expr_path)

    assert call_count["n"] > 0, (
        "_bytes_from_yaml was never called during YAML deserialization. "
        "Check that it is registered via @register_from_yaml_handler('bytes')."
    )


def test_lasso_timeseries_cv_yaml_roundtrip(lasso_timeseries_fold_expr, tmp_path):
    """The fold expression roundtrips through build/load with identical results."""
    expr_path = build_expr(lasso_timeseries_fold_expr, builds_dir=tmp_path)
    roundtrip_expr = load_expr(expr_path)

    sort_cols = lasso_timeseries_fold_expr.columns
    original_df = (
        lasso_timeseries_fold_expr.execute()
        .sort_values(sort_cols)
        .reset_index(drop=True)
    )
    roundtrip_df = (
        roundtrip_expr.execute().sort_values(sort_cols).reset_index(drop=True)
    )
    tm.assert_frame_equal(original_df, roundtrip_df)
