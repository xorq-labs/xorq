"""Tests for YAML build serialization of sklearn pipeline expressions."""

import pathlib

import numpy as np
import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import xorq.api as xo
from xorq.expr.ml.cross_validation import _make_folds_from_sklearn
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.ibis_yaml.compiler import build_expr, load_expr


sklearn = pytest.importorskip("sklearn")

from sklearn.datasets import make_regression  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
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


def test_deferred_metric_build_load_roundtrip(tmp_path: pathlib.Path) -> None:
    """A build embedding a metric loads and executes (#2233).

    The pickle-layer tests in test_metrics.py cover the mechanism; this covers
    the symptom -- ``build`` succeeding while the artifact it wrote is
    unloadable. A MetricComputation reaches the payload via the UDAF config
    (``agg.pandas_df(fn=self)`` in ``MetricComputation.on_expr``), so a dunder
    that breaks cloudpickle's by-reference lookup fails here and nowhere
    earlier.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.normal(size=50)}).assign(
        yhat=lambda t: t["y"] + rng.normal(scale=0.1, size=50)
    )
    parquet_path = tmp_path / "metric_data.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), parquet_path)

    metric_expr = deferred_sklearn_metric(
        xo.deferred_read_parquet(parquet_path),
        target="y",
        pred="yhat",
        metric=r2_score,
    )
    expected = metric_expr.execute()

    expr_path = build_expr(metric_expr, builds_dir=tmp_path / "builds")

    assert load_expr(expr_path).execute() == pytest.approx(expected)
