"""ML pipeline integration tests for ExprBuilder from_tagged registry."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import pytest
import sklearn.pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import xorq.api as xo
from xorq.catalog.catalog import Catalog
from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.builders import (
    _resolve_builder_from_tag,
    extract_builder_metadata,
)
from xorq.expr.ml.enums import FittedPipelineTagKey
from xorq.expr.ml.pipeline_lib import FittedPipeline, Pipeline
from xorq.expr.relations import Tag
from xorq.ibis_yaml.enums import ExprKind


pytestmark = pytest.mark.skipif(
    not importlib.util.find_spec("sklearn"),
    reason="sklearn not installed",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ml_train_expr():
    return xo.memtable(
        {
            "feature_0": [1.0, 2.0, 3.0],
            "feature_1": [4.0, 5.0, 6.0],
            "target": [0, 1, 0],
        },
        name="ml_train",
    )


@pytest.fixture
def ml_fitted(ml_train_expr):
    sk_pipe = sklearn.pipeline.make_pipeline(StandardScaler(), LinearRegression())
    pipeline = Pipeline.from_instance(sk_pipe)
    return pipeline.fit(ml_train_expr, target="target")


# ---------------------------------------------------------------------------
# ML pipeline from_tagged
# ---------------------------------------------------------------------------


def test_ml_extract_metadata(ml_train_expr, ml_fitted):
    """Verify extract_builder_metadata returns steps, features, target."""
    predict_expr = ml_fitted.predict(ml_train_expr)
    tags = walk_nodes((Tag,), predict_expr)
    predict_tag = next(
        t for t in tags if t.metadata.get("tag") == str(FittedPipelineTagKey.PREDICT)
    )
    meta = extract_builder_metadata(str(FittedPipelineTagKey.PREDICT), predict_tag)
    assert meta is not None
    assert meta["type"] == "fitted_pipeline"
    assert len(meta["steps"]) == 2
    assert meta["target"] == "target"
    assert set(meta["features"]) >= {"feature_0", "feature_1"}


def test_ml_from_tagged_returns_fitted_pipeline(ml_train_expr, ml_fitted):
    """Verify _resolve_builder_from_tag on a predict expr returns a FittedPipeline."""
    predict_expr = ml_fitted.predict(ml_train_expr)
    recovered = _resolve_builder_from_tag(predict_expr)
    assert isinstance(recovered, FittedPipeline)
    result = recovered.predict(ml_train_expr)
    assert result is not None
    assert "prediction" in result.columns or len(result.columns) > 0


# ---------------------------------------------------------------------------
# FittedStep.get_tag_kwargs includes target
# ---------------------------------------------------------------------------


def test_fitted_step_tag_kwargs_include_target(ml_train_expr, ml_fitted):
    for step in ml_fitted.fitted_steps:
        kwargs = step.get_tag_kwargs()
        assert "target" in kwargs
        assert kwargs["target"] == "target"


# ---------------------------------------------------------------------------
# ML from_tagged on transform expression
# ---------------------------------------------------------------------------


def test_ml_from_tagged_on_transform_expr(ml_train_expr, ml_fitted):
    transform_expr = ml_fitted.transform(ml_train_expr)
    recovered = _resolve_builder_from_tag(transform_expr)
    assert isinstance(recovered, FittedPipeline)
    result = recovered.transform(ml_train_expr)
    assert result is not None


# ---------------------------------------------------------------------------
# ML from_tagged without cache
# ---------------------------------------------------------------------------


def test_ml_from_tagged_without_cache():
    train = xo.memtable(
        {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "target": [0, 1, 0]},
        name="ml_nocache",
    )
    sk_pipe = sklearn.pipeline.make_pipeline(StandardScaler(), LinearRegression())
    pipeline = Pipeline.from_instance(sk_pipe)
    fitted = pipeline.fit(train, target="target")
    predict_expr = fitted.predict(train)
    recovered = _resolve_builder_from_tag(predict_expr)
    assert isinstance(recovered, FittedPipeline)


# ---------------------------------------------------------------------------
# Pipeline.fit features edge case
# ---------------------------------------------------------------------------


def test_fit_with_empty_features_auto_derives():
    """Passing features=() should auto-derive features (falsy tuple)."""
    train = xo.memtable(
        {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "target": [0, 1, 0]},
        name="ml_empty_features",
    )
    sk_pipe = sklearn.pipeline.make_pipeline(StandardScaler(), LinearRegression())
    pipeline = Pipeline.from_instance(sk_pipe)
    fitted = pipeline.fit(train, features=(), target="target")
    # Should auto-derive features as ("a", "b") since () is falsy
    assert set(fitted.fitted_steps[0].features) == {"a", "b"}


# ---------------------------------------------------------------------------
# ML catalog roundtrip — zip -> load -> from_tagged -> execute
# ---------------------------------------------------------------------------


@pytest.fixture
def ml_catalog_entry(ml_train_expr, ml_fitted):
    predictions = ml_fitted.predict(ml_train_expr)
    with tempfile.TemporaryDirectory() as tmp:
        catalog = Catalog.from_repo_path(Path(tmp) / "cat", init=True)
        catalog.add(predictions, aliases=("ml-roundtrip",), sync=False)
        entry = catalog.get_catalog_entry("ml-roundtrip", maybe_alias=True)
        yield entry


def test_ml_catalog_entry_kind(ml_catalog_entry):
    assert ml_catalog_entry.kind == ExprKind.ExprBuilder


def test_ml_catalog_entry_sidecar_metadata(ml_catalog_entry):
    builders = ml_catalog_entry.metadata.builders
    assert len(builders) == 1
    b = builders[0]
    assert b["type"] == "fitted_pipeline"
    assert b["target"] == "target"
    assert len(b["steps"]) == 2


def test_ml_catalog_roundtrip_recover_and_predict(ml_catalog_entry):
    recovered = ml_catalog_entry.expr.ls.builder
    assert isinstance(recovered, FittedPipeline)
    prd = xo.memtable(
        {"feature_0": [5.0, 6.0], "feature_1": [7.0, 8.0]},
        name="ml_prd",
    )
    result = recovered.predict(prd)
    df = result.execute()
    assert len(df) == 2


def test_ml_catalog_roundtrip_recover_and_transform(ml_catalog_entry):
    recovered = ml_catalog_entry.expr.ls.builder
    prd = xo.memtable(
        {"feature_0": [5.0, 6.0], "feature_1": [7.0, 8.0]},
        name="ml_prd_t",
    )
    result = recovered.transform(prd)
    df = result.execute()
    assert len(df) == 2
