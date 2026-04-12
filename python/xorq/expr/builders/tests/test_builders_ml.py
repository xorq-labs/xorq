"""ML pipeline integration tests for ExprBuilder from_tag_node registry."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

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


sklearn = pytest.importorskip("sklearn")

from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


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
    sk_pipe = make_pipeline(StandardScaler(), LinearRegression())
    pipeline = Pipeline.from_instance(sk_pipe)
    return pipeline.fit(ml_train_expr, target="target")


# ---------------------------------------------------------------------------
# ML pipeline from_tag_node
# ---------------------------------------------------------------------------


def test_ml_extract_metadata(ml_train_expr, ml_fitted):
    """Verify extract_builder_metadata returns steps, features, target."""
    predict_expr = ml_fitted.predict(ml_train_expr)
    tags = walk_nodes((Tag,), predict_expr)
    predict_tag = next(
        t for t in tags if t.metadata.get("tag") == str(FittedPipelineTagKey.PREDICT)
    )
    meta = extract_builder_metadata(predict_tag)
    assert meta is not None
    assert meta["type"] == "fitted_pipeline"
    assert len(meta["steps"]) == 2
    assert meta["target"] == "target"
    assert set(meta["features"]) >= {"feature_0", "feature_1"}


def test_ml_from_tag_node_returns_fitted_pipeline(ml_train_expr, ml_fitted):
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
# ML from_tag_node on transform expression
# ---------------------------------------------------------------------------


def test_ml_from_tag_node_on_transform_expr(ml_train_expr, ml_fitted):
    transform_expr = ml_fitted.transform(ml_train_expr)
    recovered = _resolve_builder_from_tag(transform_expr)
    assert isinstance(recovered, FittedPipeline)
    result = recovered.transform(ml_train_expr)
    assert result is not None


# ---------------------------------------------------------------------------
# ML from_tag_node without cache
# ---------------------------------------------------------------------------


def test_ml_from_tag_node_without_cache():
    train = xo.memtable(
        {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "target": [0, 1, 0]},
        name="ml_nocache",
    )
    sk_pipe = make_pipeline(StandardScaler(), LinearRegression())
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
    sk_pipe = make_pipeline(StandardScaler(), LinearRegression())
    pipeline = Pipeline.from_instance(sk_pipe)
    fitted = pipeline.fit(train, features=(), target="target")
    # Should auto-derive features as ("a", "b") since () is falsy
    assert set(fitted.fitted_steps[0].features) == {"a", "b"}


# ---------------------------------------------------------------------------
# ML catalog roundtrip — zip -> load -> from_tag_node -> execute
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


# ---------------------------------------------------------------------------
# FittedPipeline.from_expr — ValueError on non-pipeline expression
# ---------------------------------------------------------------------------


def test_from_expr_raises_on_non_pipeline():
    t = xo.memtable({"x": [1]}, name="not_pipeline")
    with pytest.raises(ValueError, match="No FittedPipeline tag found"):
        FittedPipeline.from_expr(t)


# ---------------------------------------------------------------------------
# Pipeline.fit — training_hash failure graceful degradation
# ---------------------------------------------------------------------------


def test_fit_training_hash_failure_degrades_gracefully(monkeypatch):
    monkeypatch.setattr(
        "xorq.common.utils.name_utils.make_name",
        Mock(side_effect=RuntimeError("tokenize failed")),
    )
    train = xo.memtable(
        {"a": [1.0, 2.0], "b": [3.0, 4.0], "target": [0, 1]},
        name="hash_fail",
    )
    pipeline = Pipeline.from_instance(
        make_pipeline(StandardScaler(), LinearRegression())
    )
    fitted = pipeline.fit(train, target="target")
    assert fitted.training_hash is None
