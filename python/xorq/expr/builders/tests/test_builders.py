"""Tests for Builder framework — registry, base class, and built-in builders."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from xorq.expr.builders import (
    _BUILDER_REGISTRY,
    BUILDER_META_FILENAME,
    BuilderKind,
    Builder,
    get_registry,
    register_builder,
)
from xorq.expr.builders.fitted_pipeline import (
    PIPELINE_PICKLE_FILENAME,
    FittedPipelineBuilder,
)
from xorq.vendor.ibis.expr.types.core import ExprMetadata


class TestBuilderRegistry:
    def test_register_builder_adds_to_registry(self):
        saved = dict(_BUILDER_REGISTRY)
        try:

            @register_builder("test_dummy")
            class DummyBuilder(Builder):
                tag_name = "test_dummy"

            assert "test_dummy" in _BUILDER_REGISTRY
            assert _BUILDER_REGISTRY["test_dummy"] is DummyBuilder
        finally:
            _BUILDER_REGISTRY.clear()
            _BUILDER_REGISTRY.update(saved)

    def test_get_registry_returns_dict(self):
        registry = get_registry()
        assert isinstance(registry, dict)

    def test_builder_kind_enum_values(self):
        assert str(BuilderKind.SemanticModel) == "semantic_model"
        assert str(BuilderKind.FittedPipeline) == "fitted_pipeline"

    def test_entry_point_discovery_populates_registry(self):
        registry = get_registry()
        assert str(BuilderKind.SemanticModel) in registry
        assert str(BuilderKind.FittedPipeline) in registry


class TestBuilderBase:
    def test_build_expr_raises_not_implemented(self):
        spec = Builder(tag_name="test")
        with pytest.raises(NotImplementedError):
            spec.build_expr()

    def test_from_tagged_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            Builder.from_tagged(None)

    def test_from_build_dir_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            Builder.from_build_dir(Path("/tmp"))

    def test_to_build_dir_raises_not_implemented(self):
        spec = Builder(tag_name="test")
        with pytest.raises(NotImplementedError):
            spec.to_build_dir(Path("/tmp"))

    def test_frozen(self):
        spec = Builder(tag_name="test")
        with pytest.raises(AttributeError):
            spec.tag_name = "other"


class TestFittedPipelineBuilder:
    @pytest.fixture
    def mock_fitted_pipeline(self):
        fp = MagicMock()
        fp.is_predict = True
        fp.fitted_steps = (
            MagicMock(
                step=MagicMock(name="scaler", instance=MagicMock(spec=[])),
            ),
            MagicMock(
                step=MagicMock(name="clf", instance=MagicMock(spec=[])),
            ),
        )
        fp.fitted_steps[0].step.name = "scaler"
        fp.fitted_steps[1].step.name = "clf"
        type(fp.fitted_steps[0].step.instance).__name__ = "StandardScaler"
        type(fp.fitted_steps[1].step.instance).__name__ = "LogisticRegression"
        return fp

    def test_to_build_dir_writes_meta_and_pickle(self, mock_fitted_pipeline):
        spec = FittedPipelineBuilder(fitted_pipeline=mock_fitted_pipeline)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            spec.to_build_dir(path)

            assert (path / BUILDER_META_FILENAME).exists()
            assert (path / PIPELINE_PICKLE_FILENAME).exists()

            meta = json.loads((path / BUILDER_META_FILENAME).read_text())
            assert meta["type"] == str(BuilderKind.FittedPipeline)
            assert meta["is_predict"] is True
            assert len(meta["steps"]) == 2

    def test_roundtrip_through_build_dir(self, mock_fitted_pipeline):
        spec = FittedPipelineBuilder(fitted_pipeline=mock_fitted_pipeline)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            spec.to_build_dir(path)
            restored = FittedPipelineBuilder.from_build_dir(path)

            assert restored.tag_name == str(BuilderKind.FittedPipeline)
            assert restored.is_predict is True
            assert len(restored.steps) == 2

    def test_build_expr_unknown_method(self, mock_fitted_pipeline):
        spec = FittedPipelineBuilder(fitted_pipeline=mock_fitted_pipeline)
        with pytest.raises(ValueError, match="Unknown method"):
            spec.build_expr(data=MagicMock(), method="nonexistent")

    def test_steps_property(self, mock_fitted_pipeline):
        spec = FittedPipelineBuilder(fitted_pipeline=mock_fitted_pipeline)
        steps = spec.steps
        assert isinstance(steps, tuple)
        assert len(steps) == 2
        assert steps[0]["name"] == "scaler"
        assert steps[1]["estimator"] == "LogisticRegression"


class TestExprMetadataBuilders:
    def test_from_dict_with_builders(self):
        data = {
            "kind": "expr",
            "schema_out": {"a": "int64"},
            "builders": ({"type": "fitted_pipeline", "description": "test"},),
        }
        meta = ExprMetadata.from_dict(data)
        assert len(meta.builders) == 1
        assert meta.builders[0]["type"] == "fitted_pipeline"

    def test_from_dict_without_builders_backward_compat(self):
        data = {
            "kind": "expr",
            "schema_out": {"a": "int64"},
        }
        meta = ExprMetadata.from_dict(data)
        assert meta.builders == ()

    def test_to_dict_with_builders(self):
        data = {
            "kind": "expr",
            "schema_out": {"a": "int64"},
            "builders": ({"type": "bsl", "description": "test"},),
        }
        meta = ExprMetadata.from_dict(data)
        d = meta.to_dict()
        assert "builders" in d
        assert d["builders"][0]["type"] == "bsl"

    def test_to_dict_without_builders_omits_key(self):
        data = {
            "kind": "expr",
            "schema_out": {"a": "int64"},
        }
        meta = ExprMetadata.from_dict(data)
        d = meta.to_dict()
        assert "builders" not in d
