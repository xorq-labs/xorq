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
        mock_fitted_pipeline.spec = ["nonexistent"]  # remove auto-mock for getattr
        del mock_fitted_pipeline.nonexistent
        spec = FittedPipelineBuilder(fitted_pipeline=mock_fitted_pipeline)
        with pytest.raises(ValueError, match="has no method"):
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


# ---------------------------------------------------------------------------
# Integration tests — SemanticModelBuilder
# ---------------------------------------------------------------------------


@pytest.fixture
def con():
    import xorq.api as xo

    return xo.connect()


@pytest.mark.library
def test_semantic_builder_build_expr(con):
    from boring_semantic_layer import to_semantic_table

    from xorq.expr.builders.semantic_model import SemanticModelBuilder

    table = con.create_table(
        "sm_test",
        {
            "origin": ["JFK", "LAX", "ORD"],
            "carrier": ["AA", "UA", "AA"],
            "dep_delay": [10.0, -5.0, 30.0],
        },
    )
    model = (
        to_semantic_table(table)
        .with_dimensions(origin=lambda t: t.origin)
        .with_measures(flight_count=lambda t: t.count())
    )
    builder = SemanticModelBuilder(model=model)
    assert "origin" in builder.available_dimensions
    assert "flight_count" in builder.available_measures

    result = builder.build_expr(dimensions=("origin",), measures=("flight_count",)).execute()
    assert "origin" in result.columns
    assert len(result) > 0


@pytest.mark.library
def test_semantic_builder_rebind(con):
    from boring_semantic_layer import to_semantic_table

    from xorq.expr.builders.semantic_model import SemanticModelBuilder

    dev = con.create_table("sm_dev", {"origin": ["JFK"], "delay": [10.0]})
    prd = con.create_table("sm_prd", {"origin": ["SFO"], "delay": [5.0]})
    model = (
        to_semantic_table(dev)
        .with_dimensions(origin=lambda t: t.origin)
        .with_measures(avg_delay=lambda t: t.delay.mean())
    )
    builder = SemanticModelBuilder(model=model)
    rebound = builder.rebind(prd)
    assert rebound.available_dimensions == builder.available_dimensions
    assert rebound.available_measures == builder.available_measures


@pytest.mark.library
def test_semantic_builder_roundtrip():
    from boring_semantic_layer import to_semantic_table

    import xorq.api as xo
    from xorq.expr.builders.semantic_model import SemanticModelBuilder

    c = xo.connect()
    table = c.create_table("sm_rt", {"x": ["a", "b"], "v": [1.0, 2.0]})
    model = (
        to_semantic_table(table)
        .with_dimensions(x=lambda t: t.x)
        .with_measures(total=lambda t: t.v.sum())
    )
    builder = SemanticModelBuilder(model=model)

    with tempfile.TemporaryDirectory() as tmp:
        build_path = builder.to_build_dir(Path(tmp))
        recovered = SemanticModelBuilder.from_build_dir(build_path)
        assert set(recovered.available_dimensions) == set(builder.available_dimensions)
        assert set(recovered.available_measures) == set(builder.available_measures)


# ---------------------------------------------------------------------------
# Integration tests — FittedPipelineBuilder
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_builder(con):
    import sklearn.pipeline
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    from xorq.caching import ParquetCache
    from xorq.expr.ml.pipeline_lib import Pipeline

    train = con.create_table(
        "fp_train",
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "y": [0, 0, 0, 1, 1, 1],
        },
    )
    pipe = sklearn.pipeline.Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=3)),
    ])
    xorq_pipe = Pipeline.from_instance(pipe)
    cache = ParquetCache.from_kwargs(source=con)
    fitted = xorq_pipe.fit(train, features=("x1", "x2"), target="y", cache=cache)
    return FittedPipelineBuilder(fitted_pipeline=fitted), train


def test_fitted_builder_predict(con, fitted_builder):
    builder, _ = fitted_builder
    inference = con.create_table("fp_inf", {"x1": [2.5], "x2": [25.0]})
    result = builder.build_expr(data=inference, method="predict").execute()
    assert len(result) == 1


def test_fitted_builder_transform(con, fitted_builder):
    builder, _ = fitted_builder
    inference = con.create_table("fp_inf_t", {"x1": [2.5], "x2": [25.0]})
    result = builder.build_expr(data=inference, method="transform").execute()
    assert len(result) == 1


def test_fitted_builder_rebind(con, fitted_builder):
    import cloudpickle

    builder, train = fitted_builder
    pickled = cloudpickle.dumps(builder.fitted_pipeline)
    restored_fp = cloudpickle.loads(pickled)  # noqa: S301
    stale = FittedPipelineBuilder(fitted_pipeline=restored_fp)

    rebound = stale.rebind(train)
    inference = con.create_table("fp_inf_rb", {"x1": [3.0], "x2": [35.0]})
    result = rebound.build_expr(data=inference, method="predict").execute()
    assert len(result) == 1


def test_fitted_step_getstate_preserves_model(fitted_builder):
    import cloudpickle

    builder, _ = fitted_builder
    fp = builder.fitted_pipeline
    # Eagerly materialize
    for fs in fp.fitted_steps:
        _ = fs.model

    pickled = cloudpickle.dumps(fp)
    restored = cloudpickle.loads(pickled)  # noqa: S301

    for orig_fs, rest_fs in zip(fp.fitted_steps, restored.fitted_steps, strict=True):
        assert rest_fs.model is not None
        assert type(rest_fs.model).__name__ == type(orig_fs.model).__name__


# ---------------------------------------------------------------------------
# Integration tests — catalog roundtrip
# ---------------------------------------------------------------------------


def test_catalog_add_get_fitted_builder(con, fitted_builder):
    from xorq.catalog.catalog import Catalog

    builder, _ = fitted_builder
    with tempfile.TemporaryDirectory() as tmp:
        catalog = Catalog.from_repo_path(Path(tmp) / "cat", init=True)
        catalog.add_builder(builder, __file__, aliases=("test-pipe",), sync=False)
        recovered = catalog.get_builder("test-pipe")
        assert len(recovered.steps) == len(builder.steps)
        for orig, rec in zip(builder.steps, recovered.steps, strict=True):
            assert orig["name"] == rec["name"]
            assert orig["estimator"] == rec["estimator"]


@pytest.mark.library
def test_catalog_add_get_semantic_builder():
    from boring_semantic_layer import to_semantic_table

    import xorq.api as xo
    from xorq.catalog.catalog import Catalog
    from xorq.expr.builders.semantic_model import SemanticModelBuilder

    c = xo.connect()
    table = c.create_table("cat_sm", {"origin": ["JFK", "LAX"], "delay": [10.0, -5.0]})
    model = (
        to_semantic_table(table)
        .with_dimensions(origin=lambda t: t.origin)
        .with_measures(avg_delay=lambda t: t.delay.mean())
    )
    spec = SemanticModelBuilder(model=model)

    with tempfile.TemporaryDirectory() as tmp:
        catalog = Catalog.from_repo_path(Path(tmp) / "cat", init=True)
        catalog.add_builder(spec, __file__, aliases=("test-sem",), sync=False)
        recovered = catalog.get_builder("test-sem")
        assert set(recovered.available_dimensions) == set(spec.available_dimensions)
        assert set(recovered.available_measures) == set(spec.available_measures)


# ---------------------------------------------------------------------------
# Integration tests — _extract_builders BSL traversal
# ---------------------------------------------------------------------------


@pytest.mark.library
def test_extract_builders_bsl_traversal():
    from boring_semantic_layer import to_semantic_table

    import xorq.api as xo
    from xorq.vendor.ibis.expr.types.core import _extract_builders

    c = xo.connect()
    table = c.create_table("eb_test", {"origin": ["JFK"], "delay": [10.0]})
    model = (
        to_semantic_table(table)
        .with_dimensions(origin=lambda t: t.origin)
        .with_measures(avg_delay=lambda t: t.delay.mean())
    )
    expr = model.query(dimensions=("origin",), measures=("avg_delay",)).to_tagged()
    builders = _extract_builders(expr)
    assert len(builders) >= 1
    b = builders[0]
    assert b["type"] == "semantic_model"
    assert "origin" in b["dimensions"]
    assert "avg_delay" in b["measures"]


# ---------------------------------------------------------------------------
# Integration tests — zip validation
# ---------------------------------------------------------------------------


def test_zip_accepts_builder_entry():
    import zipfile

    from xorq.catalog.zip_utils import test_zip

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "builder.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(BUILDER_META_FILENAME, json.dumps({"type": "test"}))
        test_zip(zip_path)


def test_zip_rejects_invalid():
    import zipfile

    from xorq.catalog.zip_utils import test_zip

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "invalid.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("random.txt", "nope")
        with pytest.raises(AssertionError, match="neither a valid expression"):
            test_zip(zip_path)
