"""Tests for ExprKind.ExprBuilder detection, from_tagged registry, and sidecar roundtrip."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import pytest

import xorq.api as xo
from xorq.catalog.zip_utils import test_zip as validate_zip
from xorq.expr.builders import (
    _FROM_TAGGED_REGISTRY,
    TagHandler,
    extract_builder_metadata,
    from_tagged_dispatch,
    get_from_tagged_registry,
    register_tag_handler,
)
from xorq.expr.relations import Tag
from xorq.ibis_yaml.enums import ExprKind
from xorq.vendor.ibis.common.collections import FrozenOrderedDict
from xorq.vendor.ibis.expr.types.core import (
    ExprMetadata,
    _extract_builders,
    _extract_kind,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def saved_registry():
    """Save and restore the handler registry around a test."""
    saved = dict(_FROM_TAGGED_REGISTRY)
    yield
    _FROM_TAGGED_REGISTRY.clear()
    _FROM_TAGGED_REGISTRY.update(saved)


@pytest.fixture
def con():
    return xo.connect()


# ---------------------------------------------------------------------------
# ExprKind.ExprBuilder detection
# ---------------------------------------------------------------------------


def test_expr_builder_enum_value():
    assert str(ExprKind.ExprBuilder) == "expr_builder"


def test_extract_kind_with_builders():
    kind = _extract_kind(
        unbound_node=None,
        catalog_tag_nodes=[],
        is_source=False,
        has_builders=True,
    )
    assert kind == ExprKind.ExprBuilder


def test_extract_kind_unbound_takes_priority():
    sentinel = type("FakeNode", (), {"schema": None})()
    kind = _extract_kind(
        unbound_node=sentinel,
        catalog_tag_nodes=[],
        is_source=False,
        has_builders=True,
    )
    assert kind == ExprKind.UnboundExpr


def test_extract_kind_builder_over_composed():
    kind = _extract_kind(
        unbound_node=None,
        catalog_tag_nodes=["something"],
        is_source=False,
        has_builders=True,
    )
    assert kind == ExprKind.ExprBuilder


def test_extract_kind_no_builders_falls_through():
    kind = _extract_kind(
        unbound_node=None,
        catalog_tag_nodes=[],
        is_source=True,
        has_builders=False,
    )
    assert kind == ExprKind.Source


# ---------------------------------------------------------------------------
# from_tagged registry
# ---------------------------------------------------------------------------


def test_register_and_retrieve(saved_registry):
    handler = TagHandler(
        extract_metadata=lambda tag_node: {"type": "test_dummy"},
        from_tagged=lambda tag_node: "dummy",
    )
    register_tag_handler("test_dummy", handler)

    assert "test_dummy" in _FROM_TAGGED_REGISTRY
    assert _FROM_TAGGED_REGISTRY["test_dummy"] is handler


def test_get_registry_returns_dict():
    registry = get_from_tagged_registry()
    assert isinstance(registry, dict)


def test_third_party_handler_roundtrip(saved_registry, con):
    """A third-party handler can extract metadata and recover a domain object."""
    handler = TagHandler(
        extract_metadata=lambda tag_node: {
            "type": "weather_model",
            "description": "3 features",
            "features": tag_node.metadata.get("features", ()),
        },
        from_tagged=lambda tag_node: {
            "recovered": True,
            "features": tag_node.metadata.get("features"),
        },
    )
    register_tag_handler("weather_model", handler)

    table = con.create_table("weather", {"temp": [72.0], "wind": [5.0]})
    tag_meta = FrozenOrderedDict(
        {
            "tag": "weather_model",
            "features": ("temp", "wind", "humidity"),
        }
    )
    tagged_expr = Tag(
        schema=table.schema(),
        parent=table.op(),
        metadata=tag_meta,
    ).to_expr()

    # extract_builder_metadata dispatches to our handler
    meta = extract_builder_metadata(
        "weather_model",
        Tag(
            schema=table.schema(),
            parent=table.op(),
            metadata=tag_meta,
        ),
    )
    assert meta["type"] == "weather_model"
    assert meta["features"] == ("temp", "wind", "humidity")

    # from_tagged_dispatch recovers the domain object
    recovered = from_tagged_dispatch(tagged_expr)
    assert recovered["recovered"] is True
    assert recovered["features"] == ("temp", "wind", "humidity")

    # ExprMetadata detects ExprBuilder kind
    expr_meta = ExprMetadata.from_expr(tagged_expr)
    assert expr_meta.kind == ExprKind.ExprBuilder
    assert len(expr_meta.builders) == 1
    assert expr_meta.builders[0]["type"] == "weather_model"


# ---------------------------------------------------------------------------
# ExprMetadata sidecar roundtrip — builders field
# ---------------------------------------------------------------------------


def test_from_dict_with_builders():
    data = {
        "kind": "expr_builder",
        "schema_out": {"a": "int64"},
        "builders": ({"type": "fitted_pipeline", "description": "test"},),
    }
    meta = ExprMetadata.from_dict(data)
    assert len(meta.builders) == 1
    assert meta.builders[0]["type"] == "fitted_pipeline"
    assert meta.kind == ExprKind.ExprBuilder


def test_from_dict_without_builders_backward_compat():
    data = {
        "kind": "expr",
        "schema_out": {"a": "int64"},
    }
    meta = ExprMetadata.from_dict(data)
    assert meta.builders == ()


def test_to_dict_with_builders():
    data = {
        "kind": "expr_builder",
        "schema_out": {"a": "int64"},
        "builders": ({"type": "bsl", "description": "test"},),
    }
    meta = ExprMetadata.from_dict(data)
    d = meta.to_dict()
    assert "builders" in d
    assert d["builders"][0]["type"] == "bsl"


def test_to_dict_without_builders_omits_key():
    data = {
        "kind": "expr",
        "schema_out": {"a": "int64"},
    }
    meta = ExprMetadata.from_dict(data)
    d = meta.to_dict()
    assert "builders" not in d


# ---------------------------------------------------------------------------
# Integration tests — _extract_builders BSL traversal
# ---------------------------------------------------------------------------


@pytest.mark.library
def test_extract_builders_bsl_traversal():
    bsl = pytest.importorskip("boring_semantic_layer")
    xo = pytest.importorskip("xorq.api")

    c = xo.connect()
    table = c.create_table("eb_test", {"origin": ["JFK"], "delay": [10.0]})
    model = (
        bsl.to_semantic_table(table)
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
# Integration tests — ExprKind.ExprBuilder detection from real expression
# ---------------------------------------------------------------------------


@pytest.mark.library
def test_expr_metadata_detects_expr_builder():
    bsl = pytest.importorskip("boring_semantic_layer")
    xo = pytest.importorskip("xorq.api")

    c = xo.connect()
    table = c.create_table("ek_test", {"x": ["a"], "v": [1.0]})
    model = (
        bsl.to_semantic_table(table)
        .with_dimensions(x=lambda t: t.x)
        .with_measures(total=lambda t: t.v.sum())
    )
    expr = model.query(dimensions=("x",), measures=("total",)).to_tagged()
    meta = ExprMetadata.from_expr(expr)
    assert meta.kind == ExprKind.ExprBuilder
    assert len(meta.builders) >= 1


# ---------------------------------------------------------------------------
# Integration tests — catalog add(tagged_expr)
# ---------------------------------------------------------------------------


@pytest.mark.library
def test_catalog_add_tagged_expr():
    bsl = pytest.importorskip("boring_semantic_layer")
    xo = pytest.importorskip("xorq.api")
    catalog_mod = pytest.importorskip("xorq.catalog.catalog")

    c = xo.connect()
    table = c.create_table("cat_eb", {"origin": ["JFK", "LAX"], "delay": [10.0, -5.0]})
    model = (
        bsl.to_semantic_table(table)
        .with_dimensions(origin=lambda t: t.origin)
        .with_measures(avg_delay=lambda t: t.delay.mean())
    )
    tagged_expr = model.query(
        dimensions=("origin",), measures=("avg_delay",)
    ).to_tagged()

    with tempfile.TemporaryDirectory() as tmp:
        catalog = catalog_mod.Catalog.from_repo_path(Path(tmp) / "cat", init=True)
        entry = catalog.add(tagged_expr, aliases=("test-sem",), sync=False)
        assert entry is not None
        assert entry.kind == ExprKind.ExprBuilder


# ---------------------------------------------------------------------------
# Integration tests — from_tagged_dispatch BSL recovery
# ---------------------------------------------------------------------------


@pytest.mark.library
def test_from_tagged_dispatch_bsl():
    bsl = pytest.importorskip("boring_semantic_layer")
    xo = pytest.importorskip("xorq.api")

    c = xo.connect()
    table = c.create_table("ftd_test", {"x": ["a"], "v": [1.0]})
    model = (
        bsl.to_semantic_table(table)
        .with_dimensions(x=lambda t: t.x)
        .with_measures(total=lambda t: t.v.sum())
    )
    expr = model.query(dimensions=("x",), measures=("total",)).to_tagged()
    recovered = from_tagged_dispatch(expr)
    # recovered should be a SemanticModel (from BSL)
    assert hasattr(recovered, "query")
    assert hasattr(recovered, "dimensions")


# ---------------------------------------------------------------------------
# Integration tests — zip validation (expression entries only now)
# ---------------------------------------------------------------------------


def test_zip_rejects_invalid():
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "invalid.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("random.txt", "nope")
        with pytest.raises(AssertionError, match="not a valid expression"):
            validate_zip(zip_path)


# ---------------------------------------------------------------------------
# Integration tests — ML pipeline from_tagged
# ---------------------------------------------------------------------------

sklearn = pytest.importorskip("sklearn")

from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from xorq.common.utils.graph_utils import walk_nodes  # noqa: E402
from xorq.expr.ml.enums import FittedPipelineTagKey  # noqa: E402
from xorq.expr.ml.pipeline_lib import FittedPipeline, Pipeline  # noqa: E402


@pytest.fixture
def ml_train_expr(con):
    return con.create_table(
        "ml_train",
        {
            "feature_0": [1.0, 2.0, 3.0],
            "feature_1": [4.0, 5.0, 6.0],
            "target": [0, 1, 0],
        },
    )


@pytest.fixture
def ml_fitted(ml_train_expr):
    sk_pipe = sklearn.pipeline.make_pipeline(StandardScaler(), LinearRegression())
    pipeline = Pipeline.from_instance(sk_pipe)
    return pipeline.fit(ml_train_expr, target="target")


def test_ml_training_tag_present(ml_train_expr, ml_fitted):
    """Verify FittedPipeline-training tag is discoverable via walk_nodes."""
    predict_expr = ml_fitted.predict(ml_train_expr)
    tags = walk_nodes((Tag,), predict_expr)
    training_tags = [
        t for t in tags if t.metadata.get("tag") == str(FittedPipelineTagKey.TRAINING)
    ]
    assert len(training_tags) == 1
    meta = training_tags[0].metadata
    assert meta["target"] == "target"
    assert set(meta["features"]) == {"feature_0", "feature_1"}


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
    """Verify from_tagged_dispatch on a predict expr returns a FittedPipeline."""
    predict_expr = ml_fitted.predict(ml_train_expr)
    recovered = from_tagged_dispatch(predict_expr)
    assert isinstance(recovered, FittedPipeline)
    result = recovered.predict(ml_train_expr)
    assert result is not None
    assert "prediction" in result.columns or len(result.columns) > 0
