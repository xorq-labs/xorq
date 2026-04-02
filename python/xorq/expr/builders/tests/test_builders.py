"""Tests for ExprKind.ExprBuilder detection, from_tagged registry, and sidecar roundtrip."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import pytest

from xorq.catalog.zip_utils import test_zip as validate_zip
from xorq.expr.builders import (
    _FROM_TAGGED_REGISTRY,
    from_tagged_dispatch,
    get_from_tagged_registry,
    register_from_tagged,
)
from xorq.ibis_yaml.enums import ExprKind
from xorq.vendor.ibis.expr.types.core import (
    ExprMetadata,
    _extract_builders,
    _extract_kind,
)


# ---------------------------------------------------------------------------
# ExprKind.ExprBuilder detection
# ---------------------------------------------------------------------------


class TestExprKindExprBuilder:
    def test_enum_value(self):
        assert str(ExprKind.ExprBuilder) == "expr_builder"

    def test_extract_kind_with_builders(self):
        kind = _extract_kind(
            unbound_node=None,
            catalog_tag_nodes=[],
            is_source=False,
            has_builders=True,
        )
        assert kind == ExprKind.ExprBuilder

    def test_extract_kind_unbound_takes_priority(self):
        """UnboundExpr has higher priority than ExprBuilder."""
        sentinel = type("FakeNode", (), {"schema": None})()
        kind = _extract_kind(
            unbound_node=sentinel,
            catalog_tag_nodes=[],
            is_source=False,
            has_builders=True,
        )
        assert kind == ExprKind.UnboundExpr

    def test_extract_kind_builder_over_composed(self):
        """ExprBuilder has higher priority than Composed."""
        kind = _extract_kind(
            unbound_node=None,
            catalog_tag_nodes=["something"],
            is_source=False,
            has_builders=True,
        )
        assert kind == ExprKind.ExprBuilder

    def test_extract_kind_no_builders_falls_through(self):
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


class TestFromTaggedRegistry:
    def test_register_and_retrieve(self):
        saved = dict(_FROM_TAGGED_REGISTRY)
        try:

            @register_from_tagged("test_dummy")
            def dummy_from_tagged(tag_node):
                return "dummy"

            assert "test_dummy" in _FROM_TAGGED_REGISTRY
            assert _FROM_TAGGED_REGISTRY["test_dummy"] is dummy_from_tagged
        finally:
            _FROM_TAGGED_REGISTRY.clear()
            _FROM_TAGGED_REGISTRY.update(saved)

    def test_get_registry_returns_dict(self):
        registry = get_from_tagged_registry()
        assert isinstance(registry, dict)


# ---------------------------------------------------------------------------
# ExprMetadata sidecar roundtrip — builders field
# ---------------------------------------------------------------------------


class TestExprMetadataBuilders:
    def test_from_dict_with_builders(self):
        data = {
            "kind": "expr_builder",
            "schema_out": {"a": "int64"},
            "builders": ({"type": "fitted_pipeline", "description": "test"},),
        }
        meta = ExprMetadata.from_dict(data)
        assert len(meta.builders) == 1
        assert meta.builders[0]["type"] == "fitted_pipeline"
        assert meta.kind == ExprKind.ExprBuilder

    def test_from_dict_without_builders_backward_compat(self):
        data = {
            "kind": "expr",
            "schema_out": {"a": "int64"},
        }
        meta = ExprMetadata.from_dict(data)
        assert meta.builders == ()

    def test_to_dict_with_builders(self):
        data = {
            "kind": "expr_builder",
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
