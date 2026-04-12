"""Tests for ExprKind.ExprBuilder detection, from_tag_node registry, and sidecar roundtrip."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from unittest.mock import Mock

import pytest

import xorq.api as xo
from xorq.catalog.zip_utils import test_zip as validate_zip
from xorq.expr.builders import (
    _FROM_TAG_NODE_REGISTRY,
    TagHandler,
    _discover_from_tag_node,
    _get_from_tag_node_registry,
    _reset_registry,
    _resolve_builder_from_tag,
    extract_builder_metadata,
    register_tag_handler,
)
from xorq.expr.relations import Tag
from xorq.ibis_yaml.enums import ExprKind
from xorq.vendor.ibis.common.collections import FrozenOrderedDict
from xorq.vendor.ibis.expr.types.core import (
    ExprMetadata,
    ExprTraits,
    _extract_builders,
    _extract_kind,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def saved_registry():
    """Save and restore the handler registry around a test."""
    import xorq.expr.builders as _builders_mod  # noqa: PLC0415

    saved = dict(_FROM_TAG_NODE_REGISTRY)
    saved_keys = _builders_mod._BUILTIN_KEYS
    saved_init = _builders_mod._initialized
    yield
    _FROM_TAG_NODE_REGISTRY.clear()
    _FROM_TAG_NODE_REGISTRY.update(saved)
    _builders_mod._BUILTIN_KEYS = saved_keys
    _builders_mod._initialized = saved_init


@pytest.fixture
def con():
    return xo.connect()


# ---------------------------------------------------------------------------
# ExprKind detection — outermost-only
# ---------------------------------------------------------------------------


def test_expr_builder_enum_value():
    assert str(ExprKind.ExprBuilder) == "expr_builder"


def test_extract_kind_outermost_builder_tag(saved_registry, con):
    """Outermost builder tag → ExprBuilder."""
    handler = TagHandler(
        tag_names=("test_builder",),
        extract_metadata=lambda tag_node: {"type": "test_builder"},
    )
    register_tag_handler(handler)
    table = con.create_table("kind_test", {"x": [1]})
    expr = table.tag("test_builder")
    assert _extract_kind(expr) == ExprKind.ExprBuilder


def test_extract_kind_source():
    """Plain source table → Source."""
    table = xo.memtable({"x": [1]}, name="kind_src")
    assert _extract_kind(table) == ExprKind.Source


def test_extract_kind_expr():
    """Projection on source → Expr."""
    table = xo.memtable({"x": [1], "y": [2]}, name="kind_expr")
    expr = table.select("x")
    assert _extract_kind(expr) == ExprKind.Expr


def test_extract_kind_unrecognized_tag_unwraps_to_source(con):
    """Unrecognized tag on a source → Source (tag is decorative)."""
    table = con.create_table("kind_unrec", {"x": [1]})
    expr = table.tag("debug_info")
    assert _extract_kind(expr) == ExprKind.Source


# ---------------------------------------------------------------------------
# from_tag_node registry
# ---------------------------------------------------------------------------


def test_resolve_builder_no_handler_raises(con):
    """_resolve_builder_from_tag raises ValueError when no handler matches."""
    table = con.create_table("no_handler", {"x": [1]})
    expr = table.tag("completely_unknown_tag_xyz")
    with pytest.raises(ValueError, match="No builder tags found in expression"):
        _resolve_builder_from_tag(expr)


def test_register_and_retrieve(saved_registry):
    handler = TagHandler(
        tag_names=("test_dummy",),
        extract_metadata=lambda tag_node: {"type": "test_dummy"},
        from_tag_node=lambda tag_node: "dummy",
    )
    register_tag_handler(handler)

    assert "test_dummy" in _FROM_TAG_NODE_REGISTRY
    assert _FROM_TAG_NODE_REGISTRY["test_dummy"] is handler


def test_get_registry_returns_dict():
    registry = _get_from_tag_node_registry()
    assert isinstance(registry, dict)


def test_third_party_handler_roundtrip(saved_registry, con):
    """A third-party handler can extract metadata and recover a domain object."""
    handler = TagHandler(
        tag_names=("weather_model",),
        extract_metadata=lambda tag_node: {
            "type": "weather_model",
            "description": "3 features",
            "features": tag_node.metadata.get("features", ()),
        },
        from_tag_node=lambda tag_node: {
            "recovered": True,
            "features": tag_node.metadata.get("features"),
        },
    )
    register_tag_handler(handler)

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
        Tag(
            schema=table.schema(),
            parent=table.op(),
            metadata=tag_meta,
        ),
    )
    assert meta["type"] == "weather_model"
    assert meta["features"] == ("temp", "wind", "humidity")

    # _resolve_builder_from_tag recovers the domain object
    recovered = _resolve_builder_from_tag(tagged_expr)
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
# Integration tests — _resolve_builder_from_tag BSL recovery
# ---------------------------------------------------------------------------


@pytest.mark.library
def test__resolve_builder_from_tag_bsl():
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
    recovered = _resolve_builder_from_tag(expr)
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
# register_tag_handler duplicate key guard
# ---------------------------------------------------------------------------

from xorq.expr.ml.enums import FittedPipelineTagKey  # noqa: E402


def test_register_duplicate_raises(saved_registry):
    handler = TagHandler(
        tag_names=("dup_test",), from_tag_node=lambda tag_node: "first"
    )
    register_tag_handler(handler)
    with pytest.raises(ValueError, match="already registered"):
        register_tag_handler(handler)


def test_register_duplicate_override(saved_registry):
    first = TagHandler(tag_names=("dup_test",), from_tag_node=lambda tag_node: "first")
    second = TagHandler(
        tag_names=("dup_test",), from_tag_node=lambda tag_node: "second"
    )
    register_tag_handler(first)
    register_tag_handler(second, override=True)
    assert _FROM_TAG_NODE_REGISTRY["dup_test"] is second


def test_register_builtin_key_raises(saved_registry):
    handler_bsl = TagHandler(
        tag_names=("bsl",), from_tag_node=lambda tag_node: "hijack"
    )
    with pytest.raises(ValueError, match="protected builtin"):
        register_tag_handler(handler_bsl)
    handler_predict = TagHandler(
        tag_names=("FittedPipeline-predict",),
        from_tag_node=lambda tag_node: "hijack",
    )
    with pytest.raises(ValueError, match="protected builtin"):
        register_tag_handler(handler_predict)


# ---------------------------------------------------------------------------
# _ensure_initialized — builtins present even when third-party registers first
# ---------------------------------------------------------------------------


def test_register_before_get_preserves_builtins(saved_registry):
    """Third-party register_tag_handler before any get still has builtins."""
    _reset_registry()
    register_tag_handler(
        TagHandler(tag_names=("third_party",), from_tag_node=lambda n: "tp")
    )
    registry = _get_from_tag_node_registry()
    assert "third_party" in registry
    assert str(FittedPipelineTagKey.PREDICT) in registry
    assert str(FittedPipelineTagKey.TRANSFORM) in registry


def test_builtins_not_clobbered_by_third_party(saved_registry):
    """Registering a third-party handler doesn't overwrite builtins."""
    _reset_registry()
    register_tag_handler(
        TagHandler(tag_names=("custom",), from_tag_node=lambda n: "custom")
    )
    register_tag_handler(
        TagHandler(tag_names=("custom2",), from_tag_node=lambda n: "custom2")
    )
    registry = _get_from_tag_node_registry()
    assert "custom" in registry
    assert "custom2" in registry
    assert str(FittedPipelineTagKey.PREDICT) in registry
    assert "bsl" in registry


# ---------------------------------------------------------------------------
# _register_builtins excludes ALL_STEPS key from handler dispatch
# ---------------------------------------------------------------------------


def test_all_steps_key_not_registered():
    registry = _get_from_tag_node_registry()
    assert str(FittedPipelineTagKey.ALL_STEPS) not in registry
    assert str(FittedPipelineTagKey.PREDICT) in registry
    assert str(FittedPipelineTagKey.TRANSFORM) in registry


# ---------------------------------------------------------------------------
# register_tag_handler type guard
# ---------------------------------------------------------------------------


def test_register_tag_handler_rejects_non_taghandler():
    with pytest.raises(TypeError, match="expected TagHandler"):
        register_tag_handler("not a handler")


# ---------------------------------------------------------------------------
# extract_builder_metadata fallback — handler without extract_metadata
# ---------------------------------------------------------------------------


def test_extract_metadata_fallback_when_no_callback(saved_registry, con):
    handler = TagHandler(
        tag_names=("meta_fallback",),
        from_tag_node=lambda tag_node: "domain_obj",
    )
    register_tag_handler(handler)
    table = con.create_table("fb_test", {"x": [1]})
    tag_node = Tag(
        schema=table.schema(),
        parent=table.op(),
        metadata=FrozenOrderedDict({"tag": "meta_fallback"}),
    )
    meta = extract_builder_metadata(tag_node)
    assert meta == {"type": "meta_fallback"}


# ---------------------------------------------------------------------------
# _extract_kind — UnboundExpr path
# ---------------------------------------------------------------------------


def test_extract_kind_unbound():
    t = xo.table(schema={"a": "int64"})
    assert _extract_kind(t) == ExprKind.UnboundExpr


# ---------------------------------------------------------------------------
# ExprTraits
# ---------------------------------------------------------------------------


def test_expr_traits_plain_source():
    t = xo.memtable({"x": [1]}, name="traits_src")
    traits = t.ls.expr_traits
    assert isinstance(traits, ExprTraits)
    assert traits.is_source is True
    assert traits.has_unbound is False
    assert traits.has_composition is False
    assert traits.has_builders is False


def test_expr_traits_has_builders(saved_registry, con):
    handler = TagHandler(
        tag_names=("traits_builder",),
        extract_metadata=lambda tag_node: {"type": "traits_builder"},
    )
    register_tag_handler(handler)
    table = con.create_table("traits_bld", {"x": [1]})
    expr = table.tag("traits_builder")
    traits = expr.ls.expr_traits
    assert traits.has_builders is True


def test_expr_traits_has_unbound():
    t = xo.table(schema={"a": "int64"})
    traits = t.ls.expr_traits
    assert traits.has_unbound is True
    assert traits.is_source is False


# ---------------------------------------------------------------------------
# _discover_from_tag_node — entry-point loading
# ---------------------------------------------------------------------------


def test_discover_loads_valid_entry_point(saved_registry, monkeypatch):
    _get_from_tag_node_registry()  # ensure builtins are initialized

    fake_handler = TagHandler(tag_names=("ep_test",), from_tag_node=lambda n: "ep")
    ep = Mock()
    ep.name = "my_plugin"
    ep.load.return_value = fake_handler
    monkeypatch.setattr(
        "xorq.expr.builders.importlib.metadata.entry_points",
        lambda group: [ep],
    )
    handlers = _discover_from_tag_node()
    assert len(handlers) == 1
    assert handlers[0] is fake_handler


def test_discover_skips_non_taghandler(saved_registry, monkeypatch):
    _get_from_tag_node_registry()

    ep = Mock()
    ep.name = "bad_plugin"
    ep.load.return_value = "not a handler"
    monkeypatch.setattr(
        "xorq.expr.builders.importlib.metadata.entry_points",
        lambda group: [ep],
    )
    handlers = _discover_from_tag_node()
    assert handlers == []


def test_discover_skips_builtin_override(saved_registry, monkeypatch):
    _get_from_tag_node_registry()  # populates _BUILTIN_KEYS

    fake_handler = TagHandler(tag_names=("bsl",), from_tag_node=lambda n: "hijack")
    ep = Mock()
    ep.name = "hijack_plugin"
    ep.load.return_value = fake_handler
    monkeypatch.setattr(
        "xorq.expr.builders.importlib.metadata.entry_points",
        lambda group: [ep],
    )
    handlers = _discover_from_tag_node()
    assert handlers == []


def test_discover_skips_broken_entry_point(saved_registry, monkeypatch):
    _get_from_tag_node_registry()

    ep = Mock()
    ep.name = "broken_plugin"
    ep.load.side_effect = ImportError("no such module")
    monkeypatch.setattr(
        "xorq.expr.builders.importlib.metadata.entry_points",
        lambda group: [ep],
    )
    handlers = _discover_from_tag_node()
    assert handlers == []
