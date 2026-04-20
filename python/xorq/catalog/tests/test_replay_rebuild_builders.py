"""Tests for builder-subtree rebuild (ExprBuilder via TagHandler).

Covers the generalized ``catalog replay --rebuild`` capability that
delegates to registered ``TagHandler`` rebuild protocols:
  1. Handler-level ``reemit`` callable.
  2. Domain-object ``reemit(tag_node, rebuild_subexpr)`` method
     (multi-output builders like ``FittedPipeline``).
  3. Domain-object ``with_inputs_translated`` + ``expr`` (single-output
     builders like ``ExprComposer``).
"""

from pathlib import Path

import pytest

import xorq.api as xo
from xorq.catalog.backend import GitBackend
from xorq.catalog.bind import CatalogTag, bind
from xorq.catalog.catalog import Catalog
from xorq.catalog.replay import Replayer
from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.relations import HashingTag
from xorq.vendor.ibis.expr import operations as ops


@pytest.fixture
def saved_registry():
    """Save and restore the builder handler registry around a test."""
    import xorq.expr.builders as _builders_mod  # noqa: PLC0415
    from xorq.expr.builders import _FROM_TAG_NODE_REGISTRY  # noqa: PLC0415

    saved = dict(_FROM_TAG_NODE_REGISTRY)
    saved_keys = _builders_mod._BUILTIN_KEYS
    saved_init = _builders_mod._initialized
    yield
    _FROM_TAG_NODE_REGISTRY.clear()
    _FROM_TAG_NODE_REGISTRY.update(saved)
    _builders_mod._BUILTIN_KEYS = saved_keys
    _builders_mod._initialized = saved_init


def _replay_rebuild(source_catalog_obj, target_path):
    target = Catalog.from_repo_path(target_path, init=True)
    Replayer(from_catalog=source_catalog_obj, rebuild=True).replay(target)
    return target


# ---------------------------------------------------------------------------
# FittedPipeline dispatch-table coverage (no catalog integration needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tag_key_name, method_name",
    [
        ("TRANSFORM", "transform"),
        ("PREDICT", "predict"),
        ("PREDICT_PROBA", "predict_proba"),
        ("DECISION_FUNCTION", "decision_function"),
        ("FEATURE_IMPORTANCES", "feature_importances"),
    ],
)
def test_fitted_pipeline_reemit_table_coverage(tag_key_name, method_name):
    """The reemit dispatch table maps every entry point to a real method."""
    pytest.importorskip("sklearn")
    from xorq.expr.ml.enums import FittedPipelineTagKey  # noqa: PLC0415
    from xorq.expr.ml.pipeline_lib import (  # noqa: PLC0415
        _FITTED_PIPELINE_REEMIT_METHODS,
        FittedPipeline,
    )

    tag_key = FittedPipelineTagKey[tag_key_name]
    assert _FITTED_PIPELINE_REEMIT_METHODS[tag_key] == method_name
    assert callable(getattr(FittedPipeline, method_name))


def test_fitted_pipeline_training_reserved_from_reemit():
    """TRAINING and ALL_STEPS are interior tags, not reemit entry points."""
    pytest.importorskip("sklearn")
    from xorq.expr.ml.enums import FittedPipelineTagKey  # noqa: PLC0415
    from xorq.expr.ml.pipeline_lib import (  # noqa: PLC0415
        _FITTED_PIPELINE_REEMIT_METHODS,
    )

    assert FittedPipelineTagKey.TRAINING not in _FITTED_PIPELINE_REEMIT_METHODS
    assert FittedPipelineTagKey.ALL_STEPS not in _FITTED_PIPELINE_REEMIT_METHODS


# ---------------------------------------------------------------------------
# Protocol dispatch tests using test-only builders (no sklearn / no FittedPipeline)
# ---------------------------------------------------------------------------


def test_get_rebuild_dispatch_returns_none_without_handler(saved_registry):
    from xorq.expr.builders import get_rebuild_dispatch  # noqa: PLC0415

    raw = xo.memtable({"x": [1, 2, 3]}).tag("no_such_handler")
    assert get_rebuild_dispatch(raw.op()) is None


def test_get_rebuild_dispatch_handler_level_reemit_wins(saved_registry):
    """When TagHandler.reemit is set, it's used without calling from_tag_node."""
    from xorq.expr.builders import (  # noqa: PLC0415
        TagHandler,
        get_rebuild_dispatch,
        register_tag_handler,
    )

    calls = {"reemit": 0, "from_tag_node": 0}

    def handler_reemit(tag_node, rebuild_subexpr):
        calls["reemit"] += 1
        return tag_node.parent.to_expr()

    def from_tag_node(tag_node):
        calls["from_tag_node"] += 1
        return object()

    register_tag_handler(
        TagHandler(
            tag_names=("test_handler_reemit",),
            extract_metadata=lambda tag_node: {"type": "test_handler_reemit"},
            from_tag_node=from_tag_node,
            reemit=handler_reemit,
        )
    )

    raw = xo.memtable({"x": [1, 2, 3]}).tag("test_handler_reemit")
    dispatch = get_rebuild_dispatch(raw.op())
    assert callable(dispatch)
    # Calling the dispatch invokes the handler-level reemit, not from_tag_node.
    dispatch(lambda expr: expr)
    assert calls["reemit"] == 1
    assert calls["from_tag_node"] == 0


def test_get_rebuild_dispatch_single_output_sentinel(saved_registry):
    """A domain object with with_inputs_translated + expr returns the single-output sentinel."""
    from attr import field, frozen  # noqa: PLC0415

    from xorq.expr.builders import (  # noqa: PLC0415
        _SINGLE_OUTPUT_DISPATCH,
        TagHandler,
        get_rebuild_dispatch,
        register_tag_handler,
    )

    @frozen
    class FakeBuilder:
        tag = field()

        def with_inputs_translated(self, remap, to_catalog):
            return self

        @property
        def expr(self):
            return self.tag.parent.to_expr()

    register_tag_handler(
        TagHandler(
            tag_names=("test_single_out",),
            extract_metadata=lambda tag_node: {"type": "test_single_out"},
            from_tag_node=lambda tag_node: FakeBuilder(tag=tag_node),
        )
    )

    raw = xo.memtable({"x": [1, 2, 3]}).tag("test_single_out")
    dispatch = get_rebuild_dispatch(raw.op())
    assert isinstance(dispatch, tuple)
    sentinel, builder = dispatch
    assert sentinel == _SINGLE_OUTPUT_DISPATCH
    assert isinstance(builder, FakeBuilder)


def test_get_rebuild_dispatch_domain_object_reemit(saved_registry):
    """A domain object with a reemit method returns a callable."""
    from xorq.expr.builders import (  # noqa: PLC0415
        TagHandler,
        get_rebuild_dispatch,
        register_tag_handler,
    )

    calls = {"reemit": 0}

    class FakeBuilder:
        def reemit(self, tag_node, rebuild_subexpr):
            calls["reemit"] += 1
            return tag_node.parent.to_expr()

    register_tag_handler(
        TagHandler(
            tag_names=("test_multi_out",),
            extract_metadata=lambda tag_node: {"type": "test_multi_out"},
            from_tag_node=lambda tag_node: FakeBuilder(),
        )
    )

    raw = xo.memtable({"x": [1, 2, 3]}).tag("test_multi_out")
    dispatch = get_rebuild_dispatch(raw.op())
    assert callable(dispatch)
    dispatch(lambda expr: expr)
    assert calls["reemit"] == 1


def test_get_rebuild_dispatch_no_protocol_returns_none(saved_registry):
    """A domain object with neither reemit nor with_inputs_translated returns None."""
    from xorq.expr.builders import (  # noqa: PLC0415
        TagHandler,
        get_rebuild_dispatch,
        register_tag_handler,
    )

    class Bare:
        pass

    register_tag_handler(
        TagHandler(
            tag_names=("test_bare",),
            from_tag_node=lambda tag_node: Bare(),
        )
    )
    raw = xo.memtable({"x": [1, 2, 3]}).tag("test_bare")
    assert get_rebuild_dispatch(raw.op()) is None


# ---------------------------------------------------------------------------
# Integration tests via catalog rebuild + handler-level reemit
# ---------------------------------------------------------------------------


def test_rebuild_dispatches_handler_level_reemit(tmpdir, saved_registry):
    """End-to-end: handler-level reemit on a catalog entry gets invoked."""
    from xorq.expr.builders import TagHandler, register_tag_handler  # noqa: PLC0415

    calls = {"handler_reemit": 0}

    def handler_reemit(tag_node, rebuild_subexpr):
        calls["handler_reemit"] += 1
        return tag_node.parent.to_expr().tag("test_handler_reemit")

    register_tag_handler(
        TagHandler(
            tag_names=("test_handler_reemit",),
            extract_metadata=lambda tag_node: {"type": "test_handler_reemit"},
            reemit=handler_reemit,
        )
    )

    repo = Catalog.init_repo_path(Path(tmpdir).joinpath("src"))
    catalog = Catalog(backend=GitBackend(repo=repo))
    raw = xo.memtable({"x": [1, 2, 3]}).tag("test_handler_reemit")
    catalog.add(raw, aliases=("hre",))

    _replay_rebuild(catalog, Path(tmpdir).joinpath("tgt"))
    assert calls["handler_reemit"] >= 1


def test_rebuild_refuses_schema_change(tmpdir, saved_registry):
    """Builder reemit that returns a schema-different expression is refused."""
    from xorq.expr.builders import TagHandler, register_tag_handler  # noqa: PLC0415

    def bad_reemit(tag_node, rebuild_subexpr):
        return xo.memtable({"completely": [1], "different": [2]})

    register_tag_handler(
        TagHandler(
            tag_names=("test_schema_change",),
            extract_metadata=lambda tag_node: {"type": "test_schema_change"},
            reemit=bad_reemit,
        )
    )

    repo = Catalog.init_repo_path(Path(tmpdir).joinpath("src"))
    catalog = Catalog(backend=GitBackend(repo=repo))
    raw = xo.memtable({"x": [1, 2, 3]}).tag("test_schema_change")
    entry = catalog.add(raw, aliases=("schematest",))

    target = Catalog.from_repo_path(Path(tmpdir).joinpath("tgt"), init=True)
    with pytest.raises(RuntimeError) as excinfo:
        Replayer(from_catalog=catalog, rebuild=True).replay(target)
    message = str(excinfo.value)
    assert entry.name in message
    assert "schema changed" in message


def test_rebuild_refuses_missing_protocol_silent_passthrough(tmpdir, saved_registry):
    """When the outer handler returns a bare object with no protocol, the tag
    is treated as non-rebuildable: the inner catalog composition is still
    rebuilt (by the driver's outermost-first walk), and the outer tag
    passes through on the rebuilt subtree."""
    from xorq.expr.builders import TagHandler, register_tag_handler  # noqa: PLC0415

    class Bare:
        pass

    register_tag_handler(
        TagHandler(
            tag_names=("test_bare",),
            from_tag_node=lambda tag_node: Bare(),
        )
    )

    repo = Catalog.init_repo_path(Path(tmpdir).joinpath("src"))
    catalog = Catalog(backend=GitBackend(repo=repo))
    source_expr = xo.memtable({"user_id": [1, 2, 3], "amount": [10.0, 20.0, 30.0]})
    source_entry = catalog.add(source_expr, aliases=("src",))
    unbound = ops.UnboundTable(name="p", schema=source_expr.schema()).to_expr()
    transform_entry = catalog.add(unbound.select("user_id", "amount"), aliases=("tx",))
    composed = bind(source_entry, transform_entry).tag("test_bare")
    catalog.add(composed, aliases=("bare_entry",))

    target = _replay_rebuild(catalog, Path(tmpdir).joinpath("tgt"))
    new_bare = target.get_catalog_entry("bare_entry", maybe_alias=True)

    from xorq.expr.relations import Tag  # noqa: PLC0415

    outer = new_bare.lazy_expr.op()
    assert isinstance(outer, Tag)
    assert outer.metadata["tag"] == "test_bare"

    # Inner catalog refs were translated — same check as the preserves-builder
    # test in test_replay_rebuild.py.
    inner_names = {
        ht.metadata["entry_name"]
        for ht in walk_nodes(HashingTag, new_bare.lazy_expr)
        if ht.metadata.get("tag") in frozenset(CatalogTag)
        and ht.metadata.get("entry_name")
    }
    assert inner_names <= set(target.list())
