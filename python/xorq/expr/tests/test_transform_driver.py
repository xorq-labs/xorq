"""Unit tests for the declarative transform driver (``xorq.expr.transform``).

These pin the driver's *own* guarantees -- the ones R2 introduced -- independent
of any concrete pass:

- P3: ``run_transform_passes`` raises when a pass's ``after`` dependency is not
  positioned earlier, instead of silently mis-transforming.
- P2: ``apply_pass`` selects the graph walk (descend into opaque sub-exprs vs
  stop at them) solely from the pass's ``Traversal``.
- P4: adjacent fusable passes (``DESCEND`` and ``not produces_resources``) share
  one ``replace_nodes`` walk, ``produces_resources`` gates membership, and the
  fused result is identical to applying the passes one at a time.

The pipeline-level tests in ``test_transform_scope.py`` cover the real passes;
this covers the mechanism they ride on.
"""

from __future__ import annotations

from typing import Callable

import pytest

import xorq.api as xo
import xorq.common.utils.graph_utils as graph_utils
from xorq.common.exceptions import InternalError
from xorq.common.utils.graph_utils import walk_nodes
from xorq.common.utils.provenance_utils import get_expr_hash
from xorq.expr.api import (
    _make_bind_params_replacer,
    _make_remove_tag_nodes_replacer,
    _resolve_bind_op_params,
)
from xorq.expr.enums import Traversal
from xorq.expr.operations import NamedScalarParameter
from xorq.expr.relations import RemoteTable, Tag
from xorq.expr.remote_table_exec import RemoteTableScope
from xorq.expr.transform import (
    TransformCtx,
    TransformPass,
    _fuse_replacers,
    _fusion_groups,
    _is_fusable,
    apply_pass,
    run_transform_passes,
)


def _identity_replacer(seen: list | None = None) -> Callable:
    """A no-op replacer that optionally records every node it is called on."""

    def replacer(node, kwargs):
        if seen is not None:
            seen.append(node)
        if kwargs:
            node = node.__recreate__(kwargs)
        return node

    return replacer


def _ctx() -> TransformCtx:
    # The passes below produce no resources, so the scope is never touched.
    return TransformCtx(scope=RemoteTableScope())


def test_run_transform_passes_raises_on_out_of_order() -> None:
    """P3: a pass whose ``after`` dep is not positioned earlier fails loudly."""
    t = xo.memtable({"a": [1, 2, 3]})
    passes = (
        TransformPass(
            name="needs_cache",
            traversal=Traversal.DESCEND,
            build=lambda expr, ctx: _identity_replacer(),
            after=("cache",),
        ),
        TransformPass(
            name="cache",
            traversal=Traversal.DESCEND,
            build=lambda expr, ctx: _identity_replacer(),
        ),
    )
    with pytest.raises(InternalError, match=r"must run after.*cache"):
        run_transform_passes(t, passes, _ctx())


def test_run_transform_passes_runs_in_order() -> None:
    """Correctly ordered passes all run, each build invoked once, in order."""
    t = xo.memtable({"a": [1, 2, 3]})
    built: list[str] = []

    def build(name):
        def _build(expr, ctx):
            built.append(name)
            return _identity_replacer()

        return _build

    passes = (
        TransformPass(name="cache", traversal=Traversal.DESCEND, build=build("cache")),
        TransformPass(
            name="needs_cache",
            traversal=Traversal.DESCEND,
            build=build("needs_cache"),
            after=("cache",),
        ),
    )
    out = run_transform_passes(t, passes, _ctx())
    assert built == ["cache", "needs_cache"]
    assert out.schema() == t.schema()


def test_skipped_pass_still_satisfies_after() -> None:
    """A pass skipped via ``when`` is still 'positioned', so a later ``after`` on
    it is satisfied -- ordering is by position in the table, not by execution."""
    t = xo.memtable({"a": [1, 2, 3]})
    ran: list[str] = []

    def _after_build(expr, ctx):
        ran.append("after_maybe")
        return _identity_replacer()

    passes = (
        TransformPass(
            name="maybe",
            traversal=Traversal.DESCEND,
            build=lambda expr, ctx: _identity_replacer(),
            when=lambda expr, ctx: False,  # skipped, but still positioned
        ),
        TransformPass(
            name="after_maybe",
            traversal=Traversal.DESCEND,
            build=_after_build,
            after=("maybe",),
        ),
    )
    run_transform_passes(t, passes, _ctx())  # must not raise
    assert ran == ["after_maybe"]


def test_when_false_skips_the_pass() -> None:
    """``when`` returning False skips the build/walk entirely and returns the
    expression untouched."""
    t = xo.memtable({"a": [1, 2, 3]})
    built: list[str] = []

    skipped = TransformPass(
        name="skipped",
        traversal=Traversal.DESCEND,
        build=lambda expr, ctx: built.append("built") or _identity_replacer(),
        when=lambda expr, ctx: False,
    )
    out = apply_pass(skipped, t, _ctx())
    assert built == []
    assert out is t


def test_apply_pass_traversal_selected_from_record() -> None:
    """P2: ``apply_pass`` descends into opaque sub-exprs for DESCEND and stops at
    them for BOUNDARY, chosen solely from the pass's ``traversal`` field."""
    con = xo.connect()
    inner = xo.memtable({"a": [1, 2, 3]})
    expr = inner.into_backend(con, "rt")  # a RemoteTable: an opaque boundary
    assert walk_nodes(RemoteTable, expr), "test needs an opaque sub-expr present"

    descend_seen: list = []
    boundary_seen: list = []
    descend = TransformPass(
        name="d",
        traversal=Traversal.DESCEND,
        build=lambda expr, ctx: _identity_replacer(descend_seen),
    )
    boundary = TransformPass(
        name="b",
        traversal=Traversal.BOUNDARY,
        build=lambda expr, ctx: _identity_replacer(boundary_seen),
    )

    # identity replacers: neither materializes anything (no execution / no scope)
    apply_pass(descend, expr, _ctx())
    apply_pass(boundary, expr, _ctx())

    # DESCEND recurses through RemoteTable.remote_expr, so it visits strictly more
    # nodes than BOUNDARY, which stops at the RemoteTable.
    assert len(descend_seen) > len(boundary_seen)


# --- P4: fusion of adjacent pure DESCEND passes -----------------------------


def _descend(name: str, *, produces: bool = False) -> TransformPass:
    return TransformPass(
        name=name,
        traversal=Traversal.DESCEND,
        build=lambda expr, ctx: _identity_replacer(),
        produces_resources=produces,
    )


def _boundary(name: str) -> TransformPass:
    return TransformPass(
        name=name,
        traversal=Traversal.BOUNDARY,
        build=lambda expr, ctx: _identity_replacer(),
    )


def test_is_fusable_reads_produces_resources() -> None:
    """A pass is fusable iff it is DESCEND and produces no resources -- this is
    the one consumer of ``produces_resources``."""
    assert _is_fusable(_descend("pure"))
    assert not _is_fusable(_descend("effectful", produces=True))
    assert not _is_fusable(_boundary("bnd"))


def test_fusion_groups_coalesce_adjacent_pure_descend() -> None:
    """Maximal runs of fusable passes coalesce; a BOUNDARY or a DESCEND producer
    breaks the run and stays a singleton."""
    passes = (
        _descend("a"),
        _descend("b"),
        _boundary("c"),
        _descend("d", produces=True),  # DESCEND but effectful -> not fused
        _descend("e"),
        _descend("f"),
    )
    grouped = [tuple(p.name for p in g) for g in _fusion_groups(passes)]
    assert grouped == [("a", "b"), ("c",), ("d",), ("e", "f")]


def test_fused_run_uses_one_graph_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two adjacent fusable passes share a single ``replace_nodes`` traversal;
    the same two applied one at a time would walk twice."""
    calls = {"n": 0}
    real = graph_utils.replace_nodes

    def counting(replacer, expr):
        calls["n"] += 1
        return real(replacer, expr)

    monkeypatch.setattr(graph_utils, "replace_nodes", counting)

    t = xo.memtable({"a": [1, 2, 3]})
    passes = (_descend("a"), _descend("b"))
    run_transform_passes(t, passes, _ctx())
    assert calls["n"] == 1, "adjacent fusable passes must share one walk"

    calls["n"] = 0
    apply_pass(passes[0], t, _ctx())
    apply_pass(passes[1], t, _ctx())
    assert calls["n"] == 2, "applied singly, each pass walks on its own"


def test_effectful_descend_pass_is_not_folded_into_the_fused_walk() -> None:
    """A DESCEND pass flagged ``produces_resources`` runs on its own walk, never
    under the shared fused replacer -- so its side effect fires exactly once,
    under a traversal it owns."""
    fired: list[str] = []

    def effectful_build(expr, ctx):
        def replacer(node, kwargs):
            fired.append("walk")
            return node.__recreate__(kwargs) if kwargs else node

        return replacer

    t = xo.memtable({"a": [1, 2, 3]})
    passes = (
        _descend("pure_a"),
        TransformPass(
            name="effectful",
            traversal=Traversal.DESCEND,
            build=effectful_build,
            produces_resources=True,
        ),
        _descend("pure_b"),
    )
    # three groups: (pure_a,), (effectful,), (pure_b,) -- none fuse with effectful
    grouped = [tuple(p.name for p in g) for g in _fusion_groups(passes)]
    assert grouped == [("pure_a",), ("effectful",), ("pure_b",)]
    run_transform_passes(t, passes, _ctx())
    # the effectful replacer ran its own single walk (once per node, one walk)
    assert fired, "effectful pass must still run"


def test_when_false_collapses_group_to_single_pass_path() -> None:
    """When ``when`` leaves a single active pass in a fusion group, the driver
    falls back to the plain single-pass path (no composed replacer needed)."""
    t = xo.memtable({"a": [1, 2, 3]})
    ran: list[str] = []

    def rec(name):
        def build(expr, ctx):
            ran.append(name)
            return _identity_replacer()

        return build

    passes = (
        TransformPass(
            name="skipped",
            traversal=Traversal.DESCEND,
            build=rec("skipped"),
            when=lambda expr, ctx: False,
        ),
        TransformPass(name="kept", traversal=Traversal.DESCEND, build=rec("kept")),
    )
    run_transform_passes(t, passes, _ctx())
    assert ran == ["kept"], "skipped pass contributes no replacer"


def test_fused_bind_and_remove_tags_equals_sequential() -> None:
    """The real fusable pair: fusing ``bind_params`` + ``remove_tags`` into one
    walk yields an expression identical (by build hash) to applying them singly,
    including the interaction where a bound parameter lives inside a tagged
    subtree (a Tag wraps a relation; the param sits below it)."""
    t = xo.memtable({"a": [1, 2, 3]})
    p = xo.param("thresh", "int64")
    inner = t.filter(t.a > p)
    tagged = Tag(schema=inner.op().schema, parent=inner.op()).to_expr()
    assert walk_nodes(Tag, tagged) and walk_nodes(NamedScalarParameter, tagged)

    bind = TransformPass(
        name="bind_params",
        traversal=Traversal.DESCEND,
        build=lambda expr, ctx: _make_bind_params_replacer(
            _resolve_bind_op_params(expr, ctx.name_values)
        ),
    )
    tags = TransformPass(
        name="remove_tags",
        traversal=Traversal.DESCEND,
        build=lambda expr, ctx: _make_remove_tag_nodes_replacer(),
    )
    ctx = TransformCtx(name_values={"thresh": 2})

    fused = run_transform_passes(tagged, (bind, tags), ctx)
    # sequential: two separate walks, one pass each
    step1 = apply_pass(bind, tagged, ctx)
    sequential = apply_pass(tags, step1, ctx)

    assert not walk_nodes(Tag, fused) and not walk_nodes(NamedScalarParameter, fused)
    assert get_expr_hash(fused) == get_expr_hash(sequential)


def test_fuse_replacers_applies_in_order_recreates_centrally() -> None:
    """``_fuse_replacers`` applies replacers left-to-right and performs the single
    ``__recreate__`` itself, so no individual replacer ever sees the recreate
    ``kwargs`` -- recreation happens exactly once, independent of pass order."""
    order: list[str] = []
    saw_kwargs: list[str] = []

    def make(name):
        def replacer(node, kwargs):
            order.append(name)
            if kwargs:
                saw_kwargs.append(name)
                node = node.__recreate__(kwargs)
            return node

        return replacer

    fused = _fuse_replacers([make("first"), make("second")])
    t = xo.memtable({"a": [1, 2, 3]})
    graph_utils.replace_nodes(fused, t)

    assert order[:2] == ["first", "second"], "replacers apply in list order"
    assert saw_kwargs == [], (
        "_fuse_replacers recreates centrally; no replacer is handed the kwargs"
    )
