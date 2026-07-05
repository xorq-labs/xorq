"""Unit tests for the declarative transform driver (``xorq.expr.transform``).

These pin the driver's *own* guarantees -- the ones R2 introduced -- independent
of any concrete pass:

- P3: ``run_transform_passes`` raises when a pass's ``after`` dependency is not
  positioned earlier, instead of silently mis-transforming.
- P2: ``apply_pass`` selects the graph walk (descend into opaque sub-exprs vs
  stop at them) solely from the pass's ``Traversal``.

The pipeline-level tests in ``test_transform_scope.py`` cover the real passes;
this covers the mechanism they ride on.
"""

from __future__ import annotations

from typing import Callable

import pytest

import xorq.api as xo
from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.enums import Traversal
from xorq.expr.relations import RemoteTable
from xorq.expr.remote_table_exec import RemoteTableScope
from xorq.expr.transform import (
    TransformCtx,
    TransformPass,
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
    with pytest.raises(AssertionError, match=r"must run after.*cache"):
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
