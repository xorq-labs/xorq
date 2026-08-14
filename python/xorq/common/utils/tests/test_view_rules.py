"""Exhaustiveness and name-neutrality of the ``DatabaseTableView`` rule table.

``view_rules`` is the single source of truth for how the two normalization
regimes — the global hasher and ``SnapshotStrategy`` — identify
``DatabaseTable`` subclasses; its docstring records why the tables must not be
hand-mirrored (gh-610, gh-2229: drift leaked per-process ``gen_name()`` uuid4s
into keys).

Three layers here, deliberately overlapping:

1. ``test_no_unhandled_view_op_types`` — static sweep of
   ``DatabaseTableView.__subclasses__()``.  Cheap, but blind to view ops in
   lazily-imported backend modules.
2. ``test_guard_fires_in_both_regimes`` — the runtime guard that covers what the
   sweep cannot see, in *both* tables.
3. ``test_generated_names_do_not_reach_identity`` — the property that actually
   matters, asserted per op type under both regimes plus the build hash.  This
   is the test that would have caught gh-610 and gh-2229 mechanically.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import pytest
import toolz

import xorq.api as xo
from xorq.caching import (
    ParquetSnapshotCache,
)
from xorq.caching.strategy import (
    SnapshotStrategy,
    snapshot_normalize_read,
)
from xorq.common.utils.dasher import (
    HASHER,
)
from xorq.common.utils.dasher._relations import (
    _dispatch_databasetable,
    lookup_view_normalizer,
    unhandled_view_op_types,
    view_rules,
)
from xorq.common.utils.func_utils import (
    return_constant,
)
from xorq.common.utils.graph_utils import (
    walk_nodes,
)
from xorq.common.utils.provenance_utils import (
    get_expr_hash,
)
from xorq.expr.relations import (
    CachedNode,
    DatabaseTableView,
    FlightExpr,
    FlightUDXF,
    Read,
    RemoteTable,
    gen_name,
)
from xorq.vendor.ibis.expr import (
    operations as ops,
)


if TYPE_CHECKING:
    from xorq.vendor.ibis import Expr


echo_udxf = xo.expr.relations.flight_udxf(
    process_df=toolz.identity,
    maybe_schema_in=return_constant(True),
    maybe_schema_out=toolz.identity,
)


def _diamonds_path() -> pathlib.Path:
    return pathlib.Path(xo.options.pins.get_path("diamonds"))


def build_read(i: int) -> Expr:
    """A deferred read; ``Read.name`` is caller-supplied and must not matter.

    ``con.read_parquet`` registers eagerly and yields a plain ``DatabaseTable``;
    only the deferred form produces a ``Read`` op.
    """
    return xo.deferred_read_parquet(_diamonds_path(), xo.connect(), f"diamonds_{i}")


def build_cached_node(i: int) -> Expr:
    """``CachedNode.name`` is a fresh ``gen_name()`` per construction."""
    con = xo.connect()
    return con.read_parquet(_diamonds_path(), "diamonds").cache(
        ParquetSnapshotCache.from_kwargs(source=con)
    )


def build_remote_table(i: int) -> Expr:
    """``RemoteTable.name`` is a fresh ``gen_name()`` per construction."""
    con, other_con = xo.connect(), xo.connect()
    return con.read_parquet(_diamonds_path(), "diamonds").into_backend(other_con)


def build_flight_expr(i: int) -> Expr:
    con = xo.connect()
    t = con.read_parquet(_diamonds_path(), "diamonds")
    return xo.expr.relations.flight_expr(
        t,
        xo.table(t.schema(), name="unbound"),
        inner_name=f"inner_name-{i}",
    )


def build_flight_udxf(i: int) -> Expr:
    con = xo.connect()
    return con.read_parquet(_diamonds_path(), "diamonds").pipe(
        echo_udxf, name="echo", inner_name=f"inner_name-{i}"
    )


class ViewOpSpec(NamedTuple):
    """How to build one view op twice, and whether its own ``name`` varies.

    ``name_varies`` records whether *this op's* ``name`` field differs between
    two builds, which is what makes the neutrality assertion non-vacuous.  It is
    False for ``CachedNode`` alone: that op carries the fixed sentinel
    ``xorq_cached_node_name_placeholder`` rather than a ``gen_name()``, so its
    per-process randomness lives in the ops beneath it.  Recorded explicitly so
    a future op that *starts* generating names cannot quietly be tested
    vacuously.
    """

    build: Callable[[int], Expr]
    name_varies: bool


# One spec per row of ``view_rules``.  A new row without a spec is a FAILURE,
# never a skip -- a skipped case is how this class of bug survives (gh-2229's
# three pre-existing name-neutrality tests all passed while the bug was live).
VIEW_OP_BUILDERS = {
    Read: ViewOpSpec(build_read, name_varies=True),
    CachedNode: ViewOpSpec(build_cached_node, name_varies=False),
    RemoteTable: ViewOpSpec(build_remote_table, name_varies=True),
    FlightExpr: ViewOpSpec(build_flight_expr, name_varies=True),
    FlightUDXF: ViewOpSpec(build_flight_udxf, name_varies=True),
}


def _in_library_module(cls: type) -> bool:
    """Exclude view ops defined inside test modules.

    Test-local subclasses (see ``test_guard_fires_in_both_regimes``) linger in
    ``__subclasses__()`` until garbage collected, which would otherwise make the
    static sweep order-dependent.
    """
    module = cls.__module__
    return module.startswith("xorq.") and ".tests" not in module


def test_no_unhandled_view_op_types() -> None:
    """Every imported ``DatabaseTableView`` subclass has a ``view_rules`` row."""
    unhandled = tuple(
        cls for cls in unhandled_view_op_types() if _in_library_module(cls)
    )
    assert not unhandled, (
        f"DatabaseTableView subclasses with no normalizer: "
        f"{sorted(cls.__name__ for cls in unhandled)}. Add a row to "
        f"xorq.common.utils.dasher._relations.view_rules and a builder to "
        f"VIEW_OP_BUILDERS in this module."
    )


def test_every_view_rule_has_a_builder() -> None:
    """The property test below must cover every row -- fail, never skip."""
    missing = tuple(
        rule.op_type for rule in view_rules() if rule.op_type not in VIEW_OP_BUILDERS
    )
    assert not missing, (
        f"view_rules rows with no builder in VIEW_OP_BUILDERS: "
        f"{sorted(cls.__name__ for cls in missing)}. Without one, "
        f"test_generated_names_do_not_reach_identity silently stops covering it."
    )


def test_view_rules_covers_exactly_the_known_view_ops() -> None:
    """Pins the row set, so adding an op type is a deliberate, reviewed edit."""
    assert tuple(rule.op_type for rule in view_rules()) == (
        Read,
        CachedNode,
        RemoteTable,
        FlightExpr,
        FlightUDXF,
    )


def test_only_read_diverges_between_regimes() -> None:
    """The two columns agree everywhere except ``Read``.

    ``Read`` legitimately differs: stat-based identity globally, path-only under
    snapshot (gh-1861).  Every other row must be the *same callable* in both
    columns -- that identity is the invariant whose absence let gh-2229 through,
    so assert it rather than trusting the table to stay tidy.
    """
    diverging = {
        rule.op_type
        for rule in view_rules()
        if rule.normalizer is not rule.snapshot_normalizer
    }
    assert diverging == {Read}
    (read_rule,) = (rule for rule in view_rules() if rule.op_type is Read)
    assert read_rule.snapshot_normalizer is snapshot_normalize_read


@pytest.mark.parametrize(
    "snapshot",
    [
        pytest.param(False, id="global"),
        pytest.param(True, id="snapshot"),
    ],
)
def test_guard_fires_in_both_regimes(snapshot: bool) -> None:
    """An unhandled view raises in *both* dispatch tables.

    The guard originally lived only in the snapshot table, so on the global side
    an unhandled view either died with a misleading backend planning error or
    -- on a fall-through backend -- silently folded its uuid4 ``name`` in, which
    is the very defect gh-2229 was about.
    """

    class UnhandledView(DatabaseTableView):
        pass

    dt = UnhandledView(
        name=gen_name(),
        schema=xo.schema({"a": "int64"}),
        source=xo.connect(),
    )
    with pytest.raises(NotImplementedError, match="UnhandledView"):
        lookup_view_normalizer(dt, snapshot=snapshot)


def test_guard_fires_through_the_public_dispatchers() -> None:
    """Same guard, reached the way the hashers actually reach it."""

    class UnhandledView(DatabaseTableView):
        pass

    dt = UnhandledView(
        name=gen_name(),
        schema=xo.schema({"a": "int64"}),
        source=xo.connect(),
    )
    with pytest.raises(NotImplementedError, match="UnhandledView"):
        _dispatch_databasetable(dt)
    with pytest.raises(NotImplementedError, match="UnhandledView"):
        SnapshotStrategy.normalize_databasetable(dt)


def test_genuine_backend_table_still_folds_name_in() -> None:
    """The fallback must keep ``name``: for a real table it *is* the identity.

    The gh-2229 fix narrows the fallback's reach rather than dropping ``name``
    from it -- blanket-dropping would under-key and risk wrong cache hits.
    """
    con = xo.connect()
    t0 = con.read_parquet(_diamonds_path(), "diamonds_a")
    t1 = con.read_parquet(_diamonds_path(), "diamonds_b")
    dt0, dt1 = (
        ops.DatabaseTable(
            name=t.op().name,
            schema=t.schema(),
            source=con,
        )
        for t in (t0, t1)
    )
    assert lookup_view_normalizer(dt0, snapshot=True) is None
    assert SnapshotStrategy.normalize_databasetable(
        dt0
    ) != SnapshotStrategy.normalize_databasetable(dt1)


@pytest.mark.parametrize(
    "op_type",
    [
        pytest.param(cls, id=cls.__name__)
        for cls in sorted(VIEW_OP_BUILDERS, key=lambda cls: cls.__name__)
    ],
)
def test_generated_names_do_not_reach_identity(op_type: type) -> None:
    """Two structurally identical exprs hash equal under every regime.

    Each builder is called twice, so every auto-generated name in the tree is
    freshly drawn.  Asserted under the global hasher, ``SnapshotStrategy``, and
    the build hash -- gh-2229 was invisible to the first and live in the other
    two, which is precisely why it survived three passing tests.
    """
    spec = VIEW_OP_BUILDERS[op_type]
    expr0, expr1 = spec.build(0), spec.build(1)

    nodes0 = tuple(walk(op_type, expr0))
    assert nodes0, f"builder for {op_type.__name__} produced no {op_type.__name__}"
    names0 = tuple(n.name for n in nodes0)
    names1 = tuple(n.name for n in walk(op_type, expr1))
    if spec.name_varies:
        assert names0 != names1, (
            f"{op_type.__name__} names did not differ between builds "
            f"({names0!r}); the name-neutrality assertions below would be vacuous."
        )
    else:
        assert names0 == names1, (
            f"{op_type.__name__} is recorded as not generating names, but its "
            f"names varied ({names0!r} vs {names1!r}); flip name_varies to True."
        )

    assert HASHER.tokenize(expr0) == HASHER.tokenize(expr1)
    assert SnapshotStrategy().calc_key(expr0) == SnapshotStrategy().calc_key(expr1)
    assert get_expr_hash(expr0) == get_expr_hash(expr1)


def walk(op_type: type, expr: Expr) -> tuple:
    """All ``op_type`` nodes in ``expr``, descending into opaque sub-exprs.

    ``op.find`` stops at ``Any``-typed fields, so a ``FlightUDXF`` nested under
    a ``RemoteTable.remote_expr`` is invisible to it.
    """
    return walk_nodes(op_type, expr)
