from __future__ import annotations

import inspect
import itertools
import operator
import re
import types

import pytest
import toolz

import xorq.api as xo
import xorq.expr.datatypes as dt
import xorq.expr.relations as rel
import xorq.expr.selectors as s
import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.operations as ops
from xorq.caching import SourceCache
from xorq.common.utils.graph_utils import (
    NON_EDGE_EXPR_FIELDS,
    OPAQUE_EDGES,
    OPAQUE_SPECS,
    OpaqueSpec,
    find_all_sources,
    gen_children_flight_leaf,
    gen_children_of,
    opaque_ops,
    replace_nodes,
    walk_nodes,
)
from xorq.expr.relations import Tag
from xorq.ml import deferred_fit_predict_sklearn
from xorq.vendor.ibis import Expr


LinearRegression = pytest.importorskip("sklearn.linear_model").LinearRegression


node_types = (
    ops.DatabaseTable,
    ops.SQLQueryResult,
    rel.CachedNode,
    rel.Read,
    rel.RemoteTable,
    # ExprScalarUDF has an expr we need to get to
    # FlightOperator has a dynamically generated connection: it should be passed a Profile instead
)


def make_expr():
    cons = (con0, con1, con2, con3) = (
        xo.connect(),
        xo.connect(),
        xo.duckdb.connect(),
        xo.connect(),
    )

    read_node0 = xo.examples.awards_players.fetch(con0)
    remote_node0 = read_node0.into_backend(con1)
    cached_node0 = remote_node0.cache(SourceCache.from_kwargs(source=con1))
    read_node1 = xo.examples.batting.fetch(con2)
    remote_node1 = read_node1.into_backend(con1)
    remote_node2 = cached_node0.join(
        remote_node1, predicates=["playerID", "yearID", "lgID"]
    ).into_backend(con3)
    cached_node1 = remote_node2.cache()
    expr = cached_node2 = cached_node1[lambda t: t.G == 1].cache()
    nodes = {
        rel.CachedNode: (
            cached_node0.op(),
            cached_node1.op(),
            cached_node2.op(),
        ),
        rel.Read: (
            read_node0.op(),
            read_node1.op(),
        ),
        rel.RemoteTable: (
            remote_node0.op(),
            remote_node1.op(),
            remote_node2.op(),
        ),
    }
    return (cons, nodes, expr)


def test_walk_nodes():
    (_, nodes, expr) = make_expr()
    node_types = tuple(nodes)
    walked_nodes = walk_nodes(node_types, expr)
    expected = sorted(
        ((k, set(v)) for k, v in nodes.items()),
        key=toolz.compose(operator.attrgetter("__name__"), operator.itemgetter(0)),
    )
    actual = sorted(
        ((k, set(v)) for k, v in toolz.groupby(type, walked_nodes).items()),
        key=toolz.compose(operator.attrgetter("__name__"), operator.itemgetter(0)),
    )
    assert actual == expected


def test_find_all_sources():
    (created_sources, _, expr) = make_expr()
    found_sources = find_all_sources(expr)
    actual = {con._profile for con in created_sources}
    expected = {con._profile for con in found_sources}
    assert actual == expected


def test_replace_computed_kwargs_expr(parquet_dir):
    deferred_linear_regression = deferred_fit_predict_sklearn(
        cls=LinearRegression, return_type=dt.float64
    )

    t = xo.deferred_read_parquet(parquet_dir / "diamonds.parquet", xo.connect())
    train_table, test_table = (
        el.tag(tag)
        for el, tag in zip(
            xo.train_test_splits(
                t, unique_key=tuple(t.columns), test_sizes=0.5, random_seed=42
            ),
            ("train", "test"),
        )
    )
    target = "price"
    features = tuple(c for c in t.select(s.numeric()).columns if c != target)
    predict_expr_udf = deferred_linear_regression(
        train_table, target, features
    ).deferred_other
    predicted = test_table.mutate(predict_expr_udf.on_expr(test_table))
    assert walk_nodes(Tag, predicted)
    removed = xo.expr.api._remove_tag_nodes(predicted)
    assert not walk_nodes(Tag, removed)
    assert not walk_nodes(Tag, predicted.ls.untagged)


def test_opaque_edges_name_real_fields() -> None:
    """Every ``OPAQUE_EDGES`` name must resolve on its op type.

    ``gen_children_of`` reads edge fields unguarded, so a stale name after a
    rename is an ``AttributeError`` at traversal time; catch it here instead.
    """
    for typ, edges in OPAQUE_EDGES.items():
        for name in edges:
            assert name in typ.__argnames__ or hasattr(typ, name), (
                f"{typ.__name__} has no field {name!r}"
            )


def test_opaque_ops_matches_opaque_edges() -> None:
    assert set(opaque_ops) == set(OPAQUE_EDGES)


def _op_classes(mod: types.ModuleType) -> tuple:
    return tuple(
        cls
        for _, cls in inspect.getmembers(mod, inspect.isclass)
        if issubclass(cls, ops.Node) and cls.__module__ == mod.__name__
    )


def test_expr_typed_fields_are_registered() -> None:
    """Static completeness sweep: every op class in ``relations``/``udf`` with
    an ``Expr``-annotated field must have an ``OpaqueSpec``, and each such
    field must be a descent edge or recorded in ``NON_EDGE_EXPR_FIELDS`` --
    "forgot to consider it" and "considered and excluded it" must not look the
    same.

    Complements the runtime tripwire
    (``test_traversal_raises_on_unregistered_expr_bearing_op``): this catches
    ``Expr``-annotated fields without constructing instances; the tripwire
    catches ops whose Expr payload hides behind ``Any`` annotations at first
    traversal. Known blind spot for both: an Expr living only in
    ``__config__`` (the ExprScalarUDF shape) never appears in ``__args__`` or
    annotations -- registering such ops is enforced by review, per ADR-0016.
    Annotations naming an Expr *subclass* (e.g. ``ir.Table``) are also not
    matched; the runtime tripwire backstops those instances.
    """
    expr_ann = re.compile(r"\bExpr\b")
    for cls in _op_classes(rel) + _op_classes(udf):
        expr_fields = tuple(
            name
            for name, ann in getattr(cls, "__annotations__", {}).items()
            if ann is Expr or (isinstance(ann, str) and expr_ann.search(ann))
        )
        if not expr_fields:
            continue
        spec = next(
            (sp for typ, sp in OPAQUE_SPECS.items() if issubclass(cls, typ)), None
        )
        assert spec is not None, (
            f"{cls.__name__} holds Expr-typed field(s) {expr_fields} "
            f"but has no OpaqueSpec"
        )
        allowed = (
            spec.read_edges
            + (spec.write_edges or ())
            + NON_EDGE_EXPR_FIELDS.get(cls, ())
        )
        for name in expr_fields:
            assert name in allowed, (
                f"{cls.__name__}.{name} is Expr-typed but neither a descent "
                f"edge nor recorded in NON_EDGE_EXPR_FIELDS"
            )


def test_traversal_raises_on_unregistered_expr_bearing_op() -> None:
    """Runtime tripwire: a spec-less op holding an ``Expr`` arg must raise on
    both the read path (``gen_children_of``) and the write path
    (``replace_nodes``) instead of silently not descending -- silent
    non-descent means wrong hashes/lineage with no failure anywhere.
    """

    class UnregisteredExprHolder(ops.Node):
        payload: Expr

    node = UnregisteredExprHolder(payload=xo.memtable({"a": [1]}).select("a"))
    with pytest.raises(ValueError, match="not registered in OPAQUE_SPECS"):
        tuple(gen_children_of(node))
    with pytest.raises(ValueError, match="not registered in OPAQUE_SPECS"):
        replace_nodes(lambda op, _kwargs: op, node)


def test_traversal_raises_on_unrecorded_expr_arg_of_registered_op() -> None:
    """Registered ops are tripwired too: an ``Expr``-valued arg that is
    neither a descent edge nor recorded in ``NON_EDGE_EXPR_FIELDS`` must raise
    on both traversal paths, so a new ``Any``-typed field holding an ``Expr``
    cannot grow on an existing opaque op unnoticed.
    """

    class SneakyFlightExpr(rel.FlightExpr):
        payload: Expr = None

    con = xo.connect()
    t = con.register(xo.memtable({"a": [1, 2, 3]}).to_pyarrow(), "t")
    node = SneakyFlightExpr(
        name="sneaky",
        schema=t.schema(),
        source=con,
        input_expr=t,
        unbound_expr=xo.table(t.schema(), name="u"),
        make_server=toolz.identity,
        make_connection=toolz.identity,
        payload=t,
    )
    with pytest.raises(ValueError, match="NON_EDGE_EXPR_FIELDS"):
        tuple(gen_children_of(node))
    with pytest.raises(ValueError, match="NON_EDGE_EXPR_FIELDS"):
        replace_nodes(lambda op, _kwargs: op, node)


def test_opaque_ops_mutually_non_subclassing() -> None:
    """``_opaque_lookup`` resolves to the first ``isinstance`` match, so its
    order-independence rests on no opaque op subclassing another. A new opaque
    op related by inheritance to an existing one needs an explicit ordering
    decision in ``OPAQUE_SPECS``, not an accidental one."""
    for a, b in itertools.permutations(opaque_ops, 2):
        assert not issubclass(a, b), f"{a.__name__} subclasses {b.__name__}"


def test_opaque_edges_is_read_only() -> None:
    """The edge tables are shared semantic constants; mutation must fail loudly."""
    with pytest.raises(TypeError):
        OPAQUE_EDGES[rel.Read] = ("nope",)


def test_gen_children_of_raises_on_stale_edge_name() -> None:
    """Edge fields are read unguarded: a stale name in an edge table must raise
    ``AttributeError`` at traversal time, not silently drop a child."""
    node = make_flight_expr().op()
    stale = {**OPAQUE_EDGES, rel.FlightExpr: ("input_exprs",)}
    with pytest.raises(AttributeError):
        tuple(gen_children_of(node, opaque_edges=stale))


def test_opaque_spec_rejects_multi_edge_rebind() -> None:
    """``rebind`` passes a single rewritten sub-expression, so a spec pairing it
    with more than one write-side edge must fail at construction time."""
    with pytest.raises(ValueError, match="rebind requires exactly one"):
        OpaqueSpec(("a", "b"), rebind=lambda op, sub_expr: op)
    with pytest.raises(ValueError, match="rebind requires exactly one"):
        OpaqueSpec(("a",), write_edges=(), rebind=lambda op, sub_expr: op)


def make_flight_expr() -> Expr:
    con = xo.connect()
    t = con.register(xo.memtable({"a": [1, 2, 3]}).to_pyarrow(), "t")
    return rel.FlightExpr(
        name="test_flight",
        schema=t.schema(),
        source=con,
        input_expr=t,
        unbound_expr=xo.table(t.schema(), name="u"),
        make_server=toolz.identity,
        make_connection=toolz.identity,
    ).to_expr()


def test_gen_children_flight_leaf_treats_flight_as_leaf() -> None:
    expr = make_flight_expr()
    node = expr.op()
    assert tuple(gen_children_of(node)) == (node.input_expr.op(),)
    assert gen_children_flight_leaf(node) == ()
    # the outer graph no longer reaches the input_expr's leaves
    assert walk_nodes(ops.DatabaseTable, node)
    assert walk_nodes(
        (ops.DatabaseTable,), node, gen_children=gen_children_flight_leaf
    ) == (node,)
