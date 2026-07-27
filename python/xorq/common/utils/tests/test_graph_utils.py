import operator

import pytest
import toolz

import xorq.api as xo
import xorq.expr.datatypes as dt
import xorq.expr.relations as rel
import xorq.expr.selectors as s
import xorq.vendor.ibis.expr.operations as ops
from xorq.caching import SourceCache
from xorq.common.utils.graph_utils import (
    OPAQUE_EDGES,
    _gen_children_flight_leaf,
    find_all_sources,
    gen_children_of,
    opaque_ops,
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
    assert _gen_children_flight_leaf(node) == ()
    # the outer graph no longer reaches the input_expr's leaves
    assert walk_nodes(ops.DatabaseTable, node)
    assert walk_nodes(
        (ops.DatabaseTable,), node, gen_children=_gen_children_flight_leaf
    ) == (node,)
