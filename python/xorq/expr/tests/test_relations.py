from datetime import datetime, timedelta

import pandas as pd
import pyarrow as pa
import pytest

import xorq.api as xo
import xorq.expr.relations as relations
import xorq.vendor.ibis as ibis
import xorq.vendor.ibis.expr.operations as ops
from xorq.expr.relations import (
    CachedNode,
    FlightUDXF,
    HashingTag,
    Tag,
    count_remote_table_readers,
    flight_serve,
)
from xorq.ibis_yaml.enums import ExprKind
from xorq.vendor.ibis.expr.types.core import ExprMetadata


@pytest.mark.parametrize(
    "schema_val,pattern",
    [
        (ibis.schema({"x": "float64"}), "Schema validation failed, expected"),
        (None, "^Schema validation failed$"),
    ],
)
def test_flight_udxf_validate_schema_fail(schema_val, pattern):
    schema = ibis.schema({"x": "int64"})
    input_expr = ibis.table(schema, name="t")

    class DummyUDXF:
        schema_in_required = schema_val

        @staticmethod
        def schema_in_condition(sch):
            return False

        @staticmethod
        def calc_schema_out(sch):
            return None

    with pytest.raises(ValueError, match=pattern):
        FlightUDXF.validate_schema(input_expr, DummyUDXF)


# -- ExprKind.Source detection ------------------------------------------------


def test_kind_memtable_is_source():
    expr = xo.memtable({"a": [1, 2, 3]})
    assert ExprMetadata.from_expr(expr).kind == ExprKind.Source


def test_kind_database_table_is_source():
    con = xo.connect()
    con.raw_sql("CREATE TABLE _test_src (a INT, b VARCHAR)")
    expr = con.table("_test_src")
    assert ExprMetadata.from_expr(expr).kind == ExprKind.Source


def test_kind_cached_node_is_source():
    con = xo.connect()
    con.raw_sql("CREATE TABLE _test_cached (a INT)")
    table_expr = con.table("_test_cached")
    dt = table_expr.op()
    cached = CachedNode(
        name="_test_cached",
        schema=dt.schema,
        source=dt.source,
        parent=table_expr,
    )
    assert ExprMetadata.from_expr(cached.to_expr()).kind == ExprKind.Source


def test_kind_tagged_source_is_source():
    schema = ibis.schema({"a": "int64"})
    inner = xo.memtable({"a": [1, 2, 3]}).op()
    tagged = Tag(schema=schema, parent=inner)
    assert ExprMetadata.from_expr(tagged.to_expr()).kind == ExprKind.Source


def test_kind_hashing_tagged_source_is_source():
    schema = ibis.schema({"a": "int64"})
    inner = xo.memtable({"a": [1, 2, 3]}).op()
    tagged = HashingTag(schema=schema, parent=inner)
    assert ExprMetadata.from_expr(tagged.to_expr()).kind == ExprKind.Source


def test_kind_filtered_table_is_expr():
    expr = xo.memtable({"a": [1, 2, 3]}).filter(xo._.a > 1)
    assert ExprMetadata.from_expr(expr).kind == ExprKind.Expr


def test_kind_projected_table_is_expr():
    expr = xo.memtable({"a": [1, 2], "b": [3, 4]}).select("a")
    assert ExprMetadata.from_expr(expr).kind == ExprKind.Expr


def test_kind_aggregated_table_is_expr():
    t = xo.memtable({"a": [1, 2, 3]})
    expr = t.aggregate(total=t.a.sum())
    assert ExprMetadata.from_expr(expr).kind == ExprKind.Expr


def test_kind_unbound_table_is_unbound_expr():
    t = xo.table(schema={"a": "int64"})
    assert ExprMetadata.from_expr(t).kind == ExprKind.UnboundExpr


def test_kind_filtered_unbound_is_unbound_expr():
    t = xo.table(schema={"a": "int64"})
    expr = t.filter(t.a > 0)
    assert ExprMetadata.from_expr(expr).kind == ExprKind.UnboundExpr


def test_kind_source_to_dict():
    expr = xo.memtable({"a": [1, 2, 3]})
    d = ExprMetadata.from_expr(expr).to_dict()
    assert d["kind"] == "source"
    assert "schema_out" in d
    assert "schema_in" not in d


# -- .ls.kind accessor --------------------------------------------------------


def test_ls_kind_source():
    expr = xo.memtable({"a": [1, 2, 3]})
    assert expr.ls.kind == ExprKind.Source


def test_ls_kind_expr():
    expr = xo.memtable({"a": [1, 2, 3]}).filter(xo._.a > 1)
    assert expr.ls.kind == ExprKind.Expr


def test_ls_kind_unbound_expr():
    t = xo.table(schema={"a": "int64"})
    assert t.ls.kind == ExprKind.UnboundExpr


def test_ls_kind_matches_metadata():
    exprs = [
        xo.memtable({"a": [1]}),
        xo.memtable({"a": [1]}).filter(xo._.a > 0),
        xo.table(schema={"a": "int64"}),
    ]
    for expr in exprs:
        assert expr.ls.kind == ExprMetadata.from_expr(expr).kind


# -- .ls.unwrapped accessor ---------------------------------------------------


def test_unwrapped_bare_table():
    expr = xo.memtable({"a": [1, 2, 3]})
    assert isinstance(expr.ls.unwrapped, ops.InMemoryTable)


def test_unwrapped_strips_tag():
    schema = ibis.schema({"a": "int64"})
    inner = xo.memtable({"a": [1, 2, 3]}).op()
    tagged = Tag(schema=schema, parent=inner)
    assert isinstance(tagged.to_expr().ls.unwrapped, ops.InMemoryTable)


def test_unwrapped_strips_hashing_tag():
    schema = ibis.schema({"a": "int64"})
    inner = xo.memtable({"a": [1, 2, 3]}).op()
    tagged = HashingTag(schema=schema, parent=inner)
    assert isinstance(tagged.to_expr().ls.unwrapped, ops.InMemoryTable)


def test_unwrapped_preserves_cached_node():
    con = xo.connect()
    con.raw_sql("CREATE TABLE _test_unwrapped (a INT)")
    table_expr = con.table("_test_unwrapped")
    dt = table_expr.op()
    cached = CachedNode(
        name="_test_unwrapped",
        schema=dt.schema,
        source=dt.source,
        parent=table_expr,
    )
    assert isinstance(cached.to_expr().ls.unwrapped, CachedNode)


def test_unwrapped_filtered_table():
    expr = xo.memtable({"a": [1, 2, 3]}).filter(xo._.a > 1)
    assert not isinstance(expr.ls.unwrapped, ops.InMemoryTable)


# -- flight_serve -------------------------------------------------------------


def test_flight_serve_calls_serve_unbound_with_unbound_expr(monkeypatch):
    captured = {}

    def fake_serve_unbound(unbound_expr, make_server=None, **kwargs):
        captured["unbound_expr"] = unbound_expr
        captured["make_server"] = make_server
        captured["kwargs"] = kwargs
        return ("server", "exchanger")

    monkeypatch.setattr(relations, "flight_serve_unbound", fake_serve_unbound)

    con = xo.connect()
    con.create_table("_fs_test", {"a": [1, 2, 3]})
    expr = con.table("_fs_test")

    flight_serve(expr)

    assert ExprMetadata.from_expr(captured["unbound_expr"]).kind == ExprKind.UnboundExpr


def test_flight_serve_passes_make_server_through(monkeypatch):
    captured = {}

    def fake_serve_unbound(unbound_expr, make_server=None, **kwargs):
        captured["make_server"] = make_server
        return ("server", "exchanger")

    monkeypatch.setattr(relations, "flight_serve_unbound", fake_serve_unbound)

    sentinel = object()
    flight_serve(xo.memtable({"a": [1]}), make_server=sentinel)
    assert captured["make_server"] is sentinel


def test_flight_serve_passes_kwargs_through(monkeypatch):
    captured = {}

    def fake_serve_unbound(unbound_expr, make_server=None, **kwargs):
        captured["kwargs"] = kwargs
        return ("server", "exchanger")

    monkeypatch.setattr(relations, "flight_serve_unbound", fake_serve_unbound)

    flight_serve(xo.memtable({"a": [1]}), port=9999, host="localhost")
    assert captured["kwargs"] == {"port": 9999, "host": "localhost"}


def test_flight_serve_returns_serve_unbound_result(monkeypatch):
    expected = ("my_server", "my_exchanger")

    monkeypatch.setattr(relations, "flight_serve_unbound", lambda *a, **kw: expected)

    result = flight_serve(xo.memtable({"a": [1]}))
    assert result is expected


# -- count_remote_table_readers ----------------------------------------------


@pytest.fixture
def remote_table():
    src = xo.connect()
    t = src.register(pa.table({"a": [1, 2, 3], "k": [1, 1, 2]}), "tt")
    return t.into_backend(xo.connect(), "rt")


def _only_count(expr):
    counts = count_remote_table_readers(expr)
    assert len(counts) == 1
    return next(iter(counts.values()))


def test_count_bare_remote_table_floored_at_one(remote_table):
    # one scan, but must never be 0 (max_readers=0 forbids the reader created)
    assert _only_count(remote_table) == 1


def test_count_single_scan(remote_table):
    expr = remote_table.filter(remote_table.a > 1)
    assert _only_count(expr) == 1


def test_count_many_fields_one_scan(remote_table):
    # many column refs over one table resolve within a single scan
    expr = remote_table.select(s=remote_table.a + remote_table.k)
    assert _only_count(expr) == 1


def test_count_self_join_is_two(remote_table):
    expr = remote_table.join(remote_table.view(), "k")
    assert _only_count(expr) == 2


def test_count_asof_tolerance_double_scan():
    # the #983 case: tolerance lowering scans the left input twice, right once.
    # graph-level counting sees one ref each; only the compiled SQL reveals it.
    ddb = xo.duckdb.connect()
    sdf = pd.DataFrame(
        {"site": ["a", "b"], "ts": [datetime(2024, 1, 1), datetime(2024, 1, 2)]}
    )
    edf = pd.DataFrame(
        {
            "site": ["a", "b"],
            "ev": ["x", "y"],
            "ts": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        }
    )
    left = xo.memtable(sdf).into_backend(ddb)
    right = xo.memtable(edf).into_backend(ddb)
    expr = left.asof_join(
        right, on="ts", predicates="site", tolerance=timedelta(seconds=1)
    ).drop("ts_right")

    counts = count_remote_table_readers(expr)
    assert sorted(counts.values()) == [1, 2]


def test_count_unbindable_returns_empty(monkeypatch):
    # SQL failure -> empty mapping -> caller builds unbounded cache (safe)
    src = xo.connect()
    t = src.register(pa.table({"a": [1, 2, 3]}), "tt")
    expr = t.into_backend(xo.connect(), "rt")

    def boom(*a, **k):
        raise RuntimeError("no sql")

    monkeypatch.setattr(relations.ibis, "to_sql", boom)
    assert count_remote_table_readers(expr) == {}
