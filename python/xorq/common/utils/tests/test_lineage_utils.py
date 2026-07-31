from __future__ import annotations

import json
import operator
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from attrs import evolve as attr_evolve

import xorq.api as xo
import xorq.expr.datatypes as dt
import xorq.vendor.ibis.expr.operations as ops
from xorq.caching import ParquetCache, SourceCache
from xorq.common.utils.graph_utils import gen_children_of, opaque_ops, to_node
from xorq.common.utils.lineage_utils import (
    ROOT_SCOPE,
    GenericNode,
    LineageDAG,
    TextTree,
    _boundary_kind,
    _build_column_tree,
    _mermaid_id,
    _mermaid_text,
    _redact,
    base_kind,
    build_column_trees,
    build_tree,
    compact_lineage_rows,
    compact_node_label,
    extract_lineage_dag,
    format_compact_lineage,
    format_mermaid_lineage,
    schema_diff,
)
from xorq.expr.udf import make_pandas_expr_udf
from xorq.flight.exchanger import UnboundExprExchanger
from xorq.ibis_yaml.compiler import build_expr, load_expr
from xorq.vendor.ibis.expr.operations.core import Node
from xorq.vendor.ibis.expr.operations.reductions import Sum
from xorq.vendor.ibis.expr.types.core import Expr, ExprMetadata


@xo.udf.make_pandas_udf(
    schema=xo.schema({"price": float, "discount": float}),
    return_type=dt.float,
    name="calculate_discount_value",
)
def calculate_discount_value(df):
    return df["price"] * df["discount"]


@pytest.fixture
def sample_expression():
    sales_table = xo.memtable(
        {
            "order_id": [1, 2, 1, 2],
            "price": [100.0, 150.0, 200.0, 250.0],
            "discount": [0.1, 0.2, 0.15, 0.1],
        },
        name="sales",
    )

    sales_with_discount = sales_table.mutate(
        discount_value=calculate_discount_value.on_expr(sales_table)
    )

    expr = sales_with_discount.group_by("order_id").agg(
        total_discount=xo._.discount_value.sum(),
        total_price=xo._.price.sum(),
    )

    return expr


def test__build_column_tree_basic(sample_expression):
    node = to_node(sample_expression)

    lineage_tree = _build_column_tree(node)

    assert isinstance(lineage_tree, GenericNode)
    assert lineage_tree.op is not None
    assert isinstance(lineage_tree.children, tuple)

    assert len(lineage_tree.children) > 0

    for child in lineage_tree.children:
        assert isinstance(child, GenericNode)


def test_build_column_trees_and_display(sample_expression):
    column_trees = build_column_trees(sample_expression)

    assert "total_discount" in column_trees
    assert "total_price" in column_trees

    total_discount_tree = column_trees["total_discount"]
    total_price_tree = column_trees["total_price"]

    assert isinstance(total_discount_tree, GenericNode)
    assert isinstance(total_price_tree, GenericNode)

    display_tree = build_tree(total_discount_tree, dedup=True, max_depth=5)

    assert isinstance(display_tree, TextTree)

    display_tree_no_dedup = build_tree(total_discount_tree, dedup=False)
    assert isinstance(display_tree_no_dedup, TextTree)


def test_complete_lineage_for_total_discount_column(sample_expression):
    column_trees = build_column_trees(sample_expression)
    total_discount_tree = column_trees["total_discount"]

    def gen_lineage_nodes(node):
        yield node.op
        for child in node.children:
            yield from gen_lineage_nodes(child)

    lineage_nodes = list(gen_lineage_nodes(total_discount_tree))

    # Expected lineage based on actual structure:
    # Sum -> Field(discount_value) -> UDF(calculate_discount_value) -> Field(price/discount) -> InMemoryTable

    node_types = [type(node).__name__ for node in lineage_nodes]

    sum_nodes = [node for node in lineage_nodes if isinstance(node, Sum)]
    assert len(sum_nodes) > 0, (
        f"Should contain Sum operation, found types: {node_types}"
    )

    udf_nodes = [
        node
        for node in lineage_nodes
        if type(node).__name__ == "calculate_discount_value"
    ]
    assert len(udf_nodes) > 0, (
        f"Should contain calculate_discount_value UDF in lineage, found types: {node_types}"
    )

    field_nodes = [node for node in lineage_nodes if isinstance(node, ops.Field)]
    field_names = [node.name for node in field_nodes]

    assert "discount_value" in field_names, (
        f"Should find discount_value field, found: {field_names}"
    )

    source_fields = {"price", "discount"}
    found_source_fields = source_fields.intersection(set(field_names))
    assert len(found_source_fields) > 0, (
        f"Should find at least one source field {source_fields}, found: {field_names}"
    )

    table_nodes = [
        node
        for node in lineage_nodes
        if isinstance(node, (ops.InMemoryTable, ops.UnboundTable, ops.DatabaseTable))
    ]
    assert len(table_nodes) > 0, "Should trace back to original table"


@pytest.fixture
def multi_join_expression():
    """Build a 4-table join expression that creates nested JoinChains.

    Simulates the real-world pattern where sub-expressions are loaded from the
    catalog (YAML round-trip) then joined via ``into_backend``.  The resulting
    expression has 3 nested JoinChains whose shared sub-graphs are traversed
    exponentially by ``_build_column_tree`` (which lacks memoization).
    """
    batting = xo.memtable(
        {
            "playerID": ["a", "b"],
            "yearID": [2007, 1975],
            "teamID": ["CHA", "MIL"],
            "G": [25, 137],
            "AB": [0, 465],
            "H": [0, 109],
        },
        name="batting",
    )
    people_mem = xo.memtable(
        {
            "playerID": ["a", "b"],
            "nameFirst": ["David", "Hank"],
            "nameLast": ["Aardsma", "Aaron"],
        },
        name="people",
    )
    salaries_mem = xo.memtable(
        {
            "playerID": ["a", "b"],
            "yearID": [2007, 1975],
            "teamID": ["CHA", "MIL"],
            "salary": [387500, 240000],
        },
        name="salaries",
    )
    fielding_mem = xo.memtable(
        {
            "playerID": ["a", "b"],
            "yearID": [2007, 1975],
            "teamID": ["CHA", "MIL"],
            "POS": ["P", "LF"],
        },
        name="fielding",
    )

    # YAML round-trip each table (simulates loading from the catalog)
    with tempfile.TemporaryDirectory() as td:
        p_dir = build_expr(people_mem, builds_dir=Path(td) / "p")
        s_dir = build_expr(salaries_mem, builds_dir=Path(td) / "s")
        f_dir = build_expr(fielding_mem, builds_dir=Path(td) / "f")
        people = load_expr(p_dir)
        salaries = load_expr(s_dir)
        fielding = load_expr(f_dir)

    backend = batting._find_backend()
    people_slim = people[["playerID", "nameFirst", "nameLast"]].into_backend(backend)
    salaries_slim = salaries[["playerID", "yearID", "teamID", "salary"]].into_backend(
        backend
    )
    fielding_slim = fielding[["playerID", "yearID", "teamID", "POS"]].into_backend(
        backend
    )

    with_names = (
        batting.filter(batting.AB > 100)
        .join(people_slim, predicates="playerID", how="left")
        .drop("playerID_right")
    )
    with_salary = with_names.join(
        salaries_slim,
        predicates=["playerID", "yearID", "teamID"],
        how="left",
    ).drop("playerID_right", "yearID_right", "teamID_right")
    return with_salary.join(
        fielding_slim,
        predicates=["playerID", "yearID", "teamID"],
        how="left",
    ).drop("playerID_right", "yearID_right", "teamID_right")


def _count_unique_dag_nodes(expr):
    """Count unique nodes reachable via gen_children_of."""
    visited = set()
    stack = [to_node(expr)]
    while stack:
        n = stack.pop()
        if id(n) not in visited:
            visited.add(id(n))
            stack.extend(gen_children_of(n))
    return len(visited)


def _count_unique_generic_nodes(root: GenericNode) -> int:
    seen, stack = set(), [root]
    while stack:
        n = stack.pop()
        if id(n) not in seen:
            seen.add(id(n))
            stack.extend(n.children)
    return len(seen)


def _build_column_tree_memoized(node: Node, _memo: dict | None = None) -> GenericNode:
    """Gold-standard recursive implementation with id-keyed memoization."""
    if _memo is None:
        _memo = {}
    key = id(node)
    if key in _memo:
        return _memo[key]

    match node:
        case ops.Field(rel=ops.Project(values=values)) as field_node:
            # include the field and recurse into its mapped expression
            mapped = values[field_node.name]
            child = _build_column_tree_memoized(to_node(mapped), _memo)
            result = GenericNode(op=field_node, children=(child,))

        case ops.Field() as field_node:
            children = tuple(
                _build_column_tree_memoized(to_node(child), _memo)
                for child in gen_children_of(field_node)
            )
            result = GenericNode(op=field_node, children=children)

        case ops.Project() as proj:
            result = _build_column_tree_memoized(to_node(proj.parent), _memo)

        case _:
            children = tuple(
                _build_column_tree_memoized(to_node(child), _memo)
                for child in gen_children_of(node)
            )
            result = GenericNode(op=node, children=children)

    _memo[key] = result
    return result


def _assert_matches_gold(expr):
    op = to_node(expr)
    cols = getattr(op, "values", None) or getattr(op, "fields", {})
    gold_memo = {}
    assert len(cols) > 0, (
        f"Expression has no columns to compare; got {type(to_node(expr)).__name__}"
    )
    for name, v in cols.items():
        node = to_node(v)
        assert _build_column_tree(node) == _build_column_tree_memoized(
            node, gold_memo
        ), f"Column '{name}': graph-based result differs from memoized gold standard"
    assert _build_column_tree(op) == _build_column_tree_memoized(op, gold_memo), (
        "graph-based result for the full expression root differs from memoized gold standard"
    )


def test_build_column_tree_matches_memoized_on_sample(sample_expression):
    _assert_matches_gold(sample_expression)


@pytest.mark.benchmark
def test_build_column_tree_matches_memoized_on_multi_join(multi_join_expression):
    _assert_matches_gold(multi_join_expression)


def test_build_column_tree_output_size_bounded(multi_join_expression):
    node = to_node(multi_join_expression)
    result = _build_column_tree(node)
    unique_input = _count_unique_dag_nodes(multi_join_expression)
    unique_output = _count_unique_generic_nodes(result)
    assert unique_output <= unique_input, (
        f"Output DAG has {unique_output} unique GenericNodes for {unique_input} "
        f"unique input nodes — suggests repeated construction"
    )


# ── TextTree._lines tests ─────────────────────────────────────────────


def test_text_tree_lines_leaf_root():
    tree = TextTree("root")
    assert tree._lines() == ("root",)


def test_text_tree_lines_root_single_child():
    tree = TextTree("root", children=(TextTree("child"),))
    assert tree._lines() == ("root", "└── child")


def test_text_tree_lines_root_two_children():
    tree = TextTree("root", children=(TextTree("a"), TextTree("b")))
    assert tree._lines() == ("root", "├── a", "└── b")


def test_text_tree_lines_nested_last_child_uses_space_prefix():
    # last child uses └──; its grandchildren get "    " (4 spaces) prefix
    grandchild = TextTree("gc")
    child = TextTree("c", children=(grandchild,))
    tree = TextTree("root", children=(child,))
    assert tree._lines() == ("root", "└── c", "    └── gc")


def test_text_tree_lines_nested_non_last_child_uses_pipe_prefix():
    # non-last child uses ├──; its grandchildren get "│   " prefix
    grandchild = TextTree("gc")
    child1 = TextTree("c1", children=(grandchild,))
    child2 = TextTree("c2")
    tree = TextTree("root", children=(child1, child2))
    assert tree._lines() == ("root", "├── c1", "│   └── gc", "└── c2")


def test_text_tree_str_joins_lines_with_newlines():
    tree = TextTree("root", children=(TextTree("a"), TextTree("b")))
    assert str(tree) == "root\n├── a\n└── b"


# ── extract_lineage_dag tests ──────────────────────────────────────────


def test_lineage_dag_basic_structure(sample_expression):
    """DAG must have nodes, edges, and root keys with correct types."""
    dag = extract_lineage_dag(sample_expression)

    assert isinstance(dag, LineageDAG)

    assert isinstance(dag.nodes, tuple)
    assert isinstance(dag.edges, tuple)
    assert isinstance(dag.root, str)

    assert len(dag.nodes) > 0
    assert len(dag.edges) > 0


def test_lineage_dag_node_fields(sample_expression):
    """Every node must have id, type, and label fields."""
    dag = extract_lineage_dag(sample_expression)

    for node in dag.nodes:
        assert "id" in node, f"Node missing 'id': {node}"
        assert "type" in node, f"Node missing 'type': {node}"
        assert "label" in node, f"Node missing 'label': {node}"
        assert isinstance(node["id"], str)
        assert isinstance(node["type"], str)
        assert isinstance(node["label"], str)


def test_lineage_dag_relation_schema(sample_expression):
    """Relation nodes (tables) should carry a schema dict."""
    dag = extract_lineage_dag(sample_expression)

    nodes_with_schema = [n for n in dag.nodes if "schema" in n]
    assert len(nodes_with_schema) > 0, "Expected at least one node with schema"

    for node in nodes_with_schema:
        assert isinstance(node["schema"], dict)
        for col_name, col_type in node["schema"].items():
            assert isinstance(col_name, str)
            assert isinstance(col_type, str)


def test_lineage_dag_edge_validity(sample_expression):
    """Every {from, to, scope} edge must reference existing node ids."""
    dag = extract_lineage_dag(sample_expression)

    node_ids = {n["id"] for n in dag.nodes}
    for edge in dag.edges:
        assert edge["from"] in node_ids, f"Edge source {edge['from']} not in node ids"
        assert edge["to"] in node_ids, f"Edge target {edge['to']} not in node ids"
        assert isinstance(edge["scope"], str)


def test_lineage_dag_root_validity(sample_expression):
    """Root must be a valid node id, and no edge should target the root."""
    dag = extract_lineage_dag(sample_expression)

    node_ids = {n["id"] for n in dag.nodes}
    assert dag.root in node_ids, "Root id not found in nodes"

    target_ids = {edge["to"] for edge in dag.edges}
    assert dag.root not in target_ids, "Root node should not be targeted by any edge"


def test_lineage_dag_multi_join(multi_join_expression):
    """Multi-join DAG should have multiple nodes and no duplicate node ids."""
    dag = extract_lineage_dag(multi_join_expression)

    assert len(dag.nodes) > 5, "Multi-join should produce many nodes"

    node_ids = [n["id"] for n in dag.nodes]
    assert len(node_ids) == len(set(node_ids)), "Node ids must be unique"

    # Should contain RemoteTable nodes (from into_backend)
    type_set = {n["type"] for n in dag.nodes}
    assert "RemoteTable" in type_set, (
        f"Multi-join should include RemoteTable nodes, found types: {type_set}"
    )


def test_lineage_dag_round_trip_via_expr_metadata(sample_expression):
    """Lineage DAG should survive ExprMetadata.to_dict() → from_dict() round-trip."""
    dag = extract_lineage_dag(sample_expression)
    metadata = ExprMetadata.from_expr(sample_expression)
    metadata = attr_evolve(metadata, lineage=dag)

    serialized = metadata.to_dict()
    assert "lineage" in serialized
    assert isinstance(serialized["lineage"], dict)
    assert serialized["lineage"]["version"] == 2
    # Serialized edges should be JSON-safe {from, to, scope} dicts
    assert isinstance(serialized["lineage"]["edges"], list)
    assert set(serialized["lineage"]["edges"][0]) == {"from", "to", "scope"}

    restored = ExprMetadata.from_dict(serialized)
    assert isinstance(restored.lineage, LineageDAG)
    assert restored.lineage.to_dict() == dag.to_dict()
    # Restored collections should be tuples again
    assert isinstance(restored.lineage.nodes, tuple)
    assert isinstance(restored.lineage.edges, tuple)


def test_lineage_dag_backward_compat_from_dict():
    """A legacy sidecar (integer ids, 2-tuple edges, no version) still loads."""
    json_lineage = {
        "nodes": [{"id": "0", "type": "Filter", "label": "Filter"}, {"id": "1"}],
        "edges": [["0", "1"]],
        "root": "0",
    }
    result = LineageDAG.from_dict(json_lineage)

    assert isinstance(result, LineageDAG)
    assert isinstance(result.nodes, tuple)
    assert isinstance(result.edges, tuple)
    assert result.version == 1
    assert result.overlays == {}
    # 2-tuples are normalised to root-scoped structured edges
    assert result.edges[0] == {"from": "0", "to": "1", "scope": ROOT_SCOPE}


def test_legacy_lineage_compacts_and_renders():
    """Legacy sidecars have no boundary annotation: they degrade to the root
    node alone rather than raising."""
    legacy = LineageDAG.from_dict(
        {
            "nodes": [
                {"id": "0", "type": "Filter", "label": "Filter"},
                {"id": "1", "type": "InMemoryTable", "label": "InMemoryTable"},
            ],
            "edges": [["0", "1"]],
            "root": "0",
        }
    )

    view = legacy.compact()
    assert view["root"] == "0"
    assert {n["id"] for n in view["nodes"]} == {"0"}
    assert view["edges"] == ()
    assert format_compact_lineage(legacy) == "Filter"
    assert legacy.boundaries() == ()


def test_lineage_dag_tag_metadata_preserves_structure():
    """Tag metadata with frozen mappings/sequences should round-trip as JSON,
    not be flattened into Python repr strings."""
    sales = xo.memtable({"x": [1]}, name="sales")
    tagged = sales.tag(
        tag="bsl",
        name="flights",
        dimensions=(
            ("origin", (("description", None), ("is_entity", False))),
            ("carrier", (("description", None), ("is_entity", False))),
        ),
        measures=(("flight_count", (("requires_unnest", ()),)),),
    )

    dag = extract_lineage_dag(tagged)
    [tag_node] = [n for n in dag.nodes if n["type"] == "Tag"]
    tm = tag_node["tag_metadata"]

    assert tm["tag"] == "bsl"
    assert tm["name"] == "flights"

    # Top-level dimensions should be a list of [name, attrs] pairs, not a
    # serialized "(('origin', (...)), ...)" string.
    assert isinstance(tm["dimensions"], list)
    assert [d[0] for d in tm["dimensions"]] == ["origin", "carrier"]
    assert isinstance(tm["dimensions"][0][1], list)
    assert tm["dimensions"][0][1][0] == ["description", None]

    assert isinstance(tm["measures"], list)
    assert [m[0] for m in tm["measures"]] == ["flight_count"]


# ── boundary taxonomy / compact view (XOR-363) ─────────────────────────


def _add_column(name, value=1):
    return operator.methodcaller("assign", **{name: value})


def _udxf(input_expr, con, name, extra_col="c"):
    """A FlightUDXF over *input_expr* that appends one column.

    Construction only -- no server is started, so this is safe in unit tests.
    """
    return xo.expr.relations.flight_udxf(
        input_expr,
        process_df=_add_column(extra_col),
        maybe_schema_in=input_expr.schema(),
        maybe_schema_out=xo.schema(input_expr.schema() | {extra_col: "int64"}),
        con=con,
        make_udxf_kwargs={"name": name},
    )


@pytest.fixture
def udxf_expression():
    """memtable -> into_backend -> filter -> FlightUDXF -> order_by."""
    con = xo.connect()
    t = xo.memtable({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]}, name="t")
    inner = t.into_backend(con, "inner").filter(xo._.a > 1)
    return _udxf(inner, con, "add_c").order_by("a")


@pytest.fixture
def flight_expr_expression():
    """A FlightExpr running a filter+aggregate remotely over a joined input."""
    con = xo.connect()
    left = xo.memtable({"a": [1, 2], "b": [1.0, 2.0]}, name="left")
    right = xo.memtable({"a": [1, 2], "c": ["x", "y"]}, name="right")
    joined = left.join(right.into_backend(con, "right_in"), "a")
    unbound = xo.table(joined.schema()).filter(xo._.a > 0)
    return xo.expr.relations.flight_expr(joined, unbound, con=con)


def test_boundary_kind_registered_for_every_opaque_op():
    """Every opaque (descent-boundary) op type must map to a taxonomy kind, else
    compact() would collapse a real boundary into a via run."""
    registered = tuple(t for t in _boundary_kind.registry if t is not object)
    for op_type in opaque_ops:
        assert any(issubclass(op_type, t) for t in registered), (
            f"{op_type.__name__} has no _boundary_kind registration"
        )


def test_boundaries_annotated_by_kind(udxf_expression):
    dag = extract_lineage_dag(udxf_expression)

    kinds = {n["boundary_kind"] for n in dag.boundaries()}
    assert {"table", "engine_crossing", "flight_udxf"} <= kinds

    for node in dag.nodes:
        assert node["is_boundary"] == (node.get("boundary_kind") is not None)
    for node in dag.boundaries():
        assert dag.capabilities(node), f"no capabilities for {node['boundary_kind']}"


def test_flight_udxf_node_carries_schema_transition(udxf_expression):
    dag = extract_lineage_dag(udxf_expression)

    [udxf] = dag.boundaries(kind="flight_udxf")
    assert udxf["process_boundary"] is True
    assert udxf["udxf_class"] == "add_c"
    assert udxf["udxf_command"]
    assert set(udxf["schema_in_observed"]) == {"a", "b"}
    assert set(udxf["schema_out"]) == {"a", "b", "c"}


def test_flight_input_expr_is_leaf_of_outer_walk(udxf_expression):
    """The outer walk must NOT flatten input_expr: its nodes belong to the
    Flight node's own scope only, or they would be double-counted."""
    dag = extract_lineage_dag(udxf_expression)
    [udxf] = dag.boundaries(kind="flight_udxf")

    root_ids = {e["from"] for e in dag.edges if e["scope"] == ROOT_SCOPE} | {
        e["to"] for e in dag.edges if e["scope"] == ROOT_SCOPE
    }
    nested = dag.scope(udxf["id"])
    nested_ids = {n["id"] for n in nested["nodes"]}

    assert nested_ids, "nested lineage should be populated"
    assert udxf["id"] in root_ids
    assert udxf["id"] not in nested_ids
    # the Flight node is a leaf of the outer graph
    assert not [
        e for e in dag.edges if e["scope"] == ROOT_SCOPE and e["from"] == udxf["id"]
    ]
    # input_expr's own nodes live only under the nested scope
    assert nested_ids.isdisjoint(root_ids)
    assert nested["root"] == udxf["nested_root"]
    assert nested["root"] in nested_ids


def test_nested_lineage_nodes_share_the_one_node_table(udxf_expression):
    """Storage model (a): nested nodes live in the shared table, deduped by
    snapshot_hash -- scope() is a view, not a sub-document."""
    dag = extract_lineage_dag(udxf_expression)
    [udxf] = dag.boundaries(kind="flight_udxf")

    ids = [n["id"] for n in dag.nodes]
    hashes = [n["snapshot_hash"] for n in dag.nodes]
    assert len(ids) == len(set(ids))
    assert len(hashes) == len(set(hashes))
    assert {n["id"] for n in dag.scope(udxf["id"])["nodes"]} <= set(ids)


def test_nested_of_nested_terminates():
    con = xo.connect()
    t = xo.memtable({"a": [1, 2, 3]}, name="t")
    inner = _udxf(t.into_backend(con, "inner"), con, "add_c", extra_col="c")
    outer = _udxf(inner, con, "add_d", extra_col="d")

    dag = extract_lineage_dag(outer)
    udxfs = dag.boundaries(kind="flight_udxf")

    assert len(udxfs) == 2
    scopes = {e["scope"] for e in dag.edges}
    assert scopes == {ROOT_SCOPE} | {u["id"] for u in udxfs}
    for u in udxfs:
        assert dag.scope(u["id"])["nodes"]


def test_compact_keeps_boundaries_and_collects_via(udxf_expression):
    dag = extract_lineage_dag(udxf_expression)
    view = dag.compact()

    kept = {n["id"] for n in view["nodes"]}
    assert view["root"] == dag.root
    assert view["root"] in kept
    assert all(n["is_boundary"] or n["id"] == dag.root for n in view["nodes"]), (
        "compact() must keep only boundaries (plus the root)"
    )

    # every compacted edge joins two kept nodes and records what it collapsed
    for edge in view["edges"]:
        assert edge["from"] in kept and edge["to"] in kept
        assert isinstance(edge["via"], list)
        assert all(isinstance(t, str) for t in edge["via"])

    # a boundary reachable only through non-boundary ops is still connected
    [udxf] = dag.boundaries(kind="flight_udxf")
    assert any(e["to"] == udxf["id"] for e in view["edges"])


def test_compact_of_flight_scope_is_rooted_in_the_nested_input(udxf_expression):
    dag = extract_lineage_dag(udxf_expression)
    [udxf] = dag.boundaries(kind="flight_udxf")

    nested = dag.compact(scope=udxf["id"])

    assert nested["scope"] == udxf["id"]
    assert nested["root"] == udxf["nested_root"]
    # the outer graph must not leak into the nested view
    outer_only = {n["id"] for n in dag.compact()["nodes"]}
    assert not ({n["id"] for n in nested["nodes"]} & outer_only)


def test_format_compact_lineage_renders_udxf_and_nested_tree(udxf_expression):
    dag = extract_lineage_dag(udxf_expression)
    text = format_compact_lineage(dag)

    assert "UDXF[add_c] : 2→3 cols" in text
    # the nested input lineage hangs off the Flight node, marked with ↳
    udxf_line = next(i for i, line in enumerate(text.splitlines()) if "UDXF[" in line)
    nested_lines = text.splitlines()[udxf_line + 1 :]
    assert nested_lines and any("↳" in line for line in nested_lines)
    assert "InMemoryTable" in text


def test_resolve_accepts_hash_label_kind_and_predicate(udxf_expression):
    dag = extract_lineage_dag(udxf_expression)
    [udxf] = dag.boundaries(kind="flight_udxf")

    assert dag.resolve(udxf["id"]) == (udxf,)
    assert dag.resolve(udxf["snapshot_hash"]) == (udxf,)
    assert dag.resolve("flight_udxf") == (udxf,)
    assert dag.resolve(lambda n: n["id"] == udxf["id"]) == (udxf,)
    assert dag.resolve("nope") == ()


def test_overlays_scaffold_is_empty_but_queryable(udxf_expression):
    dag = extract_lineage_dag(udxf_expression)
    [udxf] = dag.boundaries(kind="flight_udxf")

    assert dag.overlays == {}
    assert dag.available(udxf) == ()
    assert dag.expand(udxf["id"]) == {}

    with_overlay = attr_evolve(
        dag, overlays={"columns": {udxf["id"]: ["a"], "other": ["z"]}}
    )
    assert with_overlay.available(udxf) == ("columns",)
    assert with_overlay.expand(udxf["id"], "columns") == {
        "columns": {udxf["id"]: ["a"]}
    }


def test_lineage_round_trip_preserves_nested_scopes(udxf_expression):
    dag = extract_lineage_dag(udxf_expression)

    restored = LineageDAG.from_dict(dag.to_dict())

    assert restored.to_dict() == dag.to_dict()
    assert restored.version == 2
    [udxf] = restored.boundaries(kind="flight_udxf")
    assert restored.scope(udxf["id"])["nodes"]
    assert format_compact_lineage(restored) == format_compact_lineage(dag)


def test_lineage_is_stable_across_extractions(udxf_expression):
    first = extract_lineage_dag(udxf_expression)
    second = extract_lineage_dag(udxf_expression)
    assert first.to_dict() == second.to_dict()


def test_flight_expr_nested_lineage_over_multi_source_join(flight_expr_expression):
    """A Flight boundary whose input is a multi-source join: the outer view keeps
    the Flight node, the join lives in the nested scope."""
    dag = extract_lineage_dag(flight_expr_expression)

    [flight] = dag.boundaries(kind="flight_expr")
    assert flight["process_boundary"] is True
    assert set(flight["input_schema"]) == {"a", "b", "c"}
    assert set(flight["output_schema"]) == {"a", "b", "c"}
    assert flight["server_factory"]

    nested = dag.compact(scope=flight["id"])
    nested_kinds = {
        n["boundary_kind"] for n in nested["nodes"] if n.get("boundary_kind")
    }
    assert "join" in nested_kinds
    assert "engine_crossing" in nested_kinds
    assert {n["id"] for n in dag.compact()["nodes"]} == {dag.root, flight["id"]}

    text = format_compact_lineage(dag)
    assert "FlightExpr : 3→3 cols" in text
    assert "Join(" in text


def test_cache_and_pin_boundaries(tmp_path, parquet_dir):
    con = xo.connect()
    cache = ParquetCache.from_kwargs(source=con, relative_path=tmp_path)
    cached = (
        xo.deferred_read_parquet(parquet_dir / "awards_players.parquet", con=con)
        .filter(xo._.playerID == "bondto01")
        .cache(cache=cache)
    )

    dag = extract_lineage_dag(cached)
    assert dag.boundaries(kind="ingestion")
    [cache_node] = dag.boundaries(kind="cache")
    assert cache_node["cache_kind"] == "ParquetCache"
    assert "Cache[" in format_compact_lineage(dag)

    pinned_dag = extract_lineage_dag(cached.ls.pin(ensure_materialized=True))
    [pin_node] = pinned_dag.boundaries(kind="pin")

    assert pin_node["cache_key"]
    assert pin_node["cache_kind"] == "ParquetCache"
    assert not pinned_dag.boundaries(kind="cache")
    assert "Pin[" in format_compact_lineage(pinned_dag)


# ── per-kind boundary facts (XOR-363 ticket field table) ───────────────


def test_hashing_tag_kind_is_qualified_by_tag_value() -> None:
    """``tag:<value>`` per the taxonomy; ``base_kind`` strips the qualifier so
    registry lookups and kind queries keep working."""
    tagged = xo.memtable({"x": [1]}, name="sales").hashing_tag(tag="catalog-source")

    dag = extract_lineage_dag(tagged)
    [tag_node] = [n for n in dag.nodes if n["type"] == "HashingTag"]

    assert tag_node["boundary_kind"] == "tag:catalog-source"
    assert base_kind(tag_node["boundary_kind"]) == "tag"
    assert dag.capabilities(tag_node) == ("tag_metadata",)
    # both the qualified and the base form select it
    assert dag.boundaries(kind="tag:catalog-source") == (tag_node,)
    assert tag_node in dag.boundaries(kind="tag")
    assert dag.resolve("tag") == (tag_node,)
    assert "Tag[catalog-source]" in format_compact_lineage(dag)


def test_plain_tag_kind_is_unqualified() -> None:
    tagged = xo.memtable({"x": [1]}, name="sales").tag(tag="bsl")

    dag = extract_lineage_dag(tagged)
    [tag_node] = [n for n in dag.nodes if n["type"] == "Tag"]

    assert tag_node["boundary_kind"] == "tag"
    assert base_kind("tag") == "tag"
    assert base_kind(None) is None


def test_ingestion_carries_format_and_path(parquet_dir: Path) -> None:
    con = xo.connect()
    path = parquet_dir / "awards_players.parquet"
    expr = xo.deferred_read_parquet(path, con=con)

    dag = extract_lineage_dag(expr)
    [read] = dag.boundaries(kind="ingestion")

    assert read["format"] == "parquet"
    assert read["read_kind"] == "read_parquet"
    assert str(path) in str(read["path_or_uri"])
    assert read["backend"]


def test_read_kwargs_are_redacted() -> None:
    """The sidecar is committed next to the build: credentials must not travel."""
    assert _redact("password", "hunter2") == "***"
    assert _redact("aws_secret_key", "abc") == "***"
    assert _redact("API_KEY", "abc") == "***"
    assert _redact("uri", "postgres://user:pw@host:5432/db") == (
        "postgres://***@host:5432/db"
    )
    # non-credential values pass through untouched
    assert _redact("path", "/data/x.parquet") == "/data/x.parquet"
    assert _redact("uri", "s3://bucket/key.parquet") == "s3://bucket/key.parquet"
    assert _redact("n_rows", 10) == 10


def test_table_boundary_carries_namespace() -> None:
    """A namespaced backend (duckdb) fills database/catalog; a backend whose
    Namespace is empty (xorq) simply omits them."""
    con = xo.duckdb.connect()
    table = con.create_table("t_ns", xo.memtable({"a": [1]}, name="src").to_pyarrow())

    dag = extract_lineage_dag(table)
    [node] = dag.boundaries(kind="table")

    assert node["table_name"] == "t_ns"
    assert node["database"] == table.op().namespace.database
    assert node["catalog"] == table.op().namespace.catalog
    assert f"{node['backend']}:t_ns" in format_compact_lineage(dag)

    xorq_table = xo.connect().create_table(
        "t_no_ns", xo.memtable({"a": [1]}, name="src").to_pyarrow()
    )
    [plain] = extract_lineage_dag(xorq_table).boundaries(kind="table")
    assert plain["table_name"] == "t_no_ns"
    assert "database" not in plain and "catalog" not in plain


def test_udf_boundary_carries_name_and_signature() -> None:
    base = xo.memtable({"a": [1.0, 2.0]}, name="base")
    scaled = make_pandas_expr_udf(
        computed_kwargs_expr=base.a.sum(),
        fn=lambda computed, df: df.a * computed,
        schema=xo.schema({"a": "float64"}),
        return_type=dt.float64,
        name="scaled",
        post_process_fn=lambda x: x,
    )

    dag = extract_lineage_dag(base.mutate(out=scaled.on_expr(base)))
    [udf_node] = dag.boundaries(kind="udf")

    assert udf_node["udf_name"] == "scaled"
    assert udf_node["udf_signature"] == "inputs: [float64] -> output: float64"
    assert "UDF[scaled]" in format_compact_lineage(dag)


def test_join_boundary_summarises_predicates(multi_join_expression: Expr) -> None:
    dag = extract_lineage_dag(multi_join_expression)
    joins = dag.boundaries(kind="join")

    assert joins
    for join in joins:
        assert join["n_inputs"] >= 2
        summary = join["predicates_summary"]
        assert summary
        # column refs and operators only -- never a literal's value
        assert "==" in summary
        assert "Literal" not in summary


def test_engine_crossing_carries_both_ends() -> None:
    """The hop needs both ends: source is where the data lands, source_backend
    where it came from."""
    duck = xo.duckdb.connect()
    con = xo.connect()
    remote = duck.create_table("t_x", xo.memtable({"a": [1]}, name="src").to_pyarrow())

    dag = extract_lineage_dag(remote.into_backend(con, "crossed"))
    [crossing] = dag.boundaries(kind="engine_crossing")

    assert crossing["remote_name"] == "crossed"
    assert crossing["source_backend"] == duck.name
    assert crossing["backend"] == con.name
    assert f"RemoteTable[{duck.name} → {con.name}]" in format_compact_lineage(dag)


def test_engine_crossing_without_a_resolvable_origin_omits_it(
    multi_join_expression: Expr,
) -> None:
    """A crossing whose input bottoms out in a memtable has no origin backend to
    name; the field is omitted rather than guessed."""
    dag = extract_lineage_dag(multi_join_expression)
    crossings = dag.boundaries(kind="engine_crossing")

    assert crossings
    for crossing in crossings:
        assert crossing["remote_name"]
        assert crossing["backend"]
        assert "source_backend" not in crossing
    assert "RemoteTable[" not in format_compact_lineage(dag)


def test_engine_crossing_label_omits_a_same_backend_hop() -> None:
    """into_backend onto the expression's own backend has no hop to render."""
    con = xo.connect()
    t = con.create_table("t_same", xo.memtable({"a": [1]}, name="src").to_pyarrow())
    dag = extract_lineage_dag(t.into_backend(con, "same"))
    [crossing] = dag.boundaries(kind="engine_crossing")

    assert crossing["source_backend"] == crossing["backend"]
    text = format_compact_lineage(dag)
    assert "RemoteTable[" not in text
    assert f"→ {crossing['backend']}" in text


def test_flight_udxf_carries_required_schema_and_server_factory(
    udxf_expression: Expr,
) -> None:
    dag = extract_lineage_dag(udxf_expression)
    [node] = dag.boundaries(kind="flight_udxf")

    assert set(node["schema_in_required"]) == {"a", "b"}
    assert node["server_factory"]
    assert node["udxf_command"]
    assert "schema_in_required" in dag.capabilities(node)


def test_flight_udxf_stores_the_schema_diff(udxf_expression: Expr) -> None:
    """Open question #3 resolved in favour of the sidecar: the transition is
    stored, so external readers do not recompute it."""
    dag = extract_lineage_dag(udxf_expression)
    [node] = dag.boundaries(kind="flight_udxf")

    assert node["schema_diff"] == {
        "added": {"c": "int64"},
        "removed": [],
        "retyped": {},
        "renamed": {},
    }
    assert "(+c)" in format_compact_lineage(dag)


def test_schema_diff_reports_adds_drops_retypes_and_renames() -> None:
    assert schema_diff({"a": "int64"}, {"a": "int64", "b": "int64"}) == {
        "added": {"b": "int64"},
        "removed": [],
        "retyped": {},
        "renamed": {},
    }
    assert schema_diff({"a": "int64", "b": "int64"}, {"a": "int64"}) == {
        "added": {},
        "removed": ["b"],
        "retyped": {},
        "renamed": {},
    }
    assert schema_diff({"a": "int64"}, {"a": "float64"}) == {
        "added": {},
        "removed": [],
        "retyped": {"a": ["int64", "float64"]},
        "renamed": {},
    }
    # one drop + one add of the same dtype is the only rename we infer
    assert schema_diff({"a": "int64"}, {"b": "int64"}) == {
        "added": {},
        "removed": [],
        "retyped": {},
        "renamed": {"a": "b"},
    }
    # ambiguous (two drops, two adds) stays reported as adds plus drops
    ambiguous = schema_diff({"a": "int64", "b": "int64"}, {"c": "int64", "d": "int64"})
    assert ambiguous["renamed"] == {}
    assert set(ambiguous["added"]) == {"c", "d"}
    assert ambiguous["removed"] == ["a", "b"]
    # no delta, or nothing to compare against
    assert schema_diff({"a": "int64"}, {"a": "int64"}) is None
    assert schema_diff(None, {"a": "int64"}) is None
    assert schema_diff({"a": "int64"}, None) is None


def test_flight_expr_carries_the_wire_command(flight_expr_expression: Expr) -> None:
    """``flight_command`` must be the command the client actually exchanges on:
    the ``UnboundExprExchanger``'s, not a locally invented string."""
    dag = extract_lineage_dag(flight_expr_expression)
    [flight] = dag.boundaries(kind="flight_expr")

    node = to_node(flight_expr_expression)
    (flight_op,) = [
        n for n in (node, *gen_children_of(node)) if type(n).__name__ == "FlightExpr"
    ] or [None]
    expected = UnboundExprExchanger(flight_op.unbound_expr).command

    assert flight["flight_command"] == expected
    assert flight["flight_command"].startswith("execute-unbound-expr-")
    assert "flight_command" in dag.capabilities(flight)


def test_cache_boundary_carries_portable_path_and_key_prefix(
    tmp_path: Path, parquet_dir: Path
) -> None:
    """``cache_path`` is the storage's *relative* path: an absolute
    ``base_path``-resolved path would leak the developer's home dir into a
    committed sidecar and differ per machine."""
    con = xo.connect()
    cache = ParquetCache.from_kwargs(source=con, relative_path="xorq-cache")
    cached = (
        xo.deferred_read_parquet(parquet_dir / "awards_players.parquet", con=con)
        .filter(xo._.playerID == "bondto01")
        .cache(cache=cache)
    )

    dag = extract_lineage_dag(cached)
    [cache_node] = dag.boundaries(kind="cache")

    assert cache_node["cache_path"] == "xorq-cache"
    assert cache_node["cache_key_prefix"] == cache.strategy.key_prefix
    assert "cache_path" in dag.capabilities(cache_node)
    # the resolved cache dir (base_path / relative_path) must not travel
    assert str(cache.storage.path) not in str(cache_node)
    assert str(Path.home()) not in str(cache_node)


def test_source_cache_has_no_path() -> None:
    """A SourceCache lives in a backend, not on a path: its location is already
    the ``backend`` field."""
    con = xo.connect()
    cache = SourceCache.from_kwargs(source=con)
    cached = xo.memtable({"a": [1, 2]}, name="t_src").into_backend(con).cache(cache)

    dag = extract_lineage_dag(cached)
    [cache_node] = dag.boundaries(kind="cache")

    assert "cache_path" not in cache_node
    assert cache_node["cache_kind"] == "SourceCache"
    assert cache_node["cache_key_prefix"] == cache.strategy.key_prefix
    assert cache_node["backend"]


def test_new_boundary_fields_round_trip(
    udxf_expression: Expr, flight_expr_expression: Expr
) -> None:
    for expr, kind, fields in (
        (
            udxf_expression,
            "flight_udxf",
            ("schema_in_required", "schema_diff", "server_factory"),
        ),
        (flight_expr_expression, "flight_expr", ("flight_command", "server_factory")),
    ):
        dag = extract_lineage_dag(expr)
        restored = LineageDAG.from_dict(dag.to_dict())
        [before] = dag.boundaries(kind=kind)
        [after] = restored.boundaries(kind=kind)
        for name in fields:
            assert after[name] == before[name]
            assert after[name] is not None


def test_labels_carry_column_counts_and_a_backend_tag() -> None:
    """`<what> (<n> cols) [<backend>]`, per the ticket's target render. Kinds that
    already name the backend (`duckdb:orders`, `Read[..]`, `RemoteTable[a → b]`)
    do not repeat it; Flight kinds show their transition instead of one width."""
    duck = xo.duckdb.connect()
    con = xo.connect()
    customers = duck.create_table(
        "raw_customers",
        xo.memtable({"id": [1], "name": ["a"]}, name="c_src").to_pyarrow(),
    )
    orders = con.create_table(
        "raw_orders",
        xo.memtable({"id": [1], "amount": [1.0]}, name="o_src").to_pyarrow(),
    )
    enriched = _udxf(orders, con, "OrderEnricher", extra_col="score")
    expr = customers.into_backend(con, "customers_in").join(enriched, "id")

    text = format_compact_lineage(extract_lineage_dag(expr))

    # 4 cols: the join keeps both `id` columns plus name and amount
    assert f"Join(2 inputs) (4 cols) [{con.name}]" in text
    assert f"UDXF[OrderEnricher] : 2→3 cols [{con.name}]   (+score)" in text
    assert f"RemoteTable[{duck.name} → {con.name}] (2 cols)" in text
    assert f"{duck.name}:raw_customers (2 cols)" in text
    # the backend is named once per label, never twice
    for line in text.splitlines():
        assert line.count(con.name) <= 1 or "RemoteTable[" in line


def test_join_backend_is_its_left_input_not_the_remote_source() -> None:
    """Walking to the leaves would reach the *remote* side of an into_backend and
    make a single-backend join look multi-backend."""
    duck = xo.duckdb.connect()
    con = xo.connect()
    left = con.create_table(
        "left_t", xo.memtable({"id": [1]}, name="l_src").to_pyarrow()
    )
    right = duck.create_table(
        "right_t", xo.memtable({"id": [1]}, name="r_src").to_pyarrow()
    )
    expr = left.join(right.into_backend(con, "right_in"), "id")

    dag = extract_lineage_dag(expr)
    [join] = dag.boundaries(kind="join")

    assert join["backend"] == con.name
    assert {c["source_backend"] for c in dag.boundaries(kind="engine_crossing")} == {
        duck.name
    }


def test_cache_and_pin_labels_carry_no_column_count_when_schema_is_absent() -> None:
    """A node with no schema recorded (legacy sidecar) renders bare."""
    legacy = LineageDAG.from_dict(
        {
            "nodes": [
                {
                    "id": "0",
                    "type": "CachedNode",
                    "label": "CachedNode",
                    "is_boundary": True,
                    "boundary_kind": "cache",
                    "cache_kind": "ParquetCache",
                }
            ],
            "edges": [],
            "root": "0",
        }
    )

    assert format_compact_lineage(legacy) == "Cache[ParquetCache]"


def test_scope_of_a_single_node_nested_lineage_keeps_its_root() -> None:
    """A Flight whose input is a bare table has a nested lineage of one node and
    zero edges: deriving the scope's ids from edges alone dropped that node while
    `root` still pointed at it."""
    con = xo.connect()
    table = con.create_table(
        "t_bare", xo.memtable({"a": [1, 2]}, name="src").to_pyarrow()
    )

    dag = extract_lineage_dag(_udxf(table, con, "add_c"))
    [udxf] = dag.boundaries(kind="flight_udxf")
    nested = dag.scope(udxf["id"])

    assert nested["edges"] == ()
    assert nested["root"] == udxf["nested_root"]
    assert [n["id"] for n in nested["nodes"]] == [nested["root"]]
    # and the derived compact view agrees with the raw scope view
    assert {n["id"] for n in dag.compact(scope=udxf["id"])["nodes"]} == {nested["root"]}


def test_compact_lineage_rows_split_glyphs_label_kind_and_via(
    udxf_expression: Expr,
) -> None:
    """The rows carry each piece separately so a renderer can style them; their
    concatenation is exactly the plain-text tree."""
    dag = extract_lineage_dag(udxf_expression)
    rows = compact_lineage_rows(dag)

    assert "\n".join(row.text for row in rows) == format_compact_lineage(dag)
    assert rows[0].prefix == "" and rows[0].via == ()
    assert all(set(row.prefix) <= set("│├└─ ") for row in rows)
    assert all("via [" not in row.label for row in rows)

    [udxf_row] = [r for r in rows if r.kind == "flight_udxf"]
    assert udxf_row.label.startswith("UDXF[add_c]")
    # every kind on a row is a base kind, ready for a style lookup; a
    # non-boundary root (Sort here) carries None
    assert all(row.kind == base_kind(row.kind) for row in rows)
    assert rows[0].kind is None


def test_compact_lineage_rows_carry_via_off_the_label(
    multi_join_expression: Expr,
) -> None:
    """A collapsed run is a row field, not text baked into the label, so a
    renderer can dim it separately."""
    rows = compact_lineage_rows(extract_lineage_dag(multi_join_expression))

    via_rows = [r for r in rows if r.via]
    assert via_rows
    for row in via_rows:
        assert all(isinstance(t, str) for t in row.via)
        assert row.via_suffix.startswith("   via [")
        assert row.text == row.prefix + row.label + row.via_suffix


def test_compact_lineage_rooted_at_a_node_is_the_subtree_feeding_it(
    udxf_expression: Expr,
) -> None:
    """`root=` renders one node's upstream instead of the whole expression, and
    reaches a node that lives only inside a Flight's nested scope."""
    dag = extract_lineage_dag(udxf_expression)
    [udxf] = dag.boundaries(kind="flight_udxf")

    subtree = format_compact_lineage(dag, root=udxf["id"])

    assert subtree.splitlines()[0].startswith("UDXF[add_c]")
    assert "↳ " in subtree
    # the outer graph above the UDXF is excluded
    full = format_compact_lineage(dag)
    assert subtree != full
    assert len(subtree.splitlines()) < len(full.splitlines())

    # a node that lives only in the nested scope roots its own subtree, rendered
    # from that scope's view rather than the root graph's
    nested_root = udxf["nested_root"]
    nested = format_compact_lineage(dag, root=nested_root)
    assert nested.splitlines()[0] == compact_node_label(dag.by_id[nested_root])
    assert "InMemoryTable" in nested
    assert all(line.lstrip("│└├─ ↳") in subtree for line in nested.splitlines())

    assert compact_lineage_rows(dag, root=udxf["id"])[0].kind == "flight_udxf"


def test_compact_lineage_rooted_at_an_unknown_node_is_empty(
    udxf_expression: Expr,
) -> None:
    assert format_compact_lineage(
        extract_lineage_dag(udxf_expression), root="@nope"
    ) == ("(empty)")


def test_mermaid_lineage_is_a_flowchart_of_the_compact_graph(
    udxf_expression: Expr,
) -> None:
    """Same views as the text tree, emitted as mermaid: downstream edges, a
    subgraph per Flight scope, one classDef per boundary kind."""
    dag = extract_lineage_dag(udxf_expression)
    [udxf] = dag.boundaries(kind="flight_udxf")

    diagram = format_mermaid_lineage(dag)
    lines = diagram.splitlines()

    assert lines[0] == "flowchart TD"
    # ids are mermaid-safe: no '@'
    assert "@" not in diagram
    assert f'{_mermaid_id(udxf["id"])}["UDXF[add_c]' in diagram
    # the nested input lineage is a subgraph, per its stored scope
    stripped = [line.strip() for line in lines]
    assert (
        f'subgraph {_mermaid_id(udxf["id"])}_scope["input of UDXF[add_c]"]' in stripped
    )
    assert "end" in stripped
    # kinds carry a class, and every classed node was declared
    assert any(line.startswith("  classDef flight_udxf ") for line in lines)
    for line in lines:
        if line.startswith("  class "):
            for member in line.split()[1].split(","):
                assert f"{member}[" in diagram


def test_mermaid_lineage_edges_point_downstream(udxf_expression: Expr) -> None:
    """The stored edges are depends-on; the diagram reverses them so `TD` reads
    sources-to-output."""
    dag = extract_lineage_dag(udxf_expression)
    diagram = format_mermaid_lineage(dag)

    root = _mermaid_id(dag.root)
    arrows = [line.strip() for line in diagram.splitlines() if "-->" in line]
    assert arrows
    # nothing flows out of the final expression; something flows into it
    assert not any(line.startswith(f"{root} -->") for line in arrows)
    assert any(line.endswith(f"> {root}") for line in arrows)
    # a collapsed run keeps its `via` as the edge label
    if any(e["via"] for e in dag.compact()["edges"]):
        assert any("|via " in line for line in arrows)


_STABILITY_SCRIPT = """
import hashlib, json, sys
from xorq.common.utils.lineage_utils import LineageDAG, format_mermaid_lineage

dag = LineageDAG.from_dict(json.load(sys.stdin))
for kwargs in ({}, {"expand": True}, {"expand_from": "@cache"}):
    print(hashlib.md5(format_mermaid_lineage(dag, **kwargs).encode()).hexdigest())
"""


def _stability_dag() -> dict:
    """A graph with a run of non-boundary ops between two boundaries, so the
    compact, expanded and expand_from renders all differ."""
    nodes = [
        {
            "id": "@join",
            "type": "JoinChain",
            "is_boundary": True,
            "boundary_kind": "join",
            "n_inputs": 2,
        },
        {
            "id": "@cache",
            "type": "CachedNode",
            "is_boundary": True,
            "boundary_kind": "cache",
            "cache_kind": "ParquetCache",
        },
        {
            "id": "@table",
            "type": "DatabaseTable",
            "is_boundary": True,
            "boundary_kind": "table",
            "table_name": "t",
        },
    ] + [
        {"id": f"@field_{i}", "type": "Field", "label": f"Field:c{i}"} for i in range(8)
    ]
    edges = [
        {"from": "@join", "to": f"@field_{i}", "scope": ROOT_SCOPE} for i in range(8)
    ]
    edges += [
        {"from": f"@field_{i}", "to": "@cache", "scope": ROOT_SCOPE} for i in range(8)
    ]
    edges += [{"from": "@cache", "to": "@table", "scope": ROOT_SCOPE}]
    return {"version": 2, "root": "@join", "nodes": nodes, "edges": edges}


@pytest.mark.parametrize(
    "index",
    (
        pytest.param(0, id="compact"),
        pytest.param(1, id="expand"),
        pytest.param(2, id="expand_from"),
    ),
)
def test_mermaid_lineage_is_byte_stable_across_processes(
    index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A diagram pasted into docs must not churn between runs. Set iteration order
    is only stable *within* a process, so this has to fork rather than render
    twice in-process."""
    payload = json.dumps(_stability_dag())
    digests = set()
    for seed in (1, 2, 3, 4):
        monkeypatch.setenv("PYTHONHASHSEED", str(seed))
        result = subprocess.run(
            [sys.executable, "-c", _STABILITY_SCRIPT],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
        )
        digests.add(result.stdout.splitlines()[index])
    assert len(digests) == 1, digests


def test_mermaid_lineage_is_byte_stable(udxf_expression: Expr) -> None:
    """Node order comes from a BFS, not set iteration: a diagram pasted into docs
    must not churn between processes."""
    dag = extract_lineage_dag(udxf_expression)
    assert format_mermaid_lineage(dag) == format_mermaid_lineage(
        LineageDAG.from_dict(dag.to_dict())
    )


def test_mermaid_lineage_rooted_at_a_node_and_of_nothing(
    udxf_expression: Expr,
) -> None:
    dag = extract_lineage_dag(udxf_expression)
    [udxf] = dag.boundaries(kind="flight_udxf")

    subtree = format_mermaid_lineage(dag, root=udxf["id"])

    assert subtree.splitlines()[0] == "flowchart TD"
    assert _mermaid_id(udxf["id"]) in subtree
    assert _mermaid_id(dag.root) not in subtree
    assert "(empty)" in format_mermaid_lineage(dag, root="@nope")


def test_mermaid_expand_draws_the_stored_graph(udxf_expression: Expr) -> None:
    """`expand=True` renders `dag.scope()` -- the stored graph -- instead of the
    compact one, so the runs folded into `via` become real nodes."""
    dag = extract_lineage_dag(udxf_expression)

    compact = format_mermaid_lineage(dag)
    expanded = format_mermaid_lineage(dag, expand=True)

    def declared(diagram: str) -> set[str]:
        return {
            line.strip().split("[", 1)[0]
            for line in diagram.splitlines()
            if "[" in line and not line.strip().startswith(("subgraph", "class"))
        }

    assert declared(compact) < declared(expanded)
    assert len(declared(expanded)) == len(dag.nodes)
    # nothing is collapsed any more, so no edge carries a `via` label
    assert "|via " not in expanded
    assert "classDef unknown " in expanded
    # nested Flight scopes are expanded too, not just the outer graph
    [udxf] = dag.boundaries(kind="flight_udxf")
    nested_ids = {n["id"] for n in dag.scope(udxf["id"])["nodes"]}
    assert len(nested_ids) > 1
    for nid in nested_ids:
        assert _mermaid_id(nid) in expanded


def test_mermaid_expand_from_switches_detail_part_way_down(
    multi_join_expression: Expr,
) -> None:
    """`expand_from` keeps the compact graph but draws the stored one from that
    node downward -- the walk carries its own mode, so detail changes mid-graph."""
    dag = extract_lineage_dag(multi_join_expression)
    [crossing, *_] = dag.boundaries(kind="engine_crossing")

    compact = format_mermaid_lineage(dag)
    mixed = format_mermaid_lineage(dag, expand_from=crossing["id"])
    everything = format_mermaid_lineage(dag, expand=True)

    def declared(diagram: str) -> set[str]:
        """Op nodes only: column nodes are drawn for the expanded node alone, so
        they are not part of the compact-vs-expanded comparison."""
        return {
            line.strip().split("[", 1)[0]
            for line in diagram.splitlines()
            if "[" in line
            and "_col_" not in line
            and not line.strip().startswith(("subgraph", "class"))
        }

    # strictly between the two extremes
    assert declared(compact) < declared(mixed) < declared(everything)
    # the root stays in the picture, unlike root=
    assert _mermaid_id(dag.root) in mixed
    # a node is declared once even when several paths reach it
    for nid in declared(mixed):
        assert mixed.count(f"{nid}[") == 1


def test_mermaid_expand_from_expands_the_run_above_the_node(
    multi_join_expression: Expr,
) -> None:
    """`compact()` folds runs onto the edge *above* a node as much as below it, so
    rendering one node raw has to reach upward too -- otherwise expanding a node
    whose own subtree has no intermediates draws nothing new."""
    dag = extract_lineage_dag(multi_join_expression)
    [crossing, *_] = dag.boundaries(kind="engine_crossing")
    (incoming,) = [
        e for e in dag.compact()["edges"] if e["to"] == crossing["id"] and e["via"]
    ]

    diagram = format_mermaid_lineage(dag, expand_from=crossing["id"])

    # the ops that were folded into that edge's `via` are now nodes of their own
    for op_type in incoming["via"]:
        assert op_type in diagram
    # and the edge itself is gone, replaced by the stored run
    assert f"{_mermaid_id(crossing['id'])} -->|via" not in diagram, (
        "the collapsed edge should be replaced, not drawn alongside"
    )


def test_mermaid_expand_from_accepts_several_nodes(multi_join_expression: Expr) -> None:
    dag = extract_lineage_dag(multi_join_expression)
    crossings = tuple(n["id"] for n in dag.boundaries(kind="engine_crossing"))
    assert len(crossings) > 1

    one = format_mermaid_lineage(dag, expand_from=crossings[:1])
    several = format_mermaid_lineage(dag, expand_from=crossings)

    assert len(several.splitlines()) > len(one.splitlines())


def test_mermaid_label_escaping() -> None:
    """A label reaching mermaid must not break out of its quoted node."""
    assert _mermaid_text('a "b" <c> & d') == "a &quot;b&quot; &lt;c&gt; &amp; d"


def test_compact_lineage_rows_of_an_empty_dag() -> None:
    assert compact_lineage_rows(LineageDAG(nodes=(), edges=(), root="")) == ()


def test_compact_of_unknown_scope_is_empty() -> None:
    """A scope with no recorded nested_root degrades to an empty view."""
    dag = LineageDAG.from_dict(
        {"nodes": [{"id": "0", "type": "Filter"}], "edges": [], "root": "0"}
    )
    view = dag.compact(scope="@no_such_flight")
    assert view == {"nodes": (), "edges": (), "root": None, "scope": "@no_such_flight"}
