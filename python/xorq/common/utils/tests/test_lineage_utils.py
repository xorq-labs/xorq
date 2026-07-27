from __future__ import annotations

import operator
import tempfile
from pathlib import Path

import pytest
from attrs import evolve as attr_evolve

import xorq.api as xo
import xorq.expr.datatypes as dt
import xorq.vendor.ibis.expr.operations as ops
from xorq.caching import ParquetCache
from xorq.common.utils.graph_utils import gen_children_of, opaque_ops, to_node
from xorq.common.utils.lineage_utils import (
    ROOT_SCOPE,
    GenericNode,
    LineageDAG,
    TextTree,
    _boundary_kind,
    _build_column_tree,
    build_column_trees,
    build_tree,
    extract_lineage_dag,
    format_compact_lineage,
)
from xorq.ibis_yaml.compiler import build_expr, load_expr
from xorq.vendor.ibis.expr.operations.core import Node
from xorq.vendor.ibis.expr.operations.reductions import Sum
from xorq.vendor.ibis.expr.types.core import ExprMetadata


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


def test_compact_of_unknown_scope_is_empty():
    """A scope with no recorded nested_root degrades to an empty view."""
    dag = LineageDAG.from_dict(
        {"nodes": [{"id": "0", "type": "Filter"}], "edges": [], "root": "0"}
    )
    view = dag.compact(scope="@no_such_flight")
    assert view == {"nodes": (), "edges": (), "root": None, "scope": "@no_such_flight"}
