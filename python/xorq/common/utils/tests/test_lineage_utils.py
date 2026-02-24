import tempfile
from pathlib import Path

import pytest

import xorq.api as xo
import xorq.common.utils.lineage_utils as lu
import xorq.expr.datatypes as dt
from xorq.common.utils.graph_utils import gen_children_of, to_node
from xorq.common.utils.lineage_utils import (
    GenericNode,
    TextTree,
    _build_column_tree,
    build_column_trees,
    build_tree,
)


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
    import xorq.vendor.ibis.expr.operations as ops
    from xorq.vendor.ibis.expr.operations.reductions import Sum

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
    from xorq.ibis_yaml.compiler import build_expr, load_expr

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


def _count_build_column_tree_calls(expr, cap=200_000):
    """Count _build_column_tree invocations, aborting at *cap*."""
    orig = lu._build_column_tree
    calls = [0]

    def counted(node, *args, **kwargs):
        calls[0] += 1
        if calls[0] >= cap:
            raise RuntimeError(f"call cap ({cap:,}) reached")
        return orig(node, *args, **kwargs)

    lu._build_column_tree = counted
    try:
        build_column_trees(expr)
    except RuntimeError:
        pass
    finally:
        lu._build_column_tree = orig
    return calls[0]


def test_build_column_trees_scales_linearly_on_multi_join(multi_join_expression):
    """build_column_trees should complete in O(nodes) calls, not O(N^k).

    Without memoization, 3 nested JoinChains cause 100K+ recursive calls for
    ~100 unique DAG nodes because every shared sub-graph is re-traversed from
    scratch.  The memoization cache in _build_column_tree (keyed by node
    identity) keeps the call count to a small multiple of the unique node count.

    See https://github.com/xorq-labs/xorq/issues/1618
    """
    unique_nodes = _count_unique_dag_nodes(multi_join_expression)
    call_count = _count_build_column_tree_calls(multi_join_expression)

    # A well-memoized traversal should need no more than ~10x the unique
    # nodes.  The current implementation exceeds 1000x.
    max_reasonable_calls = unique_nodes * 10
    assert call_count <= max_reasonable_calls, (
        f"build_column_trees made {call_count:,} calls for {unique_nodes} unique "
        f"DAG nodes (ratio {call_count / unique_nodes:.0f}x). "
        f"Expected at most {max_reasonable_calls:,} calls with memoization."
    )
