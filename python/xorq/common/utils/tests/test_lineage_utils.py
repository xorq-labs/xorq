import pytest

import xorq as xo
import xorq.expr.datatypes as dt
from xorq.common.utils.graph_utils import to_node
from xorq.common.utils.lineage_utils import (
    ColorScheme,
    GenericNode,
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

    from rich.tree import Tree

    assert isinstance(display_tree, Tree)

    custom_palette = ColorScheme()
    display_tree_custom = build_tree(
        total_discount_tree, palette=custom_palette, dedup=False
    )
    assert isinstance(display_tree_custom, Tree)


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
