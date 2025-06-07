import xorq as xo
import xorq.expr.datatypes as dt
from xorq.common.utils.lineage_utils import build_column_trees, print_tree


@xo.udf.make_pandas_udf(
    schema=xo.schema({"price": float, "discount": float}),
    return_type=dt.float,
    name="calculate_discount_value",
)
def calculate_discount_value(df):
    return df["price"] * df["discount"]


con = xo.connect()

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


def main():
    column_trees = build_column_trees(expr)

    for column, tree in column_trees.items():
        print(f"Lineage for column '{column}':")
        print_tree(tree)
        print("\n")


if __name__ == "__pytest_main__":
    main()
    pytest_examples_passed = True
