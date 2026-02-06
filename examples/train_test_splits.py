"""Train/test splitting with both scalar and list-based partition sizes, demonstrating
deterministic, expression-level data splitting.

Traditional approach: You would use sklearn's train_test_split on in-memory DataFrames
or numpy arrays. Splits are random by default (seeded via random_state), and creating
multiple named partitions (holdout, test, validation, training) requires multiple calls
or manual index slicing.

With xorq: Splitting operates at the expression level via train_test_splits, working on
any backend without materializing data first. Multiple partition sizes can be specified
as a list, producing mutually exclusive partitions in one call. Splits are deterministic
via unique_key hashing, ensuring reproducibility without relying on global random state.
"""

import pandas as pd

import xorq.api as xo
from xorq.api import memtable


name = "split"
N = 100000
# if single float deferred partitions of train and test will be returned
# With proportions (1-test_size, test_size)
test_size = 0.25
# If test sizes is a list of floats , mutually exclusive partitions will be returned
partition_info = {
    "hold_out": 0.1,
    "test": 0.2,
    "validation": 0.3,
    "training": 0.4,
}
split_kwargs = {
    "unique_key": "key1",
    "test_sizes": list(partition_info.values()),
    "num_buckets": N,
    "random_seed": 42,
}


def demo_scalar_test_sizes(train_table, test_table):
    train_count = xo.execute(train_table.count())
    test_count = xo.execute(test_table.count())
    total = train_count + test_count
    print(f"train ratio: {round(train_count / total, 2)}")
    print(f"test ratio: {round(test_count / total, 2)}\n")


def demo_partitions(partitions):
    counts = pd.Series(xo.execute(p.count()) for p in partitions)
    total = sum(counts)
    for i, partition_name in enumerate(partition_info.keys()):
        print(f"{partition_name.upper()} Ratio: {round(counts[i] / total, 2)}")
    return counts


def demo_split_column(split_column):
    counts = xo.execute(
        split_column.value_counts().order_by(split_column.get_name())
    ).set_index(name)[f"{name}_count"]
    return counts


table = memtable([(i, "val") for i in range(N)], columns=["key1", "val"])
train_table, test_table = xo.train_test_splits(
    table,
    **(split_kwargs | {"test_sizes": test_size}),
)
partitions = tuple(
    xo.train_test_splits(
        table,
        **split_kwargs,
    )
)
split_column = xo.calc_split_column(
    table,
    name=name,
    **split_kwargs,
)


if __name__ in ("__pytest_main__", "__main__"):
    demo_scalar_test_sizes(train_table, test_table)
    partition_counts = demo_partitions(partitions)
    other_counts = demo_split_column(split_column)
    assert partition_counts.equals(other_counts)
    pytest_examples_passed = True
