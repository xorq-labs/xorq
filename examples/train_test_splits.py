import pandas as pd

import xorq as xo
from xorq import memtable


N = 100000
# if single float deferred partitions of train and test will be returned
# With proportions (1-test_size, test_size)
test_size = 0.25
# init table
table = memtable([(i, "val") for i in range(N)], columns=["key1", "val"])


train_table, test_table = xo.train_test_splits(
    table, unique_key="key1", test_sizes=test_size, num_buckets=N, random_seed=42
)

train_count = xo.execute(train_table.count())
test_count = xo.execute(test_table.count())
total = train_count + test_count
print(f"train ratio: {round(train_count / total, 2)}")
print(f"test ratio: {round(test_count / total, 2)}\n")


# If test sizes is a list of floats , mutually exclusive partitions will be returned
partition_info = {
    "hold_out": 0.1,
    "test": 0.2,
    "validation": 0.3,
    "training": 0.4,
}

partitions = tuple(
    xo.train_test_splits(
        table,
        unique_key="key1",
        test_sizes=list(partition_info.values()),
        num_buckets=N,
        random_seed=42,
    )
)
counts = pd.Series(xo.execute(p.count()) for p in partitions)
total = sum(counts)

for i, partition_name in enumerate(partition_info.keys()):
    print(f"{partition_name.upper()} Ratio: {round(counts[i] / total, 2)}")

name = "split"
c = xo.calc_split_column(
    table,
    unique_key="key1",
    test_sizes=list(partition_info.values()),
    num_buckets=N,
    random_seed=42,
    name=name,
)
other_counts = xo.execute(c.value_counts().order_by(c.get_name())).set_index(name)[
    f"{name}_count"
]
assert counts.equals(other_counts)
pytest_examples_passed = True
