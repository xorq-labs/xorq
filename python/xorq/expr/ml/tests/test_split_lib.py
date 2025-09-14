from __future__ import annotations

import itertools

import pandas as pd
import pytest

import xorq.api as xo
from xorq.api import memtable


@pytest.mark.parametrize(
    "con_name,unique_key",
    itertools.product(
        (None, "pandas", "let"),
        (
            "key1",
            ("key1",),
            ("key1", "key2"),
        ),
    ),
)
def test_train_test_split(con_name, unique_key):
    N = 100
    test_size = 0.25

    table = memtable(
        [(i, j, f"val-{i}-{j}") for i in range(N) for j in range(N)],
        columns=["key1", "key2", "val"],
    )
    if con_name is not None:
        table = table.into_backend(getattr(xo, con_name).connect())
    train_table, test_table = xo.train_test_splits(
        table,
        unique_key=unique_key,
        test_sizes=test_size,
        num_buckets=N,
        random_seed=42,
    )

    # FIXME: use single expr rather than two executions: assert (train_table.union(test_table).join(table, how="semi").count().execute() == N**2)
    # splits are mutually exclusive and joinly exhaustive
    (train, test) = (expr.execute() for expr in (train_table, test_table))
    assert train.merge(test).empty
    assert len(pd.concat((train, test)).drop_duplicates()) == N**2

    # target test size is roughly attained
    assert abs(len(test) / (test_size * N**2) - 1) < 0.01
