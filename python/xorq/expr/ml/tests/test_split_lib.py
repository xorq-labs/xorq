from __future__ import annotations

import itertools

import pandas as pd
import pytest

import xorq.api as xo
import xorq.expr.selectors as s
from xorq.api import memtable
from xorq.expr.ml import _calculate_bounds
from xorq.tests.util import assert_frame_equal


pytest.importorskip("duckdb")
pytest.importorskip("psycopg2")
pytest.importorskip("datafusion")


def test_train_test_splits_intersections():
    # This is testing the base case where a single float becomes ( 1-test_size , test_size ) proportion
    # Check counts and overlaps in train and test dataset
    N = 10000
    test_size = [0.1, 0.2, 0.7]

    # init table
    table = memtable([(i, "val") for i in range(N)], columns=["key1", "val"])
    results = [
        r
        for r in xo.train_test_splits(
            table,
            unique_key="key1",
            test_sizes=test_size,
            num_buckets=N,
            random_seed=42,
        )
    ]

    # make sure all splits mutually exclusive
    # These are all  a \ b  U  a intersect b  where b are the other splits
    element1 = results[0]
    complement1 = results[1].union(results[2])

    element2 = results[1]
    complement2 = results[0].union(results[2])

    element3 = results[2]
    complement3 = results[0].union(results[1])

    assert element1.union(complement1).join(table, how="anti").count().execute() == 0
    assert (
        element1.join(complement1, element1.key1 == complement1.key1).count().execute()
        == 0
    )

    assert element2.union(complement2).join(table, how="anti").count().execute() == 0
    assert (
        element2.join(complement2, element2.key1 == complement2.key1).count().execute()
        == 0
    )

    assert element3.union(complement3).join(table, how="anti").count().execute() == 0
    assert (
        element3.join(complement3, element3.key1 == complement3.key1).count().execute()
        == 0
    )


def test_train_test_split():
    # This is testing the base case where a single float becomes ( 1-test_size , test_size ) proportion
    # Check counts and overlaps in train and test dataset
    N = 100
    test_size = 0.25

    # init table
    table = memtable([(i, "val") for i in range(N)], columns=["key1", "val"])
    train_table, test_table = xo.train_test_splits(
        table, unique_key="key1", test_sizes=test_size, num_buckets=N, random_seed=42
    )

    # These values are for seed 42
    assert train_table.count().execute() == 75
    assert test_table.count().execute() == 25
    assert set(train_table.columns) == set(table.columns)
    assert set(test_table.columns) == set(table.columns)
    # make sure data unioned together is itself
    assert train_table.union(test_table).join(table, how="semi").count().execute() == N

    # Check reproducibility
    reproduced_train_table, reproduced_test_table = xo.train_test_splits(
        table, unique_key="key1", test_sizes=test_size, num_buckets=N, random_seed=42
    )
    assert_frame_equal(train_table.execute(), reproduced_train_table.execute())
    assert_frame_equal(test_table.execute(), reproduced_test_table.execute())

    # make sure it could generate different data with different random_seed
    different_train_table, different_test_table = xo.train_test_splits(
        table, unique_key="key1", test_sizes=test_size, num_buckets=N, random_seed=0
    )
    assert not train_table.execute().equals(different_train_table.execute())
    assert not test_table.execute().equals(different_test_table.execute())


@pytest.mark.parametrize(
    "con_name,unique_key",
    itertools.product(
        (None, "pandas", "let", "sqlite"),
        (
            "key1",
            ("key1",),
            ("key1", "key2"),
        ),
    ),
)
def test_train_test_split_parametrized(con_name, unique_key):
    N = 200
    test_size = 0.25
    tolerance = 0.01

    table = memtable(
        [(i, j, f"val-{i}-{j}") for i, j in ((i, i % N) for i in range(N**2))],
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
    assert abs(len(test) / (test_size * N**2) - 1) < tolerance


def test_train_test_split_invalid_test_size():
    table = memtable({"key": [1, 2, 3]})
    with pytest.raises(ValueError, match="test size should be a float between 0 and 1"):
        xo.train_test_splits(table, unique_key="key", test_sizes=1.5)
    with pytest.raises(ValueError, match="test size should be a float between 0 and 1"):
        xo.train_test_splits(table, unique_key="key", test_sizes=-0.5)


def test_train_test_split_invalid_num_buckets_type():
    table = memtable({"key": [1, 2, 3]})
    with pytest.raises(ValueError, match="num_buckets must be an integer"):
        xo.train_test_splits(table, unique_key="key", test_sizes=0.5, num_buckets=10.5)


def test_train_test_split_invalid_num_buckets_value():
    table = memtable({"key": [1, 2, 3]})
    with pytest.raises(
        ValueError, match="num_buckets = 1 places all data into training set"
    ):
        xo.train_test_splits(table, unique_key="key", test_sizes=0.5, num_buckets=1)


def test_train_test_split_multiple_keys():
    data = {
        "key1": range(100),
        "key2": [chr(i % 26 + 65) for i in range(100)],  # A, B, C, ...
        "value": [i % 3 for i in range(100)],
    }
    table = memtable(data)
    train_table, test_table = xo.train_test_splits(
        table,
        unique_key=["key1", "key2"],
        test_sizes=0.25,
        num_buckets=10,
        random_seed=99,
    )
    assert train_table.union(test_table).join(table, how="anti").count().execute() == 0


def test_train_test_splits_deterministic_with_seed():
    table = memtable({"key": range(100), "value": range(100)})
    test_sizes = [0.4, 0.6]

    splits1 = list(
        xo.train_test_splits(
            table,
            test_sizes=test_sizes,
            unique_key="key",
            random_seed=123,
            num_buckets=10,
        )
    )
    splits2 = list(
        xo.train_test_splits(
            table,
            test_sizes=test_sizes,
            unique_key="key",
            random_seed=123,
            num_buckets=10,
        )
    )

    for s1, s2 in zip(splits1, splits2):
        assert_frame_equal(s1.execute(), s2.execute())


def test_train_test_splits_invalid_test_sizes():
    table = memtable({"key": [1, 2, 3], "value": [4, 5, 6]})
    with pytest.raises(ValueError, match="Test size must be float."):
        next(xo.train_test_splits(table, "key", ["a", "b"]))
    with pytest.raises(
        ValueError, match="test size should be a float between 0 and 1."
    ):
        next(xo.train_test_splits(table, [-0.1, 0.5], "key"))


def test_train_test_splits_must_sum_one():
    table = memtable({"key": [1, 2, 3], "value": [4, 5, 6]})
    with pytest.raises(ValueError, match="Test sizes must sum to 1"):
        next(xo.train_test_splits(table, [0.1, 0.5], "key"))


def test_train_test_splits_with_all_selector():
    N = 50
    table = memtable({"k": range(N), "v": range(N)})
    splits = list(
        xo.train_test_splits(
            table, test_sizes=0.2, unique_key=s.all(), num_buckets=N, random_seed=0
        )
    )
    assert len(splits) == 2
    train_table, test_table = splits
    assert train_table.union(test_table).join(table, how="anti").count().execute() == 0

    splits2 = list(
        xo.train_test_splits(
            table, test_sizes=0.2, unique_key=s.all(), num_buckets=N, random_seed=0
        )
    )
    for s1, s2 in zip(splits, splits2):
        assert_frame_equal(s1.execute(), s2.execute())


@pytest.mark.parametrize(
    "test_sizes",
    ((1 / n,) * n for n in range(2, 100, 5)),
)
def test_approx_sum(test_sizes):
    _calculate_bounds(test_sizes)


def test_calculate_bounds():
    test_sizes = (0.2, 0.3, 0.5)
    expected_bounds = ((0.0, 0.2), (0.2, 0.5), (0.5, 1.0))
    assert _calculate_bounds(test_sizes) == expected_bounds


def test_train_test_splits_num_buckets_gt_one():
    table = memtable({"key": range(100), "value": range(100)})
    test_sizes = [0.4, 0.6]
    with pytest.raises(
        ValueError,
        match="num_buckets = 1 places all data into training set. For any integer x  >=0 , x mod 1 = 0 . ",
    ):
        next(
            xo.train_test_splits(
                table, "key", test_sizes, random_seed=123, num_buckets=1
            )
        )


@pytest.mark.parametrize(
    "connect_method",
    (
        lambda: xo.connect(),
        lambda: xo.duckdb.connect(),
        lambda: xo.postgres.connect_env(),
        pytest.param(
            xo.datafusion.connect,
            marks=pytest.mark.xfail(
                reason="Compilation rule for 'Hash' operation is not define"
            ),
        ),
    ),
)
def test_train_test_splits_intersections_parameterized_pass(connect_method):
    # This is testing the base case where a single float becomes ( 1-test_size , test_size ) proportion
    # Check counts and overlaps in train and test dataset
    N = 10000
    test_size = [0.1, 0.2, 0.7]

    # create test table for backend
    test_df = pd.DataFrame([(i, "val") for i in range(N)], columns=["key1", "val"])
    con = connect_method()
    test_table_name = f"{con.name}_test_df"
    con.create_table(test_table_name, test_df, temp=con.name == "postgres")

    table = con.table(test_table_name)

    results = [
        r
        for r in xo.train_test_splits(
            table,
            unique_key="key1",
            test_sizes=test_size,
            num_buckets=N,
            random_seed=42,
        )
    ]

    # make sure all splits mutually exclusive
    # These are all  a \ b  U  a intersect b  where b are the other splits
    element1 = results[0]
    complement1 = results[1].union(results[2])

    element2 = results[1]
    complement2 = results[0].union(results[2])

    element3 = results[2]
    complement3 = results[0].union(results[1])

    assert element1.union(complement1).join(table, how="anti").count().execute() == 0
    assert (
        element1.join(complement1, element1.key1 == complement1.key1).count().execute()
        == 0
    )

    assert element2.union(complement2).join(table, how="anti").count().execute() == 0
    assert (
        element2.join(complement2, element2.key1 == complement2.key1).count().execute()
        == 0
    )

    assert element3.union(complement3).join(table, how="anti").count().execute() == 0
    assert (
        element3.join(complement3, element3.key1 == complement3.key1).count().execute()
        == 0
    )
    con.drop_table(test_table_name)


@pytest.mark.parametrize(
    "connect_method",
    (
        lambda: xo.connect(),
        lambda: xo.duckdb.connect(),
        lambda: xo.postgres.connect_env(),
        pytest.param(
            lambda: xo.datafusion.connect(),
            marks=pytest.mark.xfail(
                reason="Compilation rule for 'Hash' operation is not define"
            ),
        ),
    ),
)
@pytest.mark.parametrize("n", (2, 8, 32))
@pytest.mark.parametrize("name", ("split", "other"))
def test_calc_split_column(connect_method, n, name):
    N = 10000
    test_sizes = (1 / n,) * n
    unique_key = "key1"

    # create test table for backend
    test_df = pd.DataFrame([(i, "val") for i in range(N)], columns=[unique_key, "val"])
    con = connect_method()
    test_table_name = f"{con.name}_test_df"
    con.create_table(test_table_name, test_df, temp=con.name == "postgres")

    table = con.table(test_table_name)
    expr = (
        table.mutate(
            xo.calc_split_column(
                table,
                unique_key=unique_key,
                test_sizes=test_sizes,
                random_seed=42,
                name=name,
            )
        )[name]
        .value_counts()
        .order_by(xo.asc(name))
    )
    df = xo.execute(expr)
    assert tuple(df[name].values) == tuple(range(n))
    assert df[f"{name}_count"].sum() == N
