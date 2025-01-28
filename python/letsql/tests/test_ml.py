from __future__ import annotations

import ibis.expr.datatypes as dt
import numpy as np
import pandas as pd
import pytest
from ibis import memtable

import letsql as ls
from letsql.expr.ml import _calculate_bounds
from letsql.tests.util import assert_frame_equal


def test_train_test_splits_intersections():
    # This is testing the base case where a single float becomes ( 1-test_size , test_size ) proportion
    # Check counts and overlaps in train and test dataset
    N = 10000
    test_size = [0.1, 0.2, 0.7]

    # init table
    table = memtable([(i, "val") for i in range(N)], columns=["key1", "val"])
    results = [
        r
        for r in ls.train_test_splits(
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
    train_table, test_table = ls.train_test_splits(
        table, unique_key="key1", test_sizes=test_size, num_buckets=N, random_seed=42
    )
    # These values are for seed 42
    assert train_table.count().execute() == 73
    assert test_table.count().execute() == 27
    assert set(train_table.columns) == set(table.columns)
    assert set(test_table.columns) == set(table.columns)
    # make sure data unioned together is itself
    assert train_table.union(test_table).join(table, how="semi").count().execute() == N

    # Check reproducibility
    reproduced_train_table, reproduced_test_table = ls.train_test_splits(
        table, unique_key="key1", test_sizes=test_size, num_buckets=N, random_seed=42
    )
    assert_frame_equal(train_table.execute(), reproduced_train_table.execute())
    assert_frame_equal(test_table.execute(), reproduced_test_table.execute())

    # make sure it could generate different data with different random_seed
    different_train_table, different_test_table = ls.train_test_splits(
        table, unique_key="key1", test_sizes=test_size, num_buckets=N, random_seed=0
    )
    assert not train_table.execute().equals(different_train_table.execute())
    assert not test_table.execute().equals(different_test_table.execute())


def test_train_test_split_invalid_test_size():
    table = memtable({"key": [1, 2, 3]})
    with pytest.raises(ValueError, match="test size should be a float between 0 and 1"):
        ls.train_test_splits(table, unique_key="key", test_sizes=1.5)
    with pytest.raises(ValueError, match="test size should be a float between 0 and 1"):
        ls.train_test_splits(table, unique_key="key", test_sizes=-0.5)


def test_train_test_split_invalid_num_buckets_type():
    table = memtable({"key": [1, 2, 3]})
    with pytest.raises(ValueError, match="num_buckets must be an integer"):
        ls.train_test_splits(table, unique_key="key", test_sizes=0.5, num_buckets=10.5)


def test_train_test_split_invalid_num_buckets_value():
    table = memtable({"key": [1, 2, 3]})
    with pytest.raises(
        ValueError, match="num_buckets = 1 places all data into training set"
    ):
        ls.train_test_splits(table, unique_key="key", test_sizes=0.5, num_buckets=1)


def test_train_test_split_multiple_keys():
    data = {
        "key1": range(100),
        "key2": [chr(i % 26 + 65) for i in range(100)],  # A, B, C, ...
        "value": [i % 3 for i in range(100)],
    }
    table = memtable(data)
    train_table, test_table = ls.train_test_splits(
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
        ls.train_test_splits(table, "key", test_sizes, random_seed=123, num_buckets=10)
    )
    splits2 = list(
        ls.train_test_splits(table, "key", test_sizes, random_seed=123, num_buckets=10)
    )

    result1_all = splits1[0].union(splits1[1]).execute()
    result2_all = splits2[0].union(splits2[1]).execute()
    assert result1_all.equals(result2_all)


def test_train_test_splits_invalid_test_sizes():
    table = memtable({"key": [1, 2, 3], "value": [4, 5, 6]})
    with pytest.raises(ValueError, match="Test size must be float."):
        next(ls.train_test_splits(table, "key", ["a", "b"]))
    with pytest.raises(
        ValueError, match="test size should be a float between 0 and 1."
    ):
        next(ls.train_test_splits(table, "key", [-0.1, 0.5]))


def test_train_test_splits_must_sum_one():
    table = memtable({"key": [1, 2, 3], "value": [4, 5, 6]})
    with pytest.raises(ValueError, match="Test sizes must sum to 1"):
        next(ls.train_test_splits(table, "key", [0.1, 0.5]))


def test_calculate_bounds():
    test_sizes = [0.2, 0.3, 0.5]
    expected_bounds = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
    assert _calculate_bounds(test_sizes) == expected_bounds


def test_train_test_splits_num_buckets_gt_one():
    table = memtable({"key": range(100), "value": range(100)})
    test_sizes = [0.4, 0.6]
    with pytest.raises(
        ValueError,
        match="num_buckets = 1 places all data into training set. For any integer x  >=0 , x mod 1 = 0 . ",
    ):
        next(
            ls.train_test_splits(
                table, "key", test_sizes, random_seed=123, num_buckets=1
            )
        )


@pytest.mark.parametrize(
    "connect_method",
    (
        ls.connect,
        ls.duckdb.connect,
        ls.postgres.connect_env,
        pytest.param(
            ls.datafusion.connect,
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
    con.create_table(test_table_name, test_df)

    table = con.table(test_table_name)

    results = [
        r
        for r in ls.train_test_splits(
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


def test_make_quickgrove_udf_predictions(feature_table, float_model_path):
    """quickgrove UDF predictions should match expected values"""
    predict_udf = ls.expr.ml.make_quickgrove_udf(float_model_path)
    result = feature_table.mutate(pred=predict_udf.on_expr).execute()

    np.testing.assert_almost_equal(
        result["pred"].values, result["expected_pred"].values, decimal=3
    )


def test_make_quickgrove_udf_signature(float_model_path):
    """quickgrove UDF should have correct signature with float64 inputs, float32 output"""
    predict_fn = ls.expr.ml.make_quickgrove_udf(float_model_path)

    assert predict_fn.__signature__.return_annotation == dt.float32
    assert all(
        p.annotation == dt.float64 for p in predict_fn.__signature__.parameters.values()
    )
    assert predict_fn.__name__ == "pretrained_model"


def test_make_quickgrove_udf_mixed_features(mixed_model_path):
    """quickgrove UDF should support int64 and boolean feature types"""
    predict_fn = ls.expr.ml.make_quickgrove_udf(mixed_model_path)
    assert "i" in predict_fn.model.feature_types


def test_make_quickgrove_udf__repr(mixed_model_path):
    """quickgrove UDF repr should include model metadata"""
    predict_fn = ls.expr.ml.make_quickgrove_udf(mixed_model_path)
    repr_str = repr(predict_fn)

    expected_info = [
        "Total number of trees:",
        "Average tree depth:",
        "Max tree depth:",
        "Total number of nodes:",
        "Model path:",
        "Signature:",
    ]

    for info in expected_info:
        assert info in repr_str
