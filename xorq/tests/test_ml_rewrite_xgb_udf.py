from __future__ import annotations

import numpy as np
import pytest

import xorq as xo
from xorq import _


def test_predicate_collection(prediction_expr):
    """Verifies that a single filter predicate is correctly collected from an expression"""
    expr = prediction_expr.filter(_.carat < 1.0)
    predicates = xo.expr.ml.collect_predicates(expr.op())
    assert len(predicates) == 1
    assert predicates[0]["column"] == "carat"
    assert predicates[0]["op"] == "Less"
    assert predicates[0]["value"] == 1.0


def test_predicate_collection_on_udf_column_raises(prediction_expr):
    """Confirms that filter predicates on UDF columns are not collected"""
    expr = prediction_expr.filter(_.pred < 1.0)
    with pytest.raises(ValueError, match="Unsupported predicate on UDF column: pred"):
        xo.expr.ml.collect_predicates(expr.op())


def test_multiple_predicates(prediction_expr):
    """Checks that multiple filter predicates are correctly collected with their operators"""
    expr = prediction_expr.filter([_.carat < 1.0, _.depth > 60, _.table >= 55])
    predicates = xo.expr.ml.collect_predicates(expr.op())
    assert len(predicates) == 3
    assert {p["op"] for p in predicates} == {"Less", "Greater", "GreaterEqual"}


def test_model_pruning(prediction_expr):
    """Ensures pruned model predictions match original model predictions"""
    original = prediction_expr.filter(_.carat < 1.0).execute()
    optimized = xo.expr.ml.rewrite_quickgrove_expr(
        prediction_expr.filter(_.carat < 1.0)
    ).execute()
    np.testing.assert_array_almost_equal(
        original["pred"].values, optimized["pred"].values, decimal=3
    )


@pytest.mark.xfail
def test_pruned_model_exists_in_repr(prediction_expr):
    """Checks that pruned models are correctly marked in their string representation"""
    original = prediction_expr.filter(_.carat < 1.0)
    optimized = xo.expr.ml.rewrite_quickgrove_expr(
        prediction_expr.filter(_.carat < 1.0)
    )
    assert "_pruned" not in repr(original)
    assert "_pruned" in repr(optimized)


def test_mixed_pruned_model_types(mixed_prediction_expr):
    """Verifies that mixed model types maintain prediction accuracy after pruning"""
    original = mixed_prediction_expr.filter(_.carat < 1.0).execute()
    optimized = xo.expr.ml.rewrite_quickgrove_expr(
        mixed_prediction_expr.filter(_.carat < 1.0)
    ).execute()
    np.testing.assert_array_almost_equal(
        original["pred"].values, optimized["pred"].values, decimal=3
    )


def test_prune_if_predicates_are_on_udf(mixed_model_path, mixed_feature_table):
    """Verifies that models aren't pruned when filters are on UDF columns"""
    predict_udf = xo.expr.ml.make_quickgrove_udf(mixed_model_path)
    t = mixed_feature_table.mutate(pred=predict_udf.on_expr).filter(_.pred < 1)
    with pytest.raises(ValueError):
        xo.expr.ml.rewrite_quickgrove_expr(t)


def test_prune_if_predicates_are_not_features(mixed_model_path, mixed_feature_table):
    """Confirms that models aren't pruned when filters are on non-feature columns"""
    predict_udf = xo.expr.ml.make_quickgrove_udf(mixed_model_path)
    t = mixed_feature_table.mutate(pred=predict_udf.on_expr).filter(_.target >= 1)
    optimized = xo.expr.ml.rewrite_quickgrove_expr(t)
    assert "pruned" not in repr(optimized)


def test_mixed_prune_greather_than(mixed_prediction_expr):
    """Tests model pruning with greater than comparisons"""
    original = mixed_prediction_expr.filter(_.carat > 1.0).execute()
    optimized = xo.expr.ml.rewrite_quickgrove_expr(
        mixed_prediction_expr.filter(_.carat > 1.0)
    ).execute()
    np.testing.assert_array_almost_equal(
        original["pred"].values, optimized["pred"].values, decimal=3
    )


def test_mixed_prune_greather_than_or_equal(mixed_prediction_expr):
    """Tests model pruning with greater than or equal comparisons"""
    original = mixed_prediction_expr.filter(_.carat >= 1.0).execute()
    optimized = xo.expr.ml.rewrite_quickgrove_expr(
        mixed_prediction_expr.filter(_.carat >= 1.0)
    ).execute()
    np.testing.assert_array_almost_equal(
        original["pred"].values, optimized["pred"].values, decimal=3
    )
