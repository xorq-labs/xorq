from __future__ import annotations

import numpy as np
import pytest

import xorq.expr.datatypes as dt
from xorq.ml import (
    make_quickgrove_udf,
)


def test_make_quickgrove_udf_predictions(feature_table, float_model_path):
    """quickgrove UDF predictions should match expected values"""
    predict_udf = make_quickgrove_udf(float_model_path)
    result = feature_table.mutate(pred=predict_udf.on_expr).execute()

    np.testing.assert_almost_equal(
        result["pred"].values, result["expected_pred"].values, decimal=3
    )


def test_make_quickgrove_udf_signature(float_model_path):
    """quickgrove UDF should have correct signature with float64 inputs, float32 output"""
    predict_fn = make_quickgrove_udf(float_model_path)

    assert predict_fn.__signature__.return_annotation == dt.float32
    assert all(
        p.annotation == dt.float64 for p in predict_fn.__signature__.parameters.values()
    )
    assert predict_fn.__name__ == "pretrained_model"


def test_make_quickgrove_udf_mixed_features(mixed_model_path):
    """quickgrove UDF should support int64 and boolean feature types"""
    predict_fn = make_quickgrove_udf(mixed_model_path)
    assert "i" in predict_fn.model.feature_types


def test_make_quickgrove_udf__repr(mixed_model_path):
    """quickgrove UDF repr should include model metadata"""
    predict_fn = make_quickgrove_udf(mixed_model_path)
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


def test_quickgrove_hyphen_name(feature_table, hyphen_model_path):
    assert "-" in hyphen_model_path.name
    with pytest.raises(
        ValueError,
        match="The argument model_name was None and the name extracted from the path is not a valid Python identifier",
    ):
        make_quickgrove_udf(hyphen_model_path)

    with pytest.raises(
        ValueError, match="The argument model_name is not a valid Python identifier"
    ):
        make_quickgrove_udf(hyphen_model_path, "diamonds-model")

    predict_udf = make_quickgrove_udf(hyphen_model_path, model_name="diamonds_model")
    result = feature_table.mutate(pred=predict_udf.on_expr).execute()

    np.testing.assert_almost_equal(
        result["pred"].values, result["expected_pred"].values, decimal=3
    )
