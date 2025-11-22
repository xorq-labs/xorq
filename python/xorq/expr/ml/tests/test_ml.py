from __future__ import annotations

import pytest

import xorq.api as xo


@pytest.mark.parametrize(
    "method",
    [
        "train_test_splits",
        "deferred_fit_predict",
        "deferred_fit_predict_sklearn",
        "deferred_fit_transform",
        "deferred_fit_transform_series_sklearn",
        "make_quickgrove_udf",
        "rewrite_quickgrove_expr",
    ],
)
def test_top_level_ml(method):
    assert hasattr(xo, method)
    assert hasattr(xo.ml, method)
