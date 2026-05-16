from __future__ import annotations

import pytest

import xorq.api as xo


pytestmark = pytest.mark.core


@pytest.mark.parametrize(
    "method",
    [
        "train_test_splits",
        "deferred_fit_predict",
        "deferred_fit_predict_sklearn",
        "deferred_fit_transform",
        "deferred_fit_transform_series_sklearn",
    ],
)
def test_top_level_ml(method):
    assert hasattr(xo, method)
    assert hasattr(xo.ml, method)
