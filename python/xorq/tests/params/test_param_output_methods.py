"""Tests that named parameters flow correctly through all output methods.

Parametrizes over execute, to_pyarrow, to_pyarrow_batches, to_parquet,
to_csv, and to_json.  Separate tests verify cached and remote-table exprs.
"""

import pandas as pd
import pyarrow.parquet as pq
import pytest

import xorq.api as xo


# ---------------------------------------------------------------------------
# Helpers — one callable per output method, uniform signature (expr, params, tmp_path)
# ---------------------------------------------------------------------------


def _via_execute(expr, params, _tmp_path):
    return list(expr.execute(params=params)["x"])


def _via_to_pyarrow(expr, params, _tmp_path):
    return expr.to_pyarrow(params=params)["x"].to_pylist()


def _via_to_pyarrow_batches(expr, params, _tmp_path):
    with expr.to_pyarrow_batches(params=params) as rbr:
        return rbr.read_all()["x"].to_pylist()


def _via_to_parquet(expr, params, tmp_path):
    path = tmp_path / "out.parquet"
    expr.to_parquet(path, params=params)
    return pq.read_table(path)["x"].to_pylist()


def _via_to_csv(expr, params, tmp_path):
    path = tmp_path / "out.csv"
    expr.to_csv(path, params=params)
    return pd.read_csv(path)["x"].tolist()


def _via_to_json(expr, params, tmp_path):
    path = tmp_path / "out.json"
    expr.to_json(path, params=params)
    return pd.read_json(path, lines=True)["x"].tolist()


_OUTPUT_METHODS = pytest.mark.parametrize(
    "call",
    [
        pytest.param(_via_execute, id="execute"),
        pytest.param(_via_to_pyarrow, id="to_pyarrow"),
        pytest.param(_via_to_pyarrow_batches, id="to_pyarrow_batches"),
        pytest.param(_via_to_parquet, id="to_parquet"),
        pytest.param(_via_to_csv, id="to_csv"),
        pytest.param(_via_to_json, id="to_json"),
    ],
)

# ---------------------------------------------------------------------------
# Parametrized: all methods × simple expression
# ---------------------------------------------------------------------------


@_OUTPUT_METHODS
def test_params_flow_through_output_method(call, tmp_path):
    threshold = xo.param("threshold", "float64", default=1.5)
    t = xo.memtable({"x": [1.0, 2.0, 3.0]})
    expr = t.filter(t.x > threshold)

    assert call(expr, None, tmp_path) == [2.0, 3.0]  # default=1.5
    assert call(expr, {threshold: 0.5}, tmp_path) == [1.0, 2.0, 3.0]  # custom


# ---------------------------------------------------------------------------
# CachedNode and RemoteTable — verify node traversal, not output routing
# ---------------------------------------------------------------------------


def test_params_cached_node():
    threshold = xo.param("threshold", "float64", default=1.5)
    t = xo.memtable({"x": [1.0, 2.0, 3.0]})
    expr = t.filter(t.x > threshold).cache()

    assert list(expr.execute()["x"]) == [2.0, 3.0]
    assert list(expr.execute(params={threshold: 2.5})["x"]) == [3.0]


def test_params_remote_table():
    threshold = xo.param("threshold", "float64", default=1.5)
    src = xo.memtable({"x": [1.0, 2.0, 3.0]})
    expr = src.into_backend(xo.connect()).pipe(lambda t: t.filter(t.x > threshold))

    assert list(expr.execute()["x"]) == [2.0, 3.0]
    assert list(expr.execute(params={threshold: 0.5})["x"]) == [1.0, 2.0, 3.0]


def test_params_string_keyed_dict():
    """execute(params={"name": value}) works via _transform_expr string-key path."""
    threshold = xo.param("threshold", "float64")
    t = xo.memtable({"x": [1.0, 2.0, 3.0]})
    expr = t.filter(t.x > threshold)

    assert tuple(expr.execute(params={"threshold": 1.5})["x"]) == (2.0, 3.0)
