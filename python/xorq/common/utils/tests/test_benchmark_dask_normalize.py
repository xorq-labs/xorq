import dask
import pandas as pd
import pytest

import xorq.api as xo
import xorq.common.utils.dask_normalize  # noqa: F401
from xorq.common.utils.dask_normalize.dask_normalize_expr import (
    _normalize_data_leaf,
    compute_expr_token,
    expr_metadata,
    normalize_op_split,
)


pytestmark = pytest.mark.core


def _make_simple():
    con = xo.connect()
    con.create_table("t", pd.DataFrame({"a": range(100), "b": range(100)}))
    return con.table("t").filter(xo._.a > 50).group_by("b").agg(s=xo._.a.sum())


def _make_deep_pipeline():
    con = xo.connect()
    con.create_table("t", pd.DataFrame({"a": range(100), "b": range(100)}))
    e = con.table("t")
    for i in range(50):
        e = e.mutate(**{f"c{i}": xo._.a + i}).filter(xo._.b > i % 5)
    return e


def _make_nested_into_backend():
    """Cross-engine pipeline with nested ``into_backend`` boundaries.

    Exercises the recursion in ``_opaque_structural_name``:  each
    ``RemoteTable`` boundary triggers a structural normalization of its inner
    sub-expression.  Without per-call memoization the pipeline pays
    ``O(depth²)`` for shared sub-expression roots; the memo bounds it to
    ``O(depth)``.
    """
    src = xo.connect()
    src.create_table("t", pd.DataFrame({"a": range(100), "b": range(100)}))
    e = src.table("t")
    for i in range(8):
        target = xo.connect()
        e = (
            e.mutate(**{f"c{i}": xo._.a + i})
            .into_backend(target, name=f"step_{i}")
            .filter(xo._.b > i % 5)
        )
    return e


CASES = {
    "simple_filter_agg": _make_simple,
    "pipeline_50_steps": _make_deep_pipeline,
    "nested_into_backend": _make_nested_into_backend,
}


@pytest.mark.benchmark
@pytest.mark.parametrize("case", list(CASES))
def test_benchmark_tokenize_full(benchmark, case):
    """End-to-end ``dask.base.tokenize`` — guards against regressions in the
    standard tokenize path."""
    expr = CASES[case]()
    benchmark(dask.base.tokenize, expr)


@pytest.mark.benchmark
@pytest.mark.parametrize("case", list(CASES))
def test_benchmark_tokenize_cached_structural(benchmark, case):
    """Cheap path: re-hash data leaves only, then md5-combine with a precomputed
    structural hash via :func:`compute_expr_token`. The split's intended payoff."""
    expr = CASES[case]()
    md = expr_metadata(expr)
    leaf_dts, _, _ = normalize_op_split(expr)
    structural_hash = md["structural_hash"]

    def cached():
        slot_hashes = [dask.base.tokenize(_normalize_data_leaf(dt)) for dt in leaf_dts]
        return compute_expr_token(slot_hashes, structural_hash)

    benchmark(cached)
