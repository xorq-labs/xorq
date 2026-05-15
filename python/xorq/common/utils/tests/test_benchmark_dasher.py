"""Benchmarks for :func:`xorq.common.utils.dasher.tokenize`.

Tracks tokenization cost across three expression shapes:

* ``simple_filter_agg``    — baseline shallow expression
* ``pipeline_50_steps``    — single-backend deep mutate/filter chain
* ``nested_into_backend``  — cross-engine pipeline that exercises the
                              recursion through opaque sub-expressions
                              (``RemoteTable.remote_expr``); the
                              ``_parent_token`` path inside
                              ``_xorq_opaque_to_placeholder`` is what gets
                              stressed here.

These replace the legacy ``test_benchmark_dask_normalize`` suite that was
removed when ``dask_normalize/`` was retired in favor of ``dasher``.
"""

from __future__ import annotations

import pandas as pd
import pytest

import xorq.api as xo
from xorq.common.utils.dasher import tokenize


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

    Each ``RemoteTable`` boundary triggers a recursive ``_parent_token``
    call inside ``_xorq_opaque_to_placeholder`` to fold the inner
    expression's identity into the placeholder name; without memoization
    in ``HASHER`` this would be ``O(depth²)`` for shared sub-expression
    roots.
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
def test_benchmark_tokenize(benchmark, case):
    """End-to-end ``dasher.tokenize`` — guards against tokenization regressions."""
    expr = CASES[case]()
    benchmark(tokenize, expr)
