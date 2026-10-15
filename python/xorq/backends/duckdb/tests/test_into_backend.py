"""into_backend fan-out edge cases on duckdb.

The shared shapes live in ``xorq.backends.tests.into_backend_fanout``; the
tests at the end pin shapes where duckdb's exact scan-count contract is what
matters.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import duckdb
import pandas as pd
import pytest

import xorq.api as xo
from xorq.backends.tests.into_backend_fanout import *  # noqa: F403
from xorq.backends.tests.into_backend_fanout import reader_counts


@pytest.fixture
def target() -> xo.Backend:
    return xo.duckdb.connect()


def test_into_backend_asof_tolerance_single_scan(
    df: pd.DataFrame, target: xo.Backend
) -> None:
    # #983/#2086: the tolerance lowering is a null-out projection over a single
    # ASOF LEFT JOIN, so each input is scanned once. The old filter+re-join
    # lowering scanned the left twice, consuming one-shot readers and producing
    # empty results -- assert both the single scan and a non-empty result.
    left = pd.DataFrame(
        {"site": ["a", "b"], "ts": [datetime(2024, 1, 1), datetime(2024, 1, 2)]}
    )
    right = pd.DataFrame(
        {
            "site": ["a", "b"],
            "ev": ["x", "y"],
            "ts": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        }
    )
    lt = xo.memtable(left).into_backend(target, "l")
    rt = xo.memtable(right).into_backend(target, "r")
    expr = lt.asof_join(
        rt, on="ts", predicates="site", tolerance=timedelta(seconds=1)
    ).drop("ts_right")
    assert reader_counts(expr) == [1, 1]
    assert not expr.execute().empty


def test_into_backend_partition_window_undercounts_and_raises(
    df: pd.DataFrame, target: xo.Backend
) -> None:
    # Current-behavior test (not a fix): a PARTITION BY-only aggregate window
    # with no ORDER BY undercounts. The sqlglot AST sees one Table node, but
    # duckdb lowers ``sum(v) OVER (PARTITION BY k)`` to a GROUP BY self-join
    # that scans its input twice, so max_readers=1 is one short and the second
    # reader trips the hard cap. ``max_readers`` is a best-effort lower bound,
    # not an exact scan count -- see ADR-0013 ("Counting accuracy").
    #
    # This pins the failure exactly: the day the AST count learns this shape
    # (or duckdb stops re-scanning), this test fails loudly and should be
    # converted to assert a correct, non-empty result.
    rt = xo.memtable(df).into_backend(target, "t")
    expr = rt.mutate(s=rt.v.sum().over(xo.window(group_by="k")))
    assert reader_counts(expr) == [1]
    with pytest.raises(duckdb.Error, match="Maximum number of readers"):
        expr.execute()
