"""Unit coverage for the parquet WAP (Write-Audit-Publish) strategy.

The iceberg strategies are covered in
``xorq/backends/pyiceberg/tests/test_wap.py``; this keeps the backend-agnostic
parquet path tested alongside the rest of the writes layer.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import xorq.api as xo
import xorq.vendor.ibis.expr.types as ir
from xorq.writes import make_parquet_wap_expr
from xorq.writes.wap import (
    FINAL,
    PASSED,
    STAGING,
    _make_publish_with_parquet,
)


good = {"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]}
bad = {"a": [1, None, 3, 4], "b": ["w", "x", "y", "z"]}


def no_nulls(df: pd.DataFrame) -> bool:
    return bool(df["a"].notna().all())


def _wap_expr(data: dict, staging: str, final: str, table_name: str) -> ir.Table:
    return (
        xo.connect()
        .register(xo.memtable(data), table_name=table_name)
        .pipe(make_parquet_wap_expr, staging, final, no_nulls)
    )


def test_parquet_wap_pass(tmp_path: Path) -> None:
    staging = str(tmp_path / "staging.parquet")
    final = str(tmp_path / "final.parquet")
    out = _wap_expr(good, staging, final, "src_good").execute()

    assert out["passed"].iloc[0]
    assert out["published"].iloc[0]
    assert Path(final).exists()
    assert not Path(staging).exists(), "staging is consumed on publish"
    assert len(pd.read_parquet(final)) == 4


def test_parquet_wap_append(tmp_path: Path) -> None:
    # Repeated passing runs accumulate in final, matching the iceberg strategies.
    staging = str(tmp_path / "staging.parquet")
    final = str(tmp_path / "final.parquet")
    _wap_expr(good, staging, final, "src1").execute()
    _wap_expr(good, staging, final, "src2").execute()

    assert len(pd.read_parquet(final)) == 8
    assert not Path(staging).exists(), "staging is consumed on each publish"


def test_parquet_wap_fail(tmp_path: Path) -> None:
    staging = str(tmp_path / "staging.parquet")
    final = str(tmp_path / "final.parquet")
    out = _wap_expr(bad, staging, final, "src_bad").execute()

    assert not out["passed"].iloc[0]
    assert not out["published"].iloc[0]
    assert not Path(final).exists(), "failing audit must not publish"
    assert Path(staging).exists(), "rejected data is retained in staging"


def test_parquet_publish_guard_rejects_multi_row() -> None:
    publish = _make_publish_with_parquet()
    df = pd.DataFrame(
        {
            STAGING: ["s.parquet", "s.parquet"],
            FINAL: ["f.parquet", "f.parquet"],
            PASSED: [True, True],
        }
    )
    with pytest.raises(ValueError, match="expected 1 row"):
        publish.fn(df)


def test_parquet_wap_rerun_after_fail_raises(tmp_path: Path) -> None:
    # A failed audit retains the staging file. Re-running against the same target
    # then writes create-mode over an existing file, which the sink refuses
    # rather than silently clobbering (FileExistsError surfaces as ValueError).
    staging = str(tmp_path / "staging.parquet")
    final = str(tmp_path / "final.parquet")
    _wap_expr(bad, staging, final, "src_bad").execute()
    assert Path(staging).exists()

    with pytest.raises(ValueError, match="already exists"):
        _wap_expr(good, staging, final, "src_retry").execute()


def test_parquet_publish_requires_staging_present(tmp_path: Path) -> None:
    # Backstop for the WAP ordering invariant: publishing before the staging
    # write committed (e.g. an async sink) raises rather than publishing nothing.
    final = tmp_path / "final.parquet"
    publish = _make_publish_with_parquet()
    df = pd.DataFrame(
        {
            STAGING: [str(tmp_path / "missing.parquet")],
            FINAL: [str(final)],
            PASSED: [True],
        }
    )
    with pytest.raises(RuntimeError, match="missing at publish"):
        publish.fn(df)
