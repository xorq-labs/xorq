"""Write-Audit-Publish (WAP) into a Parquet file via copy-on-pass.

WAP is a publish gate built from existing primitives — no dedicated API needed.
The audit aggregate is a pipeline breaker, so the tee'd write fully drains the
staging file before the publish decision runs.

  * **Write**   `tee(ParquetWriteThrough)` writes to a staging Parquet file.
  * **Audit**   `agg` UDAF drains the stream into a single `bool`.
  * **Publish** Pass: copy staging -> final; staging retained.
    Fail: staging retained for inspection, final untouched.

DQ check: every value in column ``a`` must be non-null; null values fail the
audit before publish.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path

import pandas as pd

import xorq.api as xo
from xorq.writes import make_parquet_wap_expr


data = {"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]}
bad_data = {"a": [1, None, 3, 4], "b": ["w", "x", "y", "z"]}


def audit_no_nulls(df: pd.DataFrame) -> bool:
    return bool(df["a"].notna().all())


tmp = tempfile.mkdtemp()
atexit.register(shutil.rmtree, tmp)
staging = str(Path(tmp) / "staging.parquet")
final = str(Path(tmp) / "final.parquet")

pass_expr = (
    xo.connect()
    .register(xo.memtable(data), table_name="src_pass")
    .pipe(make_parquet_wap_expr, staging, final, audit_no_nulls)
)

fail_staging = str(Path(tmp) / "fail_staging.parquet")
fail_final = str(Path(tmp) / "fail_final.parquet")

fail_expr = (
    xo.connect()
    .register(xo.memtable(bad_data), table_name="src_fail")
    .pipe(make_parquet_wap_expr, fail_staging, fail_final, audit_no_nulls)
)


if __name__ in ("__main__", "__pytest_main__"):
    # pass: audit succeeds -> data published to final
    out = pass_expr.execute()
    print("PASS path receipt:")
    print(out.to_string(index=False))

    assert out["passed"].iloc[0]
    assert out["published"].iloc[0]
    assert Path(final).exists(), "published data should exist at final"
    assert Path(staging).exists(), "staging is retained after the copy"
    print(f"  -> published {len(pd.read_parquet(final))} rows to final\n")

    # fail: audit fails -> nothing published, staging retained
    out = fail_expr.execute()
    print("FAIL path receipt:")
    print(out.to_string(index=False))

    assert not out["passed"].iloc[0]
    assert not out["published"].iloc[0]
    assert not Path(fail_final).exists(), "failing audit must not publish"
    assert Path(fail_staging).exists(), "rejected data is retained in staging"
    print("  -> audit failed; rejected data retained at staging, final absent\n")

    # reuse-after-failure: the retained staging file blocks an identical retry,
    # since create-mode refuses to clobber it. Clear it (or stage elsewhere) first.
    # The writer's FileExistsError crosses Arrow's C Data interface as a ValueError.
    try:
        fail_expr.execute()
    except ValueError as exc:
        assert "already exists" in str(exc), exc
        print("  -> retry refused: staging from the failed run still exists")
    else:
        raise AssertionError("retry should refuse to clobber retained staging")

    Path(fail_staging).unlink()
    out = fail_expr.execute()
    assert not out["passed"].iloc[0], "same bad data still fails the audit"
    assert not out["published"].iloc[0], "failing retry must not publish"
    assert not Path(fail_final).exists(), "final must stay absent after failing retry"
    print("  -> staging cleared; retry runs again (audit still fails on bad data)\n")

    pytest_examples_passed = True
