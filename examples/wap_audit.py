"""Write-Audit-Publish (WAP) on top of ADR-0014's `tee` / `ParquetSink`.

WAP is a publish gate: data is written to a *staging* location, an *audit*
decides whether it is fit to publish, and only on a passing audit is it moved
to its *final* location. A failing audit leaves the staging file behind for
inspection and never touches the final target.

This example wires WAP out of primitives that already exist in xorq — there is
no dedicated WAP API:

  * **Write**  `expr.tee(ParquetSink(staging))` writes every row that flows
    through it to ``staging`` as a side effect (ADR-0014).
  * **Audit**  `agg.pandas_df` builds a UDAF that receives the whole staged
    DataFrame and returns a single ``bool``. Being an aggregate, it must drain
    the entire tee'd stream to produce its value — which is exactly what
    finalizes the staging file before the publish step runs.
  * **Publish**  a `scalar.pyarrow` UDF that runs on the audit's one-row
    result. On a passing audit it moves ``staging`` -> ``final`` with
    ``os.link`` + ``os.unlink`` (atomic, and raising if ``final`` already
    exists); on a failing audit it does nothing.

The dataflow itself enforces write-then-audit-then-publish ordering: the
aggregate is a pipeline breaker, so the tee is fully drained (staging
finalized) before the scalar publish UDF ever sees a row.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa

import xorq.api as xo
import xorq.expr.datatypes as dt
from xorq.expr.udf import agg, scalar
from xorq.sinking import ParquetSink


@scalar.pyarrow
def publish(staging: dt.string, final: dt.string, passed: dt.boolean) -> dt.boolean:
    """Move staging -> final iff the audit passed. Runs on the 1-row audit result.

    Uses ``os.link`` + ``os.unlink`` so the publish is atomic and refuses to
    clobber an existing ``final`` (``os.link`` raises ``FileExistsError``).
    Returns whether the publish happened (== the audit verdict).
    """
    s, f, p = staging[0].as_py(), final[0].as_py(), passed[0].as_py()
    if p:
        os.link(s, f)  # atomic create-or-fail: raises if `final` already exists
        os.unlink(s)  # consume staging -> net move
    return pa.array([bool(p)], type=pa.bool_())


def run_wap(t, staging: str, final: str, audit_fn):
    """Build and execute the WAP pipeline; return the one-row receipt DataFrame."""
    teed = t.tee(ParquetSink(path=staging, mode="create"))
    audit_udaf = agg.pandas_df(
        fn=audit_fn,
        schema=teed.schema(),
        return_type=dt.boolean,
        name="audit",
    )
    audited = teed.agg(passed=audit_udaf.on_expr(teed))
    receipt = audited.mutate(
        published=publish(xo.literal(staging), xo.literal(final), audited.passed)
    )
    return receipt.execute()


def main() -> None:
    data = {"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]}

    # audit predicate: every value in column `a` must be present (no nulls)
    def audit_no_nulls(df: pd.DataFrame) -> bool:
        return bool(df["a"].notna().all())

    # ---- pass path: audit succeeds -> data published to `final` -------------
    with tempfile.TemporaryDirectory() as d:
        staging = str(Path(d) / "staging.parquet")
        final = str(Path(d) / "final.parquet")
        con = xo.connect()
        t = con.register(xo.memtable(data), table_name="src_pass")

        receipt = run_wap(t, staging, final, audit_no_nulls)
        print("PASS path receipt:")
        print(receipt.to_string(index=False))

        assert receipt["passed"].iloc[0]
        assert receipt["published"].iloc[0]
        assert Path(final).exists(), "published data should exist at final"
        assert not Path(staging).exists(), "staging should be consumed by the move"
        print(f"  -> published {len(pd.read_parquet(final))} rows to final\n")

    # ---- fail path: audit fails -> nothing published, staging retained ------
    def audit_always_fails(df: pd.DataFrame) -> bool:
        return False

    with tempfile.TemporaryDirectory() as d:
        staging = str(Path(d) / "staging.parquet")
        final = str(Path(d) / "final.parquet")
        con = xo.connect()
        t = con.register(xo.memtable(data), table_name="src_fail")

        receipt = run_wap(t, staging, final, audit_always_fails)
        print("FAIL path receipt:")
        print(receipt.to_string(index=False))

        assert not receipt["passed"].iloc[0]
        assert not receipt["published"].iloc[0]
        assert not Path(final).exists(), "failing audit must not publish"
        assert Path(staging).exists(), "rejected data is retained in staging"
        print("  -> audit failed; rejected data retained at staging, final absent\n")


if __name__ == "__main__":
    main()
