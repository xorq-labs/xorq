"""Write-Audit-Publish (WAP) into an Iceberg table via branch-based isolation.

WAP is a publish gate built from existing primitives — no dedicated API needed.
The audit aggregate is a pipeline breaker, so the tee'd write fully drains the
staging branch before the publish decision runs.

  * **Write**   `tee(BackendSink)` writes to a staging branch on the Iceberg table.
  * **Audit**   `agg` UDAF drains the stream into a single `bool`.
  * **Publish** Pass: fast-forward main to the staging snapshot, then remove the
    branch. Fail: staging branch retained for inspection, main untouched.
    Both paths are metadata-only — zero data rewritten.

DQ check from real soil/sensor pipelines: readings must sit in a plausible °C
range; out-of-range (often a Fahrenheit/Celsius mixup) fails the audit.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

import xorq.api as xo
from xorq.sinking import make_iceberg_branch_wap_expr


STAGING = "staging"
FINAL = "sensor_readings"

good = {"sensor": ["s1", "s2", "s3", "s4"], "temp_c": [9.4, 10.1, 11.8, 9.9]}
good2 = {"sensor": ["s5", "s6", "s7", "s8"], "temp_c": [10.0, 9.7, 11.2, 10.4]}
bad = {"sensor": ["s1", "s2", "s3", "s4"], "temp_c": [9.4, 10.1, 88.0, 9.9]}


def audit_in_range(df: pd.DataFrame) -> bool:
    return bool(df["temp_c"].between(-10.0, 50.0).all())


if __name__ == "__pytest_main__":
    with tempfile.TemporaryDirectory() as d:
        warehouse = str(Path(d) / "warehouse")
        con = xo.pyiceberg.connect(warehouse_path=warehouse)

        # pass: first batch -> creates table + staging branch, publishes to main
        src = xo.connect().register(xo.memtable(good), table_name="src_good")
        out = src.pipe(
            make_iceberg_branch_wap_expr(con, FINAL),
            STAGING,
            FINAL,
            audit_in_range,
        ).execute()
        print("PASS (create):", out.to_string(index=False))

        assert out["passed"].iloc[0]
        assert out["published"].iloc[0]
        assert FINAL in con.list_tables(), "published data should exist at final"
        ice = con.catalog.load_table(f"{con.namespace}.{FINAL}")
        # staging branch should be removed after successful publish
        assert STAGING not in ice.refs(), "staging branch should be cleaned up"
        n_created = len(con.table(FINAL).execute())
        print(f"  -> created final with {n_created} rows\n")

        # pass: second batch -> appends via staging branch
        src2 = xo.connect().register(xo.memtable(good2), table_name="src_good2")
        out = src2.pipe(
            make_iceberg_branch_wap_expr(con, FINAL),
            STAGING,
            FINAL,
            audit_in_range,
        ).execute()
        print("PASS (append):", out.to_string(index=False))

        assert out["published"].iloc[0]
        n_appended = len(con.table(FINAL).execute())
        assert n_appended == n_created + 4, "second validated batch should append"
        print(f"  -> appended; final now has {n_appended} rows\n")

        # fail: bad data -> staging branch kept for inspection, main untouched
        con2 = xo.pyiceberg.connect(warehouse_path=str(Path(d) / "warehouse2"))
        src_bad = xo.connect().register(xo.memtable(bad), table_name="src_bad")
        out = src_bad.pipe(
            make_iceberg_branch_wap_expr(con2, FINAL),
            STAGING,
            FINAL,
            audit_in_range,
        ).execute()
        print("FAIL:", out.to_string(index=False))

        assert not out["passed"].iloc[0]
        assert not out["published"].iloc[0]
        assert FINAL in con2.list_tables(), "table should exist (created for branching)"
        ice2 = con2.catalog.load_table(f"{con2.namespace}.{FINAL}")
        # staging branch retained so rejected data can be inspected
        assert STAGING in ice2.refs(), (
            "staging branch should be retained for inspection"
        )
        print("  -> audit failed; staging branch retained, main untouched\n")

    pytest_examples_passed = True
