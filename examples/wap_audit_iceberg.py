"""Write-Audit-Publish (WAP) into an Iceberg table via branch-based isolation.

WAP is a publish gate built from existing primitives — no dedicated API needed.
The audit aggregate is a pipeline breaker, so the tee'd write fully drains the
staging branch before the publish decision runs.

  * **Write**   `tee(BackendWriteThrough)` writes to a staging branch on the Iceberg table.
  * **Audit**   `agg` UDAF drains the stream into a single `bool`.
  * **Publish** Pass: fast-forward main to the staging snapshot, then remove the
    branch. Fail: staging branch retained for inspection, main untouched.
    Both paths are metadata-only — zero data rewritten.

DQ check from real soil/sensor pipelines: readings must sit in a plausible °C
range; out-of-range (often a Fahrenheit/Celsius mixup) fails the audit.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path

import pandas as pd

import xorq.api as xo
from xorq.writes import make_iceberg_wap_expr


STAGING = "staging"
FINAL = "sensor_readings"

good = {"sensor": ["s1", "s2", "s3", "s4"], "temp_c": [9.4, 10.1, 11.8, 9.9]}
good2 = {"sensor": ["s5", "s6", "s7", "s8"], "temp_c": [10.0, 9.7, 11.2, 10.4]}
bad = {"sensor": ["s1", "s2", "s3", "s4"], "temp_c": [9.4, 10.1, 88.0, 9.9]}


def audit_in_range(df: pd.DataFrame) -> bool:
    return bool(df["temp_c"].between(-10.0, 50.0).all())


tmp = tempfile.mkdtemp()
atexit.register(shutil.rmtree, tmp)
warehouse = str(Path(tmp) / "warehouse")
con = xo.pyiceberg.connect(warehouse_path=warehouse)

wap = make_iceberg_wap_expr(con, FINAL)

create_expr = (
    xo.connect()
    .register(xo.memtable(good), table_name="src_good")
    .pipe(wap, STAGING, FINAL, audit_in_range)
)

append_expr = (
    xo.connect()
    .register(xo.memtable(good2), table_name="src_good2")
    .pipe(wap, STAGING, FINAL, audit_in_range)
)

# execute in order: append_expr depends on create_expr having run first
pipeline = [
    ("create", create_expr),
    ("append", append_expr),
]

warehouse2 = str(Path(tmp) / "warehouse2")
con2 = xo.pyiceberg.connect(warehouse_path=warehouse2)

fail_expr = (
    xo.connect()
    .register(xo.memtable(bad), table_name="src_bad")
    .pipe(make_iceberg_wap_expr(con2, FINAL), STAGING, FINAL, audit_in_range)
)


if __name__ == "__pytest_main__":
    prev_count = 0
    for label, expr in pipeline:
        out = expr.execute()
        print(f"PASS ({label}):", out.to_string(index=False))

        assert out["passed"].iloc[0]
        assert out["published"].iloc[0]
        assert FINAL in con.list_tables(), "published data should exist at final"
        ice = con.catalog.load_table(f"{con.namespace}.{FINAL}")
        assert STAGING not in ice.refs(), "staging branch should be cleaned up"
        cur_count = len(con.table(FINAL).execute())
        if prev_count:
            assert cur_count == prev_count + 4, "validated batch should append"
        print(f"  -> {label}; final now has {cur_count} rows\n")
        prev_count = cur_count

    # fail: bad data -> staging branch kept for inspection, main untouched
    out = fail_expr.execute()
    print("FAIL:", out.to_string(index=False))

    assert not out["passed"].iloc[0]
    assert not out["published"].iloc[0]
    assert FINAL in con2.list_tables(), "table should exist (created for branching)"
    ice2 = con2.catalog.load_table(f"{con2.namespace}.{FINAL}")
    # staging branch retained so rejected data can be inspected
    assert STAGING in ice2.refs(), "staging branch should be retained for inspection"
    print("  -> audit failed; staging branch retained, main untouched\n")

    pytest_examples_passed = True
