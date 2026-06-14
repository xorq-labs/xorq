"""Write-Audit-Publish (WAP) into an Iceberg table from ADR-0014 `tee` / `BackendSink`.

WAP = publish gate. No dedicated API; falls out of existing primitives. Audit
aggregate is a pipeline breaker, so the tee'd write fully drains before publish.

  * **Write**   `tee(BackendSink(con, staging, mode="create"))` -> staging table.
  * **Audit**   `agg.pandas_df` UDAF drains the stream -> one `bool`. Draining
    finalizes staging before publish runs.
  * **Publish** `scalar.pyarrow` UDF, catalog-level, zero data rewritten. Pass:
    first batch `rename_table(staging -> final)`, later batches `add_files` the
    staging parquet into final by reference. Fail: staging kept, final untouched.

DQ check from real soil/sensor pipelines: readings must sit in a plausible °C
range; out-of-range (often a Fahrenheit/Celsius mixup) fails the audit.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

import xorq.api as xo
import xorq.expr.datatypes as dt
from xorq.api import _
from xorq.expr.udf import agg, scalar
from xorq.sinking import BackendSink


if TYPE_CHECKING:
    from xorq.backends.pyiceberg import Backend as IcebergBackend
    from xorq.vendor.ibis.expr.types import Table


STAGING = "sensor_staging"
FINAL = "sensor_readings"

BATCH = 24  # rows per batch


# deterministic soil readings; plausible °C sawtooth in [9.0, 11.6], no RNG.
def batch(prefix: str, n: int = BATCH) -> dict:
    return {
        "sensor": [f"{prefix}{i}" for i in range(n)],
        "temp_c": [round(9.0 + (i % 7) * 0.4, 1) for i in range(n)],
    }


good = batch("a")  # first validated batch
good2 = batch("b")  # second validated batch -> appends
bad = {
    **batch("c"),
    "temp_c": [*batch("c")["temp_c"][:5], 88.0, *batch("c")["temp_c"][6:]],
}
# bad[5] = 88.0 -> a Fahrenheit reading slipped in, out of plausible °C range


def audit_in_range(df: pd.DataFrame) -> bool:
    return bool(df["temp_c"].between(-10.0, 50.0).all())


def make_publish(con: IcebergBackend) -> Callable[..., pa.Array]:
    """Build the publishing UDF, closing over the Iceberg catalog.

    Both pass paths are metadata-only, zero data rewritten:
      * final absent: rename staging -> final.
      * final exists: add_files staging's parquet into final by reference, then
        drop staging (drop removes metadata only, not the now-shared data files).
    Fail: leave staging, never touch final.
    """
    catalog, namespace = con.catalog, con.namespace
    staging_id, final_id = f"{namespace}.{STAGING}", f"{namespace}.{FINAL}"

    @scalar.pyarrow
    def publish(passed: dt.boolean) -> dt.boolean:
        p = passed[0].as_py()
        if p:
            if catalog.table_exists(final_id):
                staging = catalog.load_table(staging_id)
                files = [task.file.file_path for task in staging.scan().plan_files()]
                catalog.load_table(final_id).add_files(files)
                catalog.drop_table(staging_id)
            else:
                catalog.rename_table(staging_id, final_id)
        return pa.array([bool(p)], type=pa.bool_())

    return publish


def run_wap(
    con: IcebergBackend, source: Table, audit_fn: Callable[[pd.DataFrame], bool]
) -> pd.DataFrame:
    """write (tee) -> audit (agg) -> publish (mutate) -> execute. Returns receipt."""
    teed = source.tee(BackendSink(con=con, table_name=STAGING, mode="create"))
    audit_udaf = agg.pandas_df(
        fn=audit_fn,
        schema=teed.schema(),
        return_type=dt.boolean,
        name="audit",
    )
    publish = make_publish(con)
    return (
        teed.agg(passed=audit_udaf.on_expr(teed))
        .mutate(published=publish(_.passed))
        .execute()
    )


if __name__ == "__pytest_main__":
    with tempfile.TemporaryDirectory() as d:
        warehouse = str(Path(d) / "warehouse")

        # pass, final absent -> rename creates final
        con = xo.pyiceberg.connect(warehouse_path=warehouse)
        src = xo.connect().register(xo.memtable(good), table_name="src_good")

        receipt = run_wap(con, src, audit_in_range)
        print("PASS path receipt (first batch -> create):")
        print(receipt.to_string(index=False))

        assert receipt["passed"].iloc[0]
        assert receipt["published"].iloc[0]
        assert FINAL in con.list_tables(), "published data should exist at final"
        assert STAGING not in con.list_tables(), "staging consumed by the promote"
        n_created = len(con.table(FINAL).execute())
        print(f"  -> created final with {n_created} rows\n")

        # pass, final exists -> append into final
        src2 = xo.connect().register(xo.memtable(good2), table_name="src_good2")

        receipt = run_wap(con, src2, audit_in_range)
        print("APPEND path receipt (second batch -> append):")
        print(receipt.to_string(index=False))

        assert receipt["published"].iloc[0]
        assert STAGING not in con.list_tables(), "staging consumed by the promote"
        n_appended = len(con.table(FINAL).execute())
        assert n_appended == n_created + BATCH, "second validated batch should append"
        print(f"  -> appended; final now has {n_appended} rows\n")

        # fail -> nothing published, staging kept
        con2 = xo.pyiceberg.connect(warehouse_path=str(Path(d) / "warehouse2"))
        src2 = xo.connect().register(xo.memtable(bad), table_name="src_bad")

        receipt = run_wap(con2, src2, audit_in_range)
        print("FAIL path receipt:")
        print(receipt.to_string(index=False))

        assert not receipt["passed"].iloc[0]
        assert not receipt["published"].iloc[0]
        assert FINAL not in con2.list_tables(), "failing audit must not publish"
        assert STAGING in con2.list_tables(), "rejected data retained in staging"
        print("  -> audit failed; rejected data retained in staging, final absent\n")

    pytest_examples_passed = True
