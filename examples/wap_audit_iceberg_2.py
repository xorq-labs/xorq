import tempfile
from pathlib import Path

import xorq.api as xo
from xorq.sinking import make_iceberg_branch_wap_expr


STAGING = "staging"  # branch name
FINAL = "sensor_readings"  # table name

good = {"sensor": ["s1", "s2", "s3", "s4"], "temp_c": [9.4, 10.1, 11.8, 9.9]}
good2 = {"sensor": ["s5", "s6", "s7", "s8"], "temp_c": [10.0, 9.7, 11.2, 10.4]}
bad = {"sensor": ["s1", "s2", "s3", "s4"], "temp_c": [9.4, 10.1, 88.0, 9.9]}


def audit_in_range(df):
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

        # fail: bad data -> staging branch kept, main untouched
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
        assert STAGING in ice2.refs(), (
            "staging branch should be retained for inspection"
        )
        print("  -> audit failed; staging branch retained, main untouched\n")

    pytest_examples_passed = True
