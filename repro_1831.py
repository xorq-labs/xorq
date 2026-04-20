"""
Reproducer for https://github.com/xorq-labs/xorq/issues/1831

con.read_csv() / con.read_parquet() registers a *lazy file scan* in the backend.
At build time, build_expr() should serialize it as method_name=read_csv with
hash_path pointing to the original file, so the expression remains reproducible
from the source data.

Instead, the build converts it to method_name=read_parquet with hash_path pointing
to a parquet snapshot *inside the build dir*, losing the original file path entirely.
The expression is no longer an accurate representation of what the user wrote.

Expected: method_name=read_csv, hash_path=/original/data.csv
Actual:   method_name=read_parquet, hash_path=<build-dir>/database_tables/<hash>.parquet
"""

import tempfile
import pathlib

import yaml12

import xorq.api as xo
from xorq.ibis_yaml.compiler import build_expr
from xorq.ibis_yaml.enums import DumpFiles


def find_reads(d):
    match d:
        case {"op": "Read", **rest}:
            return [rest]
        case dict():
            return [r for v in d.values() for r in find_reads(v)]
        case list():
            return [r for v in d for r in find_reads(v)]
        case _:
            return []


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        csv_path = tmp / "data.csv"
        csv_path.write_text("a,b\n1,2\n3,4\n5,6\n")

        con = xo.connect()
        t = con.read_csv(str(csv_path), table_name="data")
        expr = t.filter(t.a > 1)

        print(f"Building expression from con.read_csv({csv_path.name})...")
        build_path = build_expr(expr, builds_dir=tmp / "builds")

        loaded_yaml = yaml12.parse_yaml((build_path / DumpFiles.expr).read_text())
        reads = find_reads(loaded_yaml)

        print()
        for read in reads:
            kw = dict(read.get("read_kwargs", []))
            method = read.get("method_name")
            hash_path = kw.get("hash_path", "")
            read_path = kw.get("read_path", "")

            print(f"  method_name : {method}")
            print(f"  hash_path   : {hash_path}")
            print(f"  read_path   : {read_path}")
            print()

            if method != "read_csv" or str(csv_path) not in hash_path:
                print("BUG REPRODUCED:")
                print(f"  expected method_name=read_csv, got {method!r}")
                print(f"  expected hash_path to contain {csv_path.name}, got {pathlib.Path(hash_path).name!r}")
                print()
                print("The original file path is lost. The YAML no longer represents")
                print("what the user wrote — it cannot be reproduced from source data.")
            else:
                print("OK: original file path preserved in YAML (bug is fixed).")


if __name__ == "__main__":
    main()
