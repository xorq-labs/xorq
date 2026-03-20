"""End-to-end catalog usage example.

Demonstrates: init, add from expression, aliases, list, retrieve, remove.
Runs against a local temporary directory — no remote storage required.
"""

import tempfile
from pathlib import Path

import xorq.api as xo
from xorq.catalog.catalog import Catalog


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "my-catalog"

        # --- Initialize a catalog ---
        catalog = Catalog.from_repo_path(repo_path, init=True)
        print(f"Initialized catalog at {catalog.repo_path}")

        # --- Add an expression with aliases ---
        expr = xo.memtable({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        entry = catalog.add(expr, sync=False, aliases=("latest", "v1"))
        print(f"Added entry: {entry.name}")
        print(f"  kind:     {entry.kind}")
        print(f"  backends: {entry.backends}")

        # --- List entries and aliases ---
        print(f"\nEntries:  {catalog.list()}")
        print(f"Aliases:  {catalog.list_aliases()}")

        # --- Look up by alias ---
        for ca in catalog.catalog_aliases:
            print(f"  {ca.alias} -> {ca.catalog_entry.name}")

        # --- Retrieve the expression ---
        retrieved_entry = catalog.get_catalog_entry(entry.name)
        loaded_expr = retrieved_entry.expr
        print(f"\nLoaded expr type: {type(loaded_expr).__name__}")

        # --- Export archive to disk ---
        export_dir = Path(tmpdir) / "exports"
        export_dir.mkdir()
        zip_path = catalog.get_zip(entry.name, dir_path=export_dir)
        print(f"Exported to: {zip_path}")

        # --- Add a second entry and reassign alias ---
        expr2 = xo.memtable({"x": [10, 20], "y": ["d", "e"]})
        entry2 = catalog.add(expr2, sync=False)
        catalog.add_alias(entry2.name, "latest", sync=False)
        print(f"\nReassigned 'latest' -> {entry2.name}")

        # View alias revision history
        latest_alias = next(
            ca for ca in catalog.catalog_aliases if ca.alias == "latest"
        )
        print("Alias 'latest' history:")
        for rev_entry, commit in latest_alias.list_revisions():
            print(f"  {commit.authored_datetime}: {rev_entry.name}")

        # --- Remove an entry ---
        catalog.remove(entry.name, sync=False)
        print(f"\nRemoved {entry.name}")
        print(f"Remaining entries: {catalog.list()}")
        print(f"Remaining aliases: {catalog.list_aliases()}")

        # --- Consistency check ---
        catalog.assert_consistency()
        print("\nConsistency check passed.")


if __name__ == "__main__":
    main()
