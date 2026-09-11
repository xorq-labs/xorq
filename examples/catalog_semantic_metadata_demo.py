"""Demo: catalog-level semantic metadata sidecar."""

import tempfile
from pathlib import Path

import pandas as pd

import xorq.api as xo
from xorq.catalog.catalog import Catalog

catalog_dir = Path(tempfile.mkdtemp()) / "sales-catalog"

con = xo.connect()
orders = con.create_table(
    "orders",
    pd.DataFrame(
        {
            "region": ["west", "east", "west", "east", "north"],
            "amount": [120.0, 85.0, 45.0, 200.0, 150.0],
        }
    ),
)
expr = orders.group_by("region").agg(total=orders.amount.sum())

catalog = Catalog.from_repo_path(catalog_dir, init=True)
entry = catalog.add(
    expr,
    aliases=("sales-by-region",),
    metadata={
        "domain": "sales",
        "owner": "data-platform",
        "tags": ["curated", "daily"],
        "description": "Total order amount by region",
    },
    sync=False,
)
print("add ok:", entry.semantic_metadata)

reloaded = Catalog.from_repo_path(catalog_dir, init=False).get_catalog_entry(
    "sales-by-region", maybe_alias=True
)
print("reload ok:", reloaded.semantic_metadata)
assert reloaded.semantic_metadata == entry.semantic_metadata

untagged_expr = orders.filter(orders.amount > 100)
untagged = catalog.add(untagged_expr, sync=False)
print("untagged (no metadata):", untagged.semantic_metadata)
assert untagged.semantic_metadata is None
