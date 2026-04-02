"""Demonstrates ExprBuilder: catalog a BSL expression, recover the model, rebind to new data.

With xorq: A BSL semantic model produces expressions via .query(). The result is tagged
(to_tagged), which makes it an ExprBuilder entry when cataloged. You can recover the
original SemanticModel from any cataloged expression via from_tagged, rebind to new
data, and query with different selections — all while maintaining full provenance.
"""

import tempfile
from pathlib import Path

from boring_semantic_layer import to_semantic_table
from boring_semantic_layer.expr import SemanticModel

import xorq.api as xo
from xorq.catalog.catalog import Catalog


# ---------------------------------------------------------------------------
# 1. Set up a connection and base table (dev data)
# ---------------------------------------------------------------------------

con = xo.connect()
dev_data = con.create_table(
    "flights_dev",
    {
        "origin": ["JFK", "LAX", "ORD", "JFK", "LAX", "ORD", "JFK", "LAX"],
        "destination": ["LAX", "ORD", "JFK", "ORD", "JFK", "LAX", "LAX", "JFK"],
        "carrier": ["AA", "UA", "AA", "UA", "AA", "UA", "AA", "UA"],
        "dep_delay": [10.0, -5.0, 30.0, 15.0, -2.0, 45.0, 5.0, 20.0],
        "distance": [2475, 1745, 740, 1300, 2475, 1745, 2475, 2475],
    },
)

# ---------------------------------------------------------------------------
# 2. Create a SemanticModel and query it
# ---------------------------------------------------------------------------

model = (
    to_semantic_table(dev_data)
    .with_dimensions(
        origin=lambda t: t.origin,
        destination=lambda t: t.destination,
        carrier=lambda t: t.carrier,
    )
    .with_measures(
        flight_count=lambda t: t.count(),
        avg_dep_delay=lambda t: t.dep_delay.mean(),
        total_distance=lambda t: t.distance.sum(),
    )
)

print("Available dimensions:", tuple(model.dimensions))
print("Available measures:", tuple(model.measures))

# ---------------------------------------------------------------------------
# 3. Build a tagged expression and add to catalog
# ---------------------------------------------------------------------------

expr_by_origin = model.query(
    dimensions=("origin",),
    measures=("flight_count", "avg_dep_delay"),
)
tagged_expr = expr_by_origin.to_tagged()

catalog_dir = Path(tempfile.mkdtemp()) / "example-catalog"
catalog = Catalog.from_repo_path(catalog_dir, init=True)
print(f"\nCatalog directory: {catalog_dir}")

catalog.add(tagged_expr, aliases=("origin-delays-dev",), sync=False)
print("Catalog entries:", catalog.list())
print("Catalog aliases:", catalog.list_aliases())

# ---------------------------------------------------------------------------
# 4. Check that the entry has ExprBuilder kind and builder metadata
# ---------------------------------------------------------------------------

entry = catalog.get_catalog_entry("origin-delays-dev", maybe_alias=True)
print(f"\nEntry kind: {entry.kind}")
print(f"Builder metadata: {entry.metadata.builders}")

# ---------------------------------------------------------------------------
# 5. Recover the full SemanticModel from the cataloged expression
# ---------------------------------------------------------------------------

recovered_model = catalog.get_builder_from_expr(entry.expr)
print(f"\nRecovered model type: {type(recovered_model).__name__}")
print(f"Recovered dimensions: {tuple(recovered_model.dimensions)}")
print(f"Recovered measures: {tuple(recovered_model.measures)}")

# Build a different selection from the recovered model
expr_by_dest = recovered_model.query(
    dimensions=("destination",),
    measures=("flight_count", "total_distance"),
)
print("\nFlights by destination (from recovered model):")
print(expr_by_dest.execute())

# ---------------------------------------------------------------------------
# 6. Rebind to production data — same model shape, different fact table
# ---------------------------------------------------------------------------

prd_data = con.create_table(
    "flights_prd",
    {
        "origin": ["SFO", "DEN", "ATL", "SFO", "DEN", "ATL"],
        "destination": ["DEN", "ATL", "SFO", "ATL", "SFO", "DEN"],
        "carrier": ["DL", "WN", "DL", "WN", "DL", "WN"],
        "dep_delay": [8.0, 12.0, -3.0, 22.0, 5.0, 18.0],
        "distance": [967, 1199, 2139, 1837, 1535, 1199],
    },
)

prd_model = SemanticModel(
    table=prd_data,
    dimensions=recovered_model.get_dimensions(),
    measures=recovered_model.get_measures(),
    calc_measures=recovered_model.get_calculated_measures(),
    name=recovered_model.name,
    description=recovered_model.description,
)
prd_expr = prd_model.query(
    dimensions=("origin",),
    measures=("flight_count", "avg_dep_delay"),
)
catalog.add(
    prd_expr.to_tagged(), aliases=("origin-delays-prd",), sync=False, exist_ok=True
)

print("\nProd flights by origin:")
print(prd_expr.execute())

# ---------------------------------------------------------------------------
# 7. Final catalog state
# ---------------------------------------------------------------------------

print("\nFinal catalog entries:", catalog.list())
print("Final catalog aliases:", catalog.list_aliases())


if __name__ == "__pytest_main__":
    pytest_examples_passed = True
