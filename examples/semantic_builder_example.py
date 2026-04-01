"""Demonstrates the SemanticModelSpec builder: catalog a semantic model, build expressions, recover and rebind.

Traditional approach: You define a semantic model with dimensions and measures, then manually
track which selections produced which expression. When the underlying data source changes
(e.g., dev -> prod), you rebuild everything from scratch and lose the connection between the
model definition and the expressions it produced.

With xorq: SemanticModelSpec wraps a BSL SemanticModel as a catalog builder entry. You can
build multiple expressions from different dimension/measure selections, recover the original
model from any cataloged expression, rebind to new data, and catalog the result — all while
maintaining full provenance.
"""

import tempfile
from pathlib import Path

import xorq.api as xo
from boring_semantic_layer import to_semantic_table
from xorq.catalog.catalog import Catalog
from xorq.expr.builders.semantic_model import SemanticModelSpec
from xorq.ibis_yaml.packager import Sdister


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
# 2. Create a SemanticModel and wrap it as a BuilderSpec
# ---------------------------------------------------------------------------

model = to_semantic_table(dev_data).with_dimensions(
    origin=lambda t: t.origin,
    destination=lambda t: t.destination,
    carrier=lambda t: t.carrier,
).with_measures(
    flight_count=lambda t: t.count(),
    avg_dep_delay=lambda t: t.dep_delay.mean(),
    total_distance=lambda t: t.distance.sum(),
)

spec = SemanticModelSpec(model=model)
print("Available dimensions:", spec.available_dimensions)
print("Available measures:", spec.available_measures)

# ---------------------------------------------------------------------------
# 3. Add the builder to a catalog
# ---------------------------------------------------------------------------

catalog_dir = Path(tempfile.mkdtemp()) / "example-catalog"
catalog = Catalog.from_repo_path(catalog_dir, init=True)
print(f"\nCatalog directory: {catalog_dir}")

sdister = Sdister.from_script_path(__file__)
catalog.add_builder(spec, sdister.sdist_path, aliases=("flights-model-dev",), sync=False)
print("\nCatalog entries:", catalog.list())
print("Catalog aliases:", catalog.list_aliases())

# ---------------------------------------------------------------------------
# 4. Retrieve the builder from the catalog and build expressions
# ---------------------------------------------------------------------------

builder = catalog.get_builder("flights-model-dev")
print("\nRecovered dimensions:", builder.available_dimensions)
print("Recovered measures:", builder.available_measures)

expr_by_origin = builder.build_expr(
    dimensions=("origin",),
    measures=("flight_count", "avg_dep_delay"),
)
print("\nFlights by origin:")
print(expr_by_origin.execute())

expr_by_carrier = builder.build_expr(
    dimensions=("carrier",),
    measures=("total_distance",),
)
print("\nDistance by carrier:")
print(expr_by_carrier.execute())

# ---------------------------------------------------------------------------
# 5. Catalog a built expression (with BSL provenance tags)
# ---------------------------------------------------------------------------

catalog.add(expr_by_origin.to_tagged(), aliases=("origin-delays-dev",), sync=False)
print("\nCatalog after adding expression:", catalog.list())

# ---------------------------------------------------------------------------
# 6. Recover the full builder from the catalog and build a new selection
# ---------------------------------------------------------------------------

recovered_builder = catalog.get_builder("flights-model-dev")
print("\nRecovered builder from catalog:")
print("  dimensions:", recovered_builder.available_dimensions)
print("  measures:", recovered_builder.available_measures)

expr_by_dest = recovered_builder.build_expr(
    dimensions=("destination",),
    measures=("flight_count", "total_distance"),
)
print("\nFlights by destination (from recovered builder):")
print(expr_by_dest.execute())

# Check expression provenance — the cataloged expression records it was built by BSL
origin_entry = catalog.get_catalog_entry("origin-delays-dev", maybe_alias=True)
print("\nExpression metadata builders:", origin_entry.metadata.builders)

# ---------------------------------------------------------------------------
# 7. Rebind to production data — same model shape, different fact table
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

prd_spec = recovered_builder.rebind(prd_data)
prd_expr = prd_spec.build_expr(
    dimensions=("origin",),
    measures=("flight_count", "avg_dep_delay"),
)
catalog.add(prd_expr.to_tagged(), aliases=("origin-delays-prd",), sync=False, exist_ok=True)

print("\nProd flights by origin:")
print(prd_expr.execute())

# ---------------------------------------------------------------------------
# 8. Final catalog state — one builder, multiple expressions
# ---------------------------------------------------------------------------

print("\nFinal catalog entries:", catalog.list())
print("Final catalog aliases:", catalog.list_aliases())


if __name__ == "__pytest_main__":
    pytest_examples_passed = True
