"""Code samples extracted from docs/tutorials/core_tutorials/build_a_semantic_catalog.qmd.

The tutorial walks readers through several partial blocks and finishes with a
"Putting it all together" full script. We execute the partial blocks in order
(skipping the deliberate-error demonstration) so the snippets file mirrors the
reader's experience top-to-bottom; the final unified script in the tutorial is
the same code in one piece, so we don't repeat it here.
"""

# %% --- Create the flights dataset
import xorq.api as xo

flights = xo.memtable(
    {
        "origin": ["JFK", "LAX", "ORD", "JFK", "LAX", "ORD", "JFK", "LAX"],
        "destination": ["LAX", "ORD", "JFK", "ORD", "JFK", "LAX", "LAX", "JFK"],
        "carrier": ["AA", "UA", "AA", "UA", "AA", "UA", "AA", "UA"],
        "dep_delay": [10.0, -5.0, 30.0, 15.0, -2.0, 45.0, 5.0, 20.0],
        "distance": [2475, 1745, 740, 1300, 2475, 1745, 2475, 2475],
    },
    name="flights",
)

# %% --- Define the semantic model
from boring_semantic_layer import to_semantic_table

flights_model = (
    to_semantic_table(flights)
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

print("Dimensions:", tuple(flights_model.dimensions))
print("Measures:  ", tuple(flights_model.measures))

# %% --- Query: by origin
by_origin = flights_model.query(
    dimensions=("origin",),
    measures=("flight_count", "avg_dep_delay"),
)

print(by_origin.execute())

# %% --- Callout: equivalent plain-xorq query
by_origin_plain = flights.group_by("origin").agg(
    flight_count=flights.count(),
    avg_dep_delay=flights.dep_delay.mean(),
)
print(by_origin_plain.execute())

# %% --- Query: by carrier
by_carrier = flights_model.query(
    dimensions=("carrier",),
    measures=("flight_count", "total_distance"),
)

print(by_carrier.execute())

# %% --- Demonstrate the error path (deliberate failure)
# The tutorial shows that an unknown dimension raises XorqTypeError. We
# wrap the call so the script still finishes — the assertion confirms
# the model rejects the typo before touching data.
try:
    flights_model.query(dimensions=("airport",), measures=("flight_count",)).execute()
except Exception as exc:
    assert "airport" in str(exc), exc
    print(f"Got expected error: {type(exc).__name__}")
else:
    raise AssertionError("expected an error for unknown dimension 'airport'")

# %% --- Catalog the model
import tempfile
from pathlib import Path

from boring_semantic_layer import to_tagged
from xorq.catalog.catalog import Catalog

flights_model_expr = to_tagged(flights_model)

catalog_dir = Path(tempfile.mkdtemp()) / "flights-catalog"
catalog = Catalog.from_repo_path(catalog_dir, init=True)

catalog.add(flights_model_expr, aliases=("flights-model",), sync=False)

print("Catalog at:", catalog_dir)
print("Aliases:   ", catalog.list_aliases())

# %% --- Recover the model and run a new query
from boring_semantic_layer import from_tagged

flights_entry = catalog.get_catalog_entry("flights-model", maybe_alias=True)

flights_model = from_tagged(flights_entry.expr)
print("Recovered type:    ", type(flights_model).__name__)
print("Recovered dims:    ", tuple(flights_model.dimensions))
print("Recovered measures:", tuple(flights_model.measures))

by_destination = flights_model.query(
    dimensions=("destination",),
    measures=("flight_count", "total_distance"),
)

print(by_destination.execute())
