"""Code samples extracted from docs/tutorials/core_tutorials/build_a_semantic_catalog.qmd.

The tutorial walks readers through several partial blocks and finishes with a
"Putting it all together" full script. We execute the partial blocks in order
so the snippets file mirrors the reader's experience top-to-bottom; the final
unified script in the tutorial is the same code in one piece, so we don't
repeat it here.

The tutorial tells readers to ``uv init`` and ``uv add "xorq[bsl,duckdb]"`` in
a project directory before running the script. We mimic that here by chdir'ing
into a fresh tempdir with a pyproject.toml + requirements.txt that name xorq
as a dep — that's the same shape the wheel packager sees in the reader's
project, just hand-written instead of resolved by ``uv lock``. Skipping the
real lockfile keeps the smoke test offline; if the wheel-build path ever
starts to depend on a fully-resolved transitive lock, this fixture should be
upgraded to run ``uv lock`` for real.
"""

# %% --- test fixture: stand in for `uv init && uv add "xorq[bsl,duckdb]"`
import os as _os
import shutil as _shutil
import tempfile as _tempfile
from pathlib import Path as _Path

_workdir = _Path(_tempfile.mkdtemp(prefix="xorq-build-tutorial-"))
(_workdir / "pyproject.toml").write_text(
    '[project]\nname = "flights-tutorial"\nversion = "0.0.0"\n'
    'dependencies = ["xorq[bsl,duckdb]"]\n'
    "[tool.setuptools]\npackages = []\n"
)
(_workdir / "requirements.txt").write_text("xorq[bsl,duckdb]\n")
_os.chdir(_workdir)
# --- end fixture ---

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
).order_by("origin")

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
).order_by("carrier")

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
from pathlib import Path

from boring_semantic_layer import to_tagged
from xorq.catalog.catalog import Catalog

flights_model_expr = to_tagged(flights_model)

# Stable path inside the project — same shape as the prose, so a separate
# script (recover_flights.py in the tutorial) can reopen by path below.
catalog_dir = Path("flights-catalog")
catalog = Catalog.from_repo_path(catalog_dir, init=True)

catalog.add(flights_model_expr, aliases=("flights-model",), sync=False)

print("Catalog at:", catalog_dir.absolute())
print("Aliases:   ", catalog.list_aliases())

# %% --- Recover the model from a fresh Catalog handle (mirrors recover_flights.py)
# In the prose this is a separate Python file with no in-memory state from the
# publisher.  Here we simulate that by re-opening the catalog from its path
# instead of reusing the `catalog` object above; init=False just opens.
from boring_semantic_layer import from_tagged

recover_catalog = Catalog.from_repo_path(catalog_dir, init=False)

flights_entry = recover_catalog.get_catalog_entry("flights-model", maybe_alias=True)

flights_model = from_tagged(flights_entry.expr)
print("Recovered type:    ", type(flights_model).__name__)
print("Recovered dims:    ", tuple(flights_model.dimensions))
print("Recovered measures:", tuple(flights_model.measures))

by_destination = flights_model.query(
    dimensions=("destination",),
    measures=("flight_count", "total_distance"),
).order_by("destination")

print(by_destination.execute())

# %% --- cleanup (fixture)
_os.chdir(_workdir.parent)
_shutil.rmtree(_workdir, ignore_errors=True)
