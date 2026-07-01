"""Why resolve_strategy matches the con's MRO instead of lazy_singledispatch.

Demonstrates the single-backend-install failure discussed on the XOR-444 PR.

Run with ONLY duckdb installed (that's the whole point):

    uv run --active python dispatch_demo.py
    # or: nix develop .#virtualenv-editable-313 --command python dispatch_demo.py

Expected output:

    A: dispatching duckdb raised: ModuleNotFoundError("No module named 'xorq.backends.__driver_not_installed__'")
    B: duckdb UPSERT -> native_merge; other backends imported: NONE
"""

import sys

import xorq.api as xo
from xorq.vendor.ibis.common.dispatch import lazy_singledispatch


# --- A: the flaw ------------------------------------------------------------
# lazy_singledispatch buckets lazy registrations by *top-level package* and
# imports the whole bucket on first dispatch (vendor/ibis/common/dispatch.py:65,
# 105). Every xorq backend is `xorq.backends.*`, so they share one "xorq" bucket
# — and routing ANY backend tries to import them ALL. In a single-backend
# install the ones whose driver isn't installed raise ModuleNotFoundError.
@lazy_singledispatch
def strategy(con):
    return "default"


strategy.register("xorq.backends.duckdb.Backend")(lambda con: "native_merge")
# stands in for any backend whose driver isn't installed (e.g. swap for
# "xorq.backends.pyiceberg.Backend" and uninstall pyiceberg for the verbatim
# CI error):
strategy.register("xorq.backends.__driver_not_installed__.Backend")(lambda con: "x")

con = xo.duckdb.connect()
try:
    strategy(con)  # dispatch a DUCKDB con...
    print("A: no error (you must have every backend installed)")
except ModuleNotFoundError as e:
    # ...and it tries to import the *other* registration in the "xorq" bucket.
    print(f"A: dispatching duckdb raised: {e!r}")


# --- B: the fix -------------------------------------------------------------
# resolve_strategy matches the con's own already-loaded __mro__ against a string
# table (BackendType), so it imports nothing — routing one backend never touches
# another's module.
from xorq.writes.enums import PublishMode  # noqa: E402
from xorq.writes.publish import resolve_strategy  # noqa: E402


before = set(sys.modules)
strat = resolve_strategy(con, PublishMode.UPSERT)
newly = {m for m in set(sys.modules) - before if m.startswith("xorq.backends.")}
print(f"B: duckdb UPSERT -> {strat.value}; other backends imported: {newly or 'NONE'}")
