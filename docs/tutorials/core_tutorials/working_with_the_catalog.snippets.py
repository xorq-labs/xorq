"""Code samples adapted from docs/tutorials/core_tutorials/working_with_the_catalog.qmd.

The tutorial walks through pushing a catalog to GitHub and a teammate cloning
it back over the network. Without a real remote (and without an S3/GCS special
remote for git-annex content storage), the smoke test substitutes a
peer-to-peer clone — User B clones directly from User A's local working tree.
The API surface exercised is identical: ``Catalog.clone_from``,
``catalog.fetch_entries``, ``catalog.get_catalog_entry``, ``from_tagged``,
``catalog.add``, ``catalog.load(con=...)``, and the Profile API. What's
*not* exercised here: pushing through a bare GitHub remote and the
S3/GCS-backed annex content workflow — those are covered in the prose and
require credentials this script doesn't have.

Like build_a_semantic_catalog.snippets.py we chdir into a fresh tempdir with
a minimal pyproject.toml so the snippet works regardless of where the test
launched it from — that mirrors the ``uv init`` step the tutorials prescribe.
"""

# --- test fixtures: stand-in paths for User A and User B ---
import os as _os
import shutil as _shutil
import stat as _stat
import tempfile as _tempfile
from pathlib import Path as _Path


_workdir = _Path(_tempfile.mkdtemp(prefix="xorq-catalog-tutorial-"))
(_workdir / "pyproject.toml").write_text(
    '[project]\nname = "flights-tutorial"\nversion = "0.0.0"\n'
    "[tool.setuptools]\npackages = []\n"
)
# Wheel packager needs requirements.txt or uv.lock; readers get one from
# `uv add`, snippet shortcuts with an empty file (smoke-test scope only).
(_workdir / "requirements.txt").write_text("")
_os.chdir(_workdir)

_USER_A = _workdir / "userA" / "flights-catalog"
_USER_B = _workdir / "userB" / "flights-catalog"
_USER_A.parent.mkdir(parents=True, exist_ok=True)
_USER_B.parent.mkdir(parents=True, exist_ok=True)
# --- end fixtures ---


def _force_remove(path):
    """git-annex objects are read-only; chmod the tree before rmtree."""
    for p in _Path(path).rglob("*"):
        try:
            p.chmod(_stat.S_IRWXU | _stat.S_IRWXG | _stat.S_IRWXO)
        except (FileNotFoundError, PermissionError):
            pass
    _shutil.rmtree(path, ignore_errors=True)


try:
    # %% --- Publish the catalog (User A)
    import xorq.api as xo
    from boring_semantic_layer import to_semantic_table, to_tagged
    from xorq.catalog.catalog import Catalog
    from xorq.catalog.annex import LOCAL_ANNEX

    catalog = Catalog.from_repo_path(
        _USER_A,  # tutorial: Path("~/work/flights-catalog").expanduser()
        init=True,
        annex=LOCAL_ANNEX,
    )

    BASE_URL = "https://pub-a45a6a332b4646f2a6f44775695c64df.r2.dev"
    flights = xo.deferred_read_parquet(
        f"{BASE_URL}/flights.parquet",
        table_name="flights",
    )
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
        )
    )
    flights_model_expr = to_tagged(flights_model)
    catalog.add(flights_model_expr, aliases=("flights-model",), sync=False)

    # Tutorial: `git remote add origin <github>` + `git push -u origin main`
    # + `catalog.push()`. Skipped here — see fixtures comment.

    # %% --- Clone the catalog (User B)
    # Tutorial: clone from https://github.com/.../flights-catalog.git.
    # Smoke test: clone from User A's local working tree so git-annex can
    # fetch content peer-to-peer without a configured special remote.
    catalog = Catalog.clone_from(str(_USER_A), _USER_B)

    print("Aliases:", catalog.list_aliases())
    assert catalog.list_aliases() == ["flights-model"]

    # %% --- Recover and query the model (User B)
    from boring_semantic_layer import from_tagged

    # Until xorq#1891 lands, neither fetch_entries(alias) nor
    # get_catalog_entry(alias, maybe_alias=True) works against a freshly
    # cloned catalog; alias resolution requires content-local. So look up
    # the entry name from catalog.list() first, fetch, then alias works.
    entry_name = catalog.list()[0]
    catalog.fetch_entries(entry_name)

    flights_entry = catalog.get_catalog_entry("flights-model", maybe_alias=True)
    flights_model = from_tagged(flights_entry.expr)
    assert type(flights_model).__name__ == "SemanticModel"
    assert tuple(flights_model.dimensions) == ("origin", "destination", "carrier")

    # Tutorial executes a query against the live R2 URL; the smoke test
    # asserts the recovered model has the right shape rather than fetching
    # ~12 MB over the network on every CI run.

    # %% --- Propose a change (User B)
    # Tutorial: branch, catalog.add, git push, gh pr create. The catalog.add
    # is the load-bearing API call; the git plumbing is standard.
    aa_flights = flights.filter(flights.carrier == "AA")
    aa_model = (
        to_semantic_table(aa_flights)
        .with_dimensions(
            origin=lambda t: t.origin,
            destination=lambda t: t.destination,
        )
        .with_measures(
            flight_count=lambda t: t.count(),
            avg_dep_delay=lambda t: t.dep_delay.mean(),
        )
    )

    aa_model_expr = to_tagged(aa_model)
    catalog.add(aa_model_expr, aliases=("flights-aa-only",), sync=False)
    assert set(catalog.list_aliases()) == {"flights-model", "flights-aa-only"}

    # %% --- Pull the merged change (User A)
    # Tutorial: User A pulls from origin after the PR merges (catalog.pull()).
    # Without a real bidirectional remote, the smoke test reopens User A's
    # catalog to confirm catalog reopen + the API surface still work.
    catalog_a = Catalog.from_repo_path(_USER_A, init=False)
    print("User A aliases:", catalog_a.list_aliases())
    assert catalog_a.list_aliases() == ["flights-model"]

    # %% --- Swap the profile at recovery time
    from xorq.vendor.ibis.backends.profiles import Profile

    local_con = xo.connect()
    Profile.from_con(local_con).save(alias="local_dev", clobber=True)

    profile = Profile.load("local_dev")
    expr = catalog.load("flights-model", con=profile.get_con())

    assert expr.ls.kind == "composed"
finally:
    _os.chdir(_workdir.parent)
    _force_remove(_workdir)
