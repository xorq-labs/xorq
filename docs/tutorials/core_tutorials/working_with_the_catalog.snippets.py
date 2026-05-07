"""Code samples adapted from docs/tutorials/core_tutorials/working_with_the_catalog.qmd.

The tutorial walks through pushing a catalog to GitHub and a teammate cloning
it back over the network. Without a real GitHub remote, the smoke test stands
in a local *bare* git repository to play the role of the GitHub-hosted origin:
User A pushes to the bare repo, User B clones from it, User B pushes a feature
branch back to it, and "merging the PR" becomes fast-forwarding ``main`` on
the bare repo. End state is identical to a real merge — User A pulls and sees
the new alias — so the API surface exercised here matches what a reader doing
the live GitHub workflow would hit.

What's still *not* exercised: the GitHub UI itself (branch protection, PR
reviews, merge button). The prose covers those; the smoke test only verifies
the xorq APIs that bracket the human-in-the-loop step.
"""

# %% --- test fixture: stand in for `uv init && uv add "xorq[bsl,duckdb,sqlite]"`
import os as _os
import shutil as _shutil
import subprocess as _subprocess
import tempfile as _tempfile
from pathlib import Path as _Path

_workdir = _Path(_tempfile.mkdtemp(prefix="xorq-catalog-tutorial-"))
(_workdir / "pyproject.toml").write_text(
    '[project]\nname = "flights-tutorial"\nversion = "0.0.0"\n'
    'dependencies = ["xorq[bsl,duckdb,sqlite]"]\n'
    "[tool.setuptools]\npackages = []\n"
)
(_workdir / "requirements.txt").write_text("xorq[bsl,duckdb,sqlite]\n")
_os.chdir(_workdir)

# Stand-in for the GitHub-hosted origin: a bare repo both users push to.
_BARE = _workdir / "remote.git"
_subprocess.run(
    ["git", "init", "--bare", "-b", "main", _BARE],
    check=True,
    capture_output=True,
)

_USER_A = _workdir / "userA" / "flights-catalog"
_USER_B = _workdir / "userB" / "flights-catalog"
_USER_A.parent.mkdir(parents=True, exist_ok=True)
# --- end fixture ---


try:
    # %% --- Publish the catalog (User A)
    import xorq.api as xo
    from boring_semantic_layer import to_semantic_table, to_tagged
    from xorq.catalog.catalog import Catalog

    catalog = Catalog.from_repo_path(_USER_A, init=True)

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
    flights_model_expr = to_tagged(flights_model)
    catalog.add(flights_model_expr, aliases=("flights-model",), sync=False)

    # Tutorial: `gh repo create` + `git remote add origin <url>` + `git push -u origin main`.
    # Smoke test: point origin at the local bare repo and push.
    _subprocess.run(
        ["git", "remote", "add", "origin", _BARE],
        cwd=_USER_A,
        check=True,
        capture_output=True,
    )
    _subprocess.run(
        ["git", "push", "-u", "origin", "main"],
        cwd=_USER_A,
        check=True,
        capture_output=True,
    )

    # %% --- Clone the catalog (User B)
    catalog_b = Catalog.clone_from(_BARE, _USER_B)

    print("Aliases:", catalog_b.list_aliases())
    assert catalog_b.list_aliases() == ["flights-model"]

    # %% --- Recover and query the model (User B)
    from boring_semantic_layer import from_tagged

    flights_entry = catalog_b.get_catalog_entry("flights-model", maybe_alias=True)
    flights_model_b = from_tagged(flights_entry.expr)
    assert type(flights_model_b).__name__ == "SemanticModel"
    assert tuple(flights_model_b.dimensions) == ("origin", "destination", "carrier")

    by_origin = flights_model_b.query(
        dimensions=("origin",),
        measures=("flight_count", "avg_dep_delay"),
    ).order_by("origin")
    print(by_origin.execute())

    # %% --- Propose a change (User B)
    # Tutorial: branch + catalog.add + git push branch + gh pr create.
    _subprocess.run(
        ["git", "checkout", "-b", "add-aa-only-model"],
        cwd=_USER_B,
        check=True,
        capture_output=True,
    )

    # User B reconstructs the same flights memtable they got from the foundation tutorial.
    flights_b = xo.memtable(
        {
            "origin": ["JFK", "LAX", "ORD", "JFK", "LAX", "ORD", "JFK", "LAX"],
            "destination": ["LAX", "ORD", "JFK", "ORD", "JFK", "LAX", "LAX", "JFK"],
            "carrier": ["AA", "UA", "AA", "UA", "AA", "UA", "AA", "UA"],
            "dep_delay": [10.0, -5.0, 30.0, 15.0, -2.0, 45.0, 5.0, 20.0],
            "distance": [2475, 1745, 740, 1300, 2475, 1745, 2475, 2475],
        },
        name="flights",
    )
    aa_flights = flights_b.filter(flights_b.carrier == "AA")
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
    catalog_b.add(aa_model_expr, aliases=("flights-aa-only",), sync=False)
    assert set(catalog_b.list_aliases()) == {"flights-model", "flights-aa-only"}

    _subprocess.run(
        ["git", "push", "-u", "origin", "add-aa-only-model"],
        cwd=_USER_B,
        check=True,
        capture_output=True,
    )

    # %% --- Merge the PR (the prose says: User A clicks Merge on GitHub).
    # Equivalent on the bare repo: fast-forward main to the feature branch.
    _subprocess.run(
        ["git", "update-ref", "refs/heads/main", "refs/heads/add-aa-only-model"],
        cwd=_BARE,
        check=True,
        capture_output=True,
    )

    # %% --- Pull the merged change (User A)
    catalog_a = Catalog.from_repo_path(_USER_A, init=False)
    catalog_a.pull()
    print("User A aliases after pull:", catalog_a.list_aliases())
    assert set(catalog_a.list_aliases()) == {"flights-model", "flights-aa-only"}

    # %% --- Swap the profile at recovery time
    from xorq.vendor.ibis.backends.profiles import Profile

    sqlite_con = xo.sqlite.connect()
    Profile.from_con(sqlite_con).save(alias="local_dev_sqlite", clobber=True)

    profile = Profile.load("local_dev_sqlite")
    expr = catalog_a.load("flights-model", con=profile.get_con())

    result = expr.execute()
    print(result.head())
    # 8 rows of inline flights data, executed against SQLite instead of the
    # default xorq_datafusion backend the entry was built on.
    assert len(result) == 8
finally:
    _os.chdir(_workdir.parent)
    _shutil.rmtree(_workdir, ignore_errors=True)
