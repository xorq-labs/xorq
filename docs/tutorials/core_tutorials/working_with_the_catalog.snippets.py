"""Code samples adapted from docs/tutorials/core_tutorials/working_with_the_catalog.qmd.

The tutorial drives the publish / clone / pull loop from the ``xorq`` CLI
(`xorq uv build`, `xorq uv run`, `xorq catalog ...`) rather than the Python
API. This smoke test mirrors that flow: each shell snippet in the tutorial
becomes a ``subprocess.run([...])`` call here. Python only re-appears where
the tutorial uses it — recovering the BSL ``SemanticModel`` via
``from_tagged`` and rebinding the entry to a SQLite profile.

Without a real GitHub remote we stand in a local *bare* git repository to
play the role of the GitHub-hosted origin: User A pushes to the bare repo,
User B clones from it, User B pushes a feature branch back, and "merging
the PR" becomes fast-forwarding ``main`` on the bare repo. End state is
identical to a real merge — User A pulls and sees the new alias.
"""

# %% --- test fixture: stand in for `uv init && uv add "xorq[bsl,duckdb,sqlite]"`
import os as _os
import shutil as _shutil
import subprocess as _subprocess
import sys as _sys
import tempfile as _tempfile
from pathlib import Path as _Path


_workdir = _Path(_tempfile.mkdtemp(prefix="xorq-catalog-tutorial-"))

# Two project dirs, mirroring `~/flights-tutorial` (User A) and
# `~/flights-tutorial-userb` (User B). Each gets its own pyproject.toml so
# `xorq uv build` can resolve the project root from the script path.
_USER_A_PROJ = _workdir / "flights-tutorial"
_USER_B_PROJ = _workdir / "flights-tutorial-userb"
for _proj in (_USER_A_PROJ, _USER_B_PROJ):
    _proj.mkdir(parents=True, exist_ok=True)
    (_proj / "pyproject.toml").write_text(
        '[project]\nname = "flights-tutorial"\nversion = "0.0.0"\n'
        'dependencies = ["xorq[bsl,duckdb,sqlite]"]\n'
        "[tool.setuptools]\npackages = []\n"
    )
    # `xorq uv build` needs a requirements.txt or uv.lock to pin the project's
    # deps inside the produced wheel. The tutorial reader gets one via
    # `uv add ...`; the smoke test ships a hand-written requirements.txt for
    # the same effect without touching the network.
    (_proj / "requirements.txt").write_text("xorq[bsl,duckdb,sqlite]\n")

_BARE = _workdir / "remote.git"
_subprocess.run(
    ["git", "init", "--bare", "-b", "main", _BARE],
    check=True,
    capture_output=True,
)

_CATALOG_A = _workdir / "user_a" / "flights-catalog"
_CATALOG_B = _workdir / "user_b" / "flights-catalog"
_CATALOG_A.parent.mkdir(parents=True, exist_ok=True)


def _run(cmd, cwd=None, env=None):
    """Run a CLI command, fail loudly on non-zero, return stdout."""
    result = _subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise AssertionError(
            f"command failed ({result.returncode}): {cmd}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    return result.stdout


# Path to the `xorq` CLI installed alongside the running interpreter.
_XORQ = str(_Path(_sys.executable).parent / "xorq")
assert _Path(_XORQ).exists(), f"xorq CLI not found at {_XORQ}"


_FLIGHTS_MODEL_PY = """\
from boring_semantic_layer import to_semantic_table, to_tagged
import xorq.api as xo

flights = xo.memtable(
    {
        "origin":      ["JFK", "LAX", "ORD", "JFK", "LAX", "ORD", "JFK", "LAX"],
        "destination": ["LAX", "ORD", "JFK", "ORD", "JFK", "LAX", "LAX", "JFK"],
        "carrier":     ["AA",  "UA",  "AA",  "UA",  "AA",  "UA",  "AA",  "UA"],
        "dep_delay":   [10.0, -5.0,  30.0,  15.0, -2.0,  45.0,   5.0,  20.0],
        "distance":    [2475, 1745,   740,  1300, 2475,  1745,  2475,  2475],
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
expr = to_tagged(flights_model)
"""

_AA_FLIGHTS_MODEL_PY = """\
from boring_semantic_layer import to_semantic_table, to_tagged
import xorq.api as xo

flights = xo.memtable(
    {
        "origin":      ["JFK", "LAX", "ORD", "JFK", "LAX", "ORD", "JFK", "LAX"],
        "destination": ["LAX", "ORD", "JFK", "ORD", "JFK", "LAX", "LAX", "JFK"],
        "carrier":     ["AA",  "UA",  "AA",  "UA",  "AA",  "UA",  "AA",  "UA"],
        "dep_delay":   [10.0, -5.0,  30.0,  15.0, -2.0,  45.0,   5.0,  20.0],
        "distance":    [2475, 1745,   740,  1300, 2475,  1745,  2475,  2475],
    },
    name="flights",
)
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
expr = to_tagged(aa_model)
"""
# --- end fixture ---


def _latest_build(builds_dir):
    builds = sorted(
        (p for p in builds_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    assert builds, f"no build dirs in {builds_dir}"
    return builds[0]


try:
    # %% --- Define and build the model (User A)
    (_USER_A_PROJ / "flights_model.py").write_text(_FLIGHTS_MODEL_PY)
    _run(
        [
            _XORQ,
            "uv",
            "build",
            "flights_model.py",
            "-e",
            "expr",
            "--builds-dir",
            "builds",
        ],
        cwd=_USER_A_PROJ,
    )
    _build_a = _latest_build(_USER_A_PROJ / "builds")

    # %% --- Initialize catalog and add the entry (User A)
    _run([_XORQ, "catalog", "--path", str(_CATALOG_A), "init"])
    _run(
        [
            _XORQ,
            "catalog",
            "--path",
            str(_CATALOG_A),
            "add",
            str(_build_a),
            "-a",
            "flights-model",
        ]
    )
    _aliases_a = (
        _run([_XORQ, "catalog", "--path", str(_CATALOG_A), "list-aliases"])
        .strip()
        .splitlines()
    )
    assert _aliases_a == ["flights-model"], _aliases_a

    # %% --- Wire up the (bare) remote and push (User A)
    _run(
        [_XORQ, "catalog", "--path", str(_CATALOG_A), "set-remote", str(_BARE)],
    )
    _run(["git", "push", "-u", "origin", "main"], cwd=_CATALOG_A)
    _run([_XORQ, "catalog", "--path", str(_CATALOG_A), "push"])

    # %% --- Clone the catalog (User B)
    _run(
        [
            _XORQ,
            "catalog",
            "clone",
            str(_BARE),
            "--path",
            str(_CATALOG_B),
        ]
    )
    _aliases_b = (
        _run([_XORQ, "catalog", "--path", str(_CATALOG_B), "list-aliases"])
        .strip()
        .splitlines()
    )
    assert _aliases_b == ["flights-model"], _aliases_b

    # %% --- Recover and query the model (User B) — Python-only step
    from boring_semantic_layer import from_tagged
    from xorq.catalog.catalog import Catalog

    catalog_b = Catalog.from_repo_path(_CATALOG_B, init=False)
    flights_entry = catalog_b.get_catalog_entry("flights-model", maybe_alias=True)
    flights_model_b = from_tagged(flights_entry.expr)
    assert type(flights_model_b).__name__ == "SemanticModel"
    assert tuple(flights_model_b.dimensions) == ("origin", "destination", "carrier")

    by_origin = flights_model_b.query(
        dimensions=("origin",),
        measures=("flight_count", "avg_dep_delay"),
    ).order_by("origin")
    print(by_origin.execute())

    # %% --- Propose a change (User B): branch, build, add with --no-sync
    _run(["git", "checkout", "-b", "add-aa-only-model"], cwd=_CATALOG_B)
    (_USER_B_PROJ / "aa_flights_model.py").write_text(_AA_FLIGHTS_MODEL_PY)
    _run(
        [
            _XORQ,
            "uv",
            "build",
            "aa_flights_model.py",
            "-e",
            "expr",
            "--builds-dir",
            "builds",
        ],
        cwd=_USER_B_PROJ,
    )
    _build_b = _latest_build(_USER_B_PROJ / "builds")
    _run(
        [
            _XORQ,
            "catalog",
            "--path",
            str(_CATALOG_B),
            "add",
            str(_build_b),
            "-a",
            "flights-aa-only",
            "--no-sync",
        ]
    )
    _aliases_b_after = set(
        _run([_XORQ, "catalog", "--path", str(_CATALOG_B), "list-aliases"])
        .strip()
        .splitlines()
    )
    assert _aliases_b_after == {"flights-model", "flights-aa-only"}, _aliases_b_after

    _run(["git", "push", "-u", "origin", "add-aa-only-model"], cwd=_CATALOG_B)

    # %% --- "Merge the PR" — tutorial prose: User A clicks Merge on GitHub.
    # On the bare repo, equivalent to fast-forwarding main to the feature branch.
    _run(
        ["git", "update-ref", "refs/heads/main", "refs/heads/add-aa-only-model"],
        cwd=_BARE,
    )

    # %% --- Pull the merged change (User A)
    _run([_XORQ, "catalog", "--path", str(_CATALOG_A), "pull"])
    _aliases_a_after = set(
        _run([_XORQ, "catalog", "--path", str(_CATALOG_A), "list-aliases"])
        .strip()
        .splitlines()
    )
    assert _aliases_a_after == {"flights-model", "flights-aa-only"}, _aliases_a_after

    # %% --- Swap the profile at recovery time — Python-only step
    import xorq.api as xo
    from xorq.vendor.ibis.backends.profiles import Profile

    sqlite_con = xo.sqlite.connect()
    Profile.from_con(sqlite_con).save(alias="local_dev_sqlite", clobber=True)

    catalog_a = Catalog.from_repo_path(_CATALOG_A, init=False)
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
