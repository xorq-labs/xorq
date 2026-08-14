"""Cross-process determinism of build hashes and cache keys.

Every other hash-stability test in this repo runs inside one interpreter, so
none of them can see a per-process value leaking into identity -- the
most-repeated bug in this subsystem's history (gh-610, gh-1728, gh-1738,
gh-2229: generated names, counters, and process-global state reaching
identity; see ``view_rules`` for the latest). Each was found by hand, in
production, by noticing an unchanged script writing new ``builds/<hash>``
directories. This module automates that observation: build a corpus of
expressions in two *fresh interpreters* with different ``PYTHONHASHSEED`` and
assert the identities agree.

It is deliberately not marked ``slow`` -- CI lanes that filter ``not slow``
are exactly the lanes that need this guard. Cost is two interpreter starts
total, because one child computes the whole corpus.

Unlike the op-type sweep in ``test_view_rules.py``, this catches the leak class
regardless of which op, which mechanism, or which dispatch table is at fault:
the corpus below leaves every generated name *unset* so that ``gen_name()``
uuid4s are live in the tree.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


# Built in the child, one entry per expression shape. Generated names are left
# unset on purpose: passing ``inner_name``/``name`` would pin the very values
# whose leakage this module exists to detect.
CHILD_SCRIPT = """
import json, pathlib, sys, traceback

import toolz

import xorq.api as xo
from xorq.caching import ParquetSnapshotCache
from xorq.caching.strategy import SnapshotStrategy
from xorq.common.utils.func_utils import return_constant
from xorq.common.utils.provenance_utils import get_expr_hash

echo_udxf = xo.expr.relations.flight_udxf(
    process_df=toolz.identity,
    maybe_schema_in=return_constant(True),
    maybe_schema_out=toolz.identity,
)


def diamonds_path():
    return pathlib.Path(xo.options.pins.get_path("diamonds"))


def build_flight_udxf():
    # no inner_name -> FlightUDXF.name is a fresh gen_name() uuid4 (gh-2229)
    con = xo.connect()
    return con.read_parquet(diamonds_path(), "diamonds").pipe(echo_udxf)


def build_flight_expr():
    con = xo.connect()
    t = con.read_parquet(diamonds_path(), "diamonds")
    return xo.expr.relations.flight_expr(t, xo.table(t.schema(), name="unbound"))


def build_remote_table():
    con, other_con = xo.connect(), xo.connect()
    return con.read_parquet(diamonds_path(), "diamonds").into_backend(other_con)


def build_cached_node():
    con = xo.connect()
    return con.read_parquet(diamonds_path(), "diamonds").cache(
        ParquetSnapshotCache.from_kwargs(source=con)
    )


def build_deferred_read():
    return xo.deferred_read_parquet(diamonds_path(), xo.connect())


def build_flight_udxf_cached():
    con = xo.connect()
    return (
        con.read_parquet(diamonds_path(), "diamonds")
        .pipe(echo_udxf)
        .cache(ParquetSnapshotCache.from_kwargs(source=con))
    )


BUILDERS = {
    "flight_udxf": build_flight_udxf,
    "flight_expr": build_flight_expr,
    "remote_table": build_remote_table,
    "cached_node": build_cached_node,
    "deferred_read": build_deferred_read,
    "flight_udxf_cached": build_flight_udxf_cached,
}

out = {}
for name, build in BUILDERS.items():
    try:
        expr = build()
        out[name] = {
            "build_hash": get_expr_hash(expr),
            "snapshot_key": SnapshotStrategy().calc_key(expr),
            "tokenized": expr.ls.tokenized,
        }
    except Exception:
        out[name] = {"error": traceback.format_exc(limit=3)}

json.dump(out, sys.stdout)
"""

CASES = (
    "flight_udxf",
    "flight_expr",
    "remote_table",
    "cached_node",
    "deferred_read",
    "flight_udxf_cached",
)

IDENTITIES = ("build_hash", "snapshot_key", "tokenized")


def _run_child() -> dict:
    """Run the corpus in a fresh interpreter, inheriting the current env."""
    proc = subprocess.run(
        [sys.executable, "-c", CHILD_SCRIPT],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"child interpreter failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
    )
    return json.loads(proc.stdout)


@pytest.fixture(scope="module")
def child_identities() -> tuple[dict, dict]:
    """Identities from two fresh interpreters with different hash seeds.

    Module-scoped: two interpreter starts for the whole corpus, not per case.
    Differing ``PYTHONHASHSEED`` also makes this sensitive to identity that
    depends on str/bytes hash randomization or dict iteration order, not only
    to uuid4s.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PYTHONHASHSEED", "0")
        first = _run_child()
        mp.setenv("PYTHONHASHSEED", "123456")
        second = _run_child()
    return first, second


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("identity", IDENTITIES)
def test_identity_is_stable_across_processes(
    case: str, identity: str, child_identities: tuple[dict, dict]
) -> None:
    first, second = child_identities
    for run in (first, second):
        assert "error" not in run[case], (
            f"child failed to build {case!r}:\n{run[case]['error']}"
        )
    assert first[case][identity] == second[case][identity], (
        f"{identity} for {case!r} differs between two fresh interpreters "
        f"({first[case][identity]!r} vs {second[case][identity]!r}). Something "
        f"per-process -- a gen_name() uuid4, a counter, an id(), a "
        f"hash-randomized ordering -- is reaching identity. See gh-2229."
    )


def test_corpus_matches_child(child_identities: tuple[dict, dict]) -> None:
    """The parametrized case list covers everything the child builds."""
    first, _ = child_identities
    assert set(first) == set(CASES), (
        "CHILD_SCRIPT builders and CASES disagree; a case built in the child "
        "but missing from CASES is silently unasserted"
    )
