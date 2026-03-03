"""Tests for load_expr(..., lazy=True / limit=N) options."""

import time
from pathlib import Path

import pandas as pd
import pytest

import xorq.api as xo
from xorq.backends.lazy import LazyBackend
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.common.utils.graph_utils import find_all_sources
from xorq.ibis_yaml.compiler import build_expr, load_expr
from xorq.tests.util import assert_frame_equal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_sources_lazy(expr):
    return all(isinstance(src, LazyBackend) for src in find_all_sources(expr))


def _all_sources_unconnected(expr):
    return all(not src.is_connected for src in find_all_sources(expr))


# ---------------------------------------------------------------------------
# hydrate_cons / Profile.get_lazy_con
# ---------------------------------------------------------------------------


def test_hydrate_cons_lazy_returns_lazy_backends(builds_dir, parquet_dir):
    """hydrate_cons(lazy=True) must return LazyBackend sources, not live connections."""
    from xorq.ibis_yaml.compiler import ArtifactStore, DumpFiles, hydrate_cons

    backend = xo.duckdb.connect()
    parquet_path = parquet_dir / "awards_players.parquet"
    expr = deferred_read_parquet(parquet_path, backend, table_name="awards_players")
    build_path = build_expr(expr, builds_dir=builds_dir)

    artifact_store = ArtifactStore(build_path)
    raw_profiles = artifact_store.load_yaml(DumpFiles.profiles)

    lazy_cons = hydrate_cons(raw_profiles, lazy=True)
    eager_cons = hydrate_cons(raw_profiles, lazy=False)

    assert all(isinstance(con, LazyBackend) for con in lazy_cons.values())
    assert all(not isinstance(con, LazyBackend) for con in eager_cons.values())


# ---------------------------------------------------------------------------
# load_expr(lazy=True) — duckdb
# ---------------------------------------------------------------------------


def test_load_expr_lazy_sources_unconnected_before_execute(builds_dir, parquet_dir):
    """Sources must be LazyBackend and unconnected until execute() is called."""
    backend = xo.duckdb.connect()
    parquet_path = parquet_dir / "awards_players.parquet"
    expr = deferred_read_parquet(parquet_path, backend, table_name="awards_players")

    build_path = build_expr(expr, builds_dir=builds_dir)
    lazy_expr = load_expr(build_path, lazy=True)

    assert _all_sources_lazy(lazy_expr), "Expected all sources to be LazyBackend"
    assert _all_sources_unconnected(lazy_expr), "Expected all sources to be unconnected"


def test_load_expr_lazy_connects_on_execute(builds_dir, parquet_dir):
    """After execute(), all LazyBackend sources must report is_connected=True."""
    backend = xo.duckdb.connect()
    parquet_path = parquet_dir / "awards_players.parquet"
    expr = deferred_read_parquet(parquet_path, backend, table_name="awards_players")

    build_path = build_expr(expr, builds_dir=builds_dir)
    lazy_expr = load_expr(build_path, lazy=True)

    lazy_expr.execute()

    assert all(src.is_connected for src in find_all_sources(lazy_expr))


def test_load_expr_lazy_result_matches_eager(builds_dir, parquet_dir):
    """Lazy and eager load_expr must produce identical results."""
    backend = xo.duckdb.connect()
    parquet_path = parquet_dir / "awards_players.parquet"
    expr = deferred_read_parquet(parquet_path, backend, table_name="awards_players")
    expr = expr.filter(expr.lgID == "NL").drop("yearID", "lgID")

    build_path = build_expr(expr, builds_dir=builds_dir)

    eager_result = load_expr(build_path, lazy=False).execute()
    lazy_result = load_expr(build_path, lazy=True).execute()

    assert len(eager_result) > 0
    assert_frame_equal(eager_result, lazy_result)


def test_load_expr_lazy_datafusion(builds_dir, parquet_dir):
    """Lazy load works with the DataFusion backend too."""
    backend = xo.datafusion.connect()
    parquet_path = parquet_dir / "awards_players.parquet"
    expr = deferred_read_parquet(parquet_path, backend, table_name="awards_players")

    build_path = build_expr(expr, builds_dir=builds_dir)
    lazy_expr = load_expr(build_path, lazy=True)

    assert _all_sources_lazy(lazy_expr)
    assert _all_sources_unconnected(lazy_expr)

    result = lazy_expr.execute()
    assert len(result) > 0
    assert all(src.is_connected for src in find_all_sources(lazy_expr))


def test_load_expr_default_is_not_lazy(builds_dir, parquet_dir):
    """load_expr without lazy= must behave as before (eager connections)."""
    backend = xo.duckdb.connect()
    parquet_path = parquet_dir / "awards_players.parquet"
    expr = deferred_read_parquet(parquet_path, backend, table_name="awards_players")

    build_path = build_expr(expr, builds_dir=builds_dir)
    loaded = load_expr(build_path)

    assert not any(isinstance(src, LazyBackend) for src in find_all_sources(loaded))


def _memtable_build_path(builds_dir, n_rows=50):
    """Build an expr from an in-memory DataFrame so deferred_reads_to_memtables fires."""
    df = pd.DataFrame({"x": range(n_rows), "y": range(n_rows)})
    expr = xo.memtable(df)
    return build_expr(expr, builds_dir=builds_dir), n_rows


def test_load_expr_limit_caps_rows(builds_dir):
    """limit=N must return at most N rows from the in-memory parquet read."""
    build_path, n_rows = _memtable_build_path(builds_dir)

    result = load_expr(build_path, limit=10).execute()
    assert len(result) == 10


def test_load_expr_limit_none_returns_all_rows(builds_dir):
    """limit=None (default) must return all rows."""
    build_path, n_rows = _memtable_build_path(builds_dir)

    result = load_expr(build_path, limit=None).execute()
    assert len(result) == n_rows


def test_load_expr_limit_larger_than_dataset_returns_all(builds_dir):
    """limit larger than the dataset must return all rows, not raise."""
    build_path, n_rows = _memtable_build_path(builds_dir)

    result = load_expr(build_path, limit=n_rows * 10).execute()
    assert len(result) == n_rows


def test_load_expr_limit_is_pushed_into_parquet_read(builds_dir, parquet_dir):

    from xorq.ibis_yaml.compiler import (
        ArtifactStore,
        DumpFiles,
        ExprLoader,
        YamlExpressionTranslator,
        hydrate_cons,
    )

    df = pd.read_parquet(parquet_dir / "batting.parquet")
    n_rows = len(df)  # 101_332
    build_path = build_expr(xo.memtable(df), builds_dir=builds_dir)

    # Do YAML loading once — fixed overhead, not part of the timed section.
    artifact_store = ArtifactStore(build_path)
    profiles = hydrate_cons(artifact_store.load_yaml(DumpFiles.profiles))
    loaded = YamlExpressionTranslator.from_yaml(
        artifact_store.load_yaml(DumpFiles.expr), profiles=profiles
    )

    # Warmup: stabilize OS page cache and DuckDB internals before timing.
    ExprLoader.deferred_reads_to_memtables(loaded, build_path, limit=1)

    limits = [10, 10_000, n_rows]
    times = []
    for limit in limits:
        t0 = time.perf_counter()
        expr = ExprLoader.deferred_reads_to_memtables(loaded, build_path, limit=limit)
        times.append(time.perf_counter() - t0)
        assert len(expr.execute()) == limit

    small_time = times[0]
    full_time = times[-1]
    assert full_time > small_time, (
        f"Expected full-dataset read ({full_time:.4f}s) to be slower than "
        f"limit={limits[0]} read ({small_time:.4f}s).\n"
        + "\n".join(f"  limit={lim:>7}: {t:.4f}s" for lim, t in zip(limits, times))
    )


@pytest.fixture
def lahman_parquet_dir(tmp_path_factory, n_rows: int = 1_000) -> Path:
    """Write synthetic Lahman-style parquet files and return the directory."""
    parquet_dir = tmp_path_factory.mktemp("lahman_parquet")
    player_ids = [f"player{i:04d}" for i in range(n_rows)]
    years = [1990 + (i % 30) for i in range(n_rows)]
    teams = [f"T{i % 20:02d}" for i in range(n_rows)]

    pd.DataFrame(
        {
            "playerID": player_ids,
            "nameFirst": [f"First{i}" for i in range(n_rows)],
            "nameLast": [f"Last{i}" for i in range(n_rows)],
        }
    ).to_parquet(parquet_dir / "people.parquet", index=False)

    pd.DataFrame(
        {
            "playerID": player_ids,
            "yearID": years,
            "teamID": teams,
            "salary": [100_000 + i * 500 for i in range(n_rows)],
        }
    ).to_parquet(parquet_dir / "salaries.parquet", index=False)

    pd.DataFrame(
        {
            "playerID": player_ids,
            "yearID": years,
            "teamID": teams,
            "POS": [
                ["P", "C", "1B", "2B", "SS", "3B", "LF", "CF", "RF"][i % 9]
                for i in range(n_rows)
            ],
        }
    ).to_parquet(parquet_dir / "fielding.parquet", index=False)

    return parquet_dir


def _make_multi_join_expr(parquet_dir: Path):

    pg = xo.postgres.connect_env()
    batting = pg.table("batting")
    pg_backend = batting._find_backend()

    local = xo.connect()

    people = deferred_read_parquet(
        parquet_dir / "people.parquet", local, table_name="people"
    )
    salaries = deferred_read_parquet(
        parquet_dir / "salaries.parquet", local, table_name="salaries"
    )
    fielding = deferred_read_parquet(
        parquet_dir / "fielding.parquet", local, table_name="fielding"
    )

    people_pg = people[["playerID", "nameFirst", "nameLast"]].into_backend(pg_backend)
    salaries_pg = salaries[["playerID", "yearID", "teamID", "salary"]].into_backend(
        pg_backend
    )
    fielding_pg = fielding[["playerID", "yearID", "teamID", "POS"]].into_backend(
        pg_backend
    )

    with_names = (
        batting.filter(batting.AB > 0)
        .join(people_pg, predicates="playerID", how="left")
        .drop("playerID_right")
    )
    with_salary = with_names.join(
        salaries_pg,
        predicates=["playerID", "yearID", "teamID"],
        how="left",
    ).drop("playerID_right", "yearID_right", "teamID_right")
    return with_salary.join(
        fielding_pg,
        predicates=["playerID", "yearID", "teamID"],
        how="left",
    ).drop("playerID_right", "yearID_right", "teamID_right")


def _mean_load_time(expr_path: Path, n_runs: int = 10, **kwargs) -> float:
    """Return mean load_expr wall time in seconds over n_runs iterations."""
    # warmup
    for _ in range(3):
        load_expr(expr_path, **kwargs)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        load_expr(expr_path, **kwargs)
    return (time.perf_counter() - t0) / n_runs


@pytest.mark.benchmark
@pytest.mark.postgres
def test_lazy_load_expr_faster_than_eager_postgres(builds_dir, lahman_parquet_dir):

    expr = _make_multi_join_expr(lahman_parquet_dir)
    expr_path = build_expr(expr, builds_dir=builds_dir)

    eager_s = _mean_load_time(expr_path, lazy=False)
    lazy_s = _mean_load_time(expr_path, lazy=True, only_metadata=True)
    speedup = eager_s / lazy_s

    assert speedup > 1, (
        f"Expected lazy load_expr to be >1x faster than eager, "
        f"got {speedup:.2f}x  (eager={eager_s * 1000:.1f}ms, lazy={lazy_s * 1000:.1f}ms)"
    )
