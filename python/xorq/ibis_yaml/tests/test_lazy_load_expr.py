import time
from pathlib import Path

import pandas as pd
import pytest

import xorq.api as xo
from xorq.backends._lazy import LazyBackend
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.common.utils.graph_utils import find_all_sources
from xorq.ibis_yaml.compiler import (
    ArtifactStore,
    DumpFiles,
    build_expr,
    hydrate_cons,
    load_expr,
)
from xorq.tests.util import assert_frame_equal


def _all_sources_lazy(expr):
    return all(isinstance(src, LazyBackend) for src in find_all_sources(expr))


def _all_sources_unconnected(expr):
    return all(not src.is_connected for src in find_all_sources(expr))


def test_hydrate_cons_lazy_returns_lazy_backends(builds_dir, parquet_dir):
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


@pytest.mark.duckdb
def test_load_expr_lazy_sources_unconnected_before_execute(builds_dir, parquet_dir):
    backend = xo.duckdb.connect()
    parquet_path = parquet_dir / "awards_players.parquet"
    expr = deferred_read_parquet(parquet_path, backend, table_name="awards_players")

    build_path = build_expr(expr, builds_dir=builds_dir)
    lazy_expr = load_expr(build_path, lazy=True)

    assert _all_sources_lazy(lazy_expr), "Expected all sources to be LazyBackend"
    assert _all_sources_unconnected(lazy_expr), "Expected all sources to be unconnected"


@pytest.mark.duckdb
def test_load_expr_lazy_connects_on_execute(builds_dir, parquet_dir):
    backend = xo.duckdb.connect()
    parquet_path = parquet_dir / "awards_players.parquet"
    expr = deferred_read_parquet(parquet_path, backend, table_name="awards_players")

    build_path = build_expr(expr, builds_dir=builds_dir)
    lazy_expr = load_expr(build_path, lazy=True)

    lazy_expr.execute()

    assert all(src.is_connected for src in find_all_sources(lazy_expr))


@pytest.mark.duckdb
def test_load_expr_lazy_result_matches_eager(builds_dir, parquet_dir):
    backend = xo.duckdb.connect()
    parquet_path = parquet_dir / "awards_players.parquet"
    expr = deferred_read_parquet(parquet_path, backend, table_name="awards_players")
    expr = expr.filter(expr.lgID == "NL").drop("yearID", "lgID")

    build_path = build_expr(expr, builds_dir=builds_dir)

    eager_result = load_expr(build_path, lazy=False).execute()
    lazy_result = load_expr(build_path, lazy=True).execute()

    assert len(eager_result) > 0
    assert_frame_equal(eager_result, lazy_result)


@pytest.mark.datafusion
def test_load_expr_lazy_datafusion(builds_dir, parquet_dir):
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


@pytest.mark.duckdb
def test_load_expr_default_is_not_lazy(builds_dir, parquet_dir):
    backend = xo.duckdb.connect()
    parquet_path = parquet_dir / "awards_players.parquet"
    expr = deferred_read_parquet(parquet_path, backend, table_name="awards_players")

    build_path = build_expr(expr, builds_dir=builds_dir)
    loaded = load_expr(build_path)

    assert not any(isinstance(src, LazyBackend) for src in find_all_sources(loaded))


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
    pg = xo.postgres.connect_examples()
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
    for _ in range(3):
        load_expr(expr_path, **kwargs)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        load_expr(expr_path, **kwargs)
    return (time.perf_counter() - t0) / n_runs


@pytest.mark.postgres
def test_lazy_load_expr_faster_than_eager_postgres(builds_dir, lahman_parquet_dir):
    expr = _make_multi_join_expr(lahman_parquet_dir)
    expr_path = build_expr(expr, builds_dir=builds_dir)

    eager_s = _mean_load_time(expr_path, lazy=False)
    lazy_s = _mean_load_time(expr_path, lazy=True, read_only_parquet_metadata=True)
    speedup = eager_s / lazy_s

    assert speedup > 1, (
        f"Expected lazy load_expr to be >1x faster than eager, "
        f"got {speedup:.2f}x  (eager={eager_s * 1000:.1f}ms, lazy={lazy_s * 1000:.1f}ms)"
    )
