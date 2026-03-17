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
from xorq.vendor.ibis.backends.profiles import Profile


def _all_sources_lazy(expr):
    return all(isinstance(src, LazyBackend) for src in find_all_sources(expr))


def _all_sources_unconnected(expr):
    return all(not src.is_connected for src in find_all_sources(expr))


@pytest.fixture(scope="session")
def awards_players_parquet(parquet_dir):
    return parquet_dir / "awards_players.parquet"


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


def test_hydrate_cons_lazy_returns_lazy_backends(builds_dir, awards_players_parquet):
    backend = xo.duckdb.connect()
    expr = deferred_read_parquet(
        awards_players_parquet, backend, table_name="awards_players"
    )
    build_path = build_expr(expr, builds_dir=builds_dir)

    artifact_store = ArtifactStore(build_path)
    raw_profiles = artifact_store.load_yaml(DumpFiles.profiles)

    lazy_cons = hydrate_cons(raw_profiles, lazy=True)
    eager_cons = hydrate_cons(raw_profiles, lazy=False)

    assert all(isinstance(con, LazyBackend) for con in lazy_cons.values())
    assert all(not isinstance(con, LazyBackend) for con in eager_cons.values())


@pytest.mark.duckdb
def test_load_expr_lazy_sources_unconnected_before_execute(
    builds_dir, awards_players_parquet
):
    backend = xo.duckdb.connect()
    expr = deferred_read_parquet(
        awards_players_parquet, backend, table_name="awards_players"
    )

    build_path = build_expr(expr, builds_dir=builds_dir)
    lazy_expr = load_expr(build_path, lazy=True)

    assert _all_sources_lazy(lazy_expr), "Expected all sources to be LazyBackend"
    assert _all_sources_unconnected(lazy_expr), "Expected all sources to be unconnected"


@pytest.mark.parametrize(
    "backend_factory",
    [
        pytest.param(xo.duckdb.connect, id="duckdb", marks=pytest.mark.duckdb),
        pytest.param(
            xo.datafusion.connect, id="datafusion", marks=pytest.mark.datafusion
        ),
        pytest.param(xo.sqlite.connect, id="sqlite", marks=pytest.mark.sqlite),
        pytest.param(xo.connect, id="xorq"),
    ],
)
def test_load_expr_lazy_connects_and_matches_eager(
    backend_factory, builds_dir, awards_players_parquet
):
    backend = backend_factory()
    expr = (
        deferred_read_parquet(
            awards_players_parquet, backend, table_name="awards_players"
        )
        .filter(lambda t: t.lgID == "NL")
        .drop("yearID", "lgID")
    )

    build_path = build_expr(expr, builds_dir=builds_dir)
    lazy_expr = load_expr(build_path, lazy=True)

    assert _all_sources_lazy(lazy_expr)
    assert _all_sources_unconnected(lazy_expr)

    eager_result = expr.execute()
    lazy_result = lazy_expr.execute()

    assert all(src.is_connected for src in find_all_sources(lazy_expr))
    assert len(lazy_result) > 0
    assert_frame_equal(eager_result, lazy_result)


@pytest.mark.datafusion
def test_load_expr_lazy_datafusion(builds_dir, awards_players_parquet):
    backend = xo.datafusion.connect()
    expr = deferred_read_parquet(
        awards_players_parquet, backend, table_name="awards_players"
    )

    build_path = build_expr(expr, builds_dir=builds_dir)
    lazy_expr = load_expr(build_path, lazy=True)

    assert _all_sources_lazy(lazy_expr)
    assert _all_sources_unconnected(lazy_expr)

    result = lazy_expr.execute()
    assert len(result) > 0
    assert all(src.is_connected for src in find_all_sources(lazy_expr))


@pytest.mark.duckdb
def test_load_expr_default_is_not_lazy(builds_dir, awards_players_parquet):
    backend = xo.duckdb.connect()
    expr = deferred_read_parquet(
        awards_players_parquet, backend, table_name="awards_players"
    )

    build_path = build_expr(expr, builds_dir=builds_dir)
    loaded = load_expr(build_path)

    assert not any(isinstance(src, LazyBackend) for src in find_all_sources(loaded))


@pytest.mark.parametrize(
    "con_name,kwargs",
    [
        pytest.param("duckdb", {"database": ":memory:"}, marks=pytest.mark.duckdb),
        pytest.param("datafusion", {}, marks=pytest.mark.datafusion),
        pytest.param("sqlite", {"database": None}, marks=pytest.mark.sqlite),
    ],
)
def test_get_con_lazy_returns_unconnected_lazy_backend(con_name, kwargs):
    profile = Profile(con_name=con_name, kwargs_tuple=tuple(kwargs.items()))
    lazy_con = profile.get_con(lazy=True)

    assert isinstance(lazy_con, LazyBackend)
    assert not lazy_con.is_connected


@pytest.mark.parametrize(
    "con_name,kwargs",
    [
        pytest.param("duckdb", {"database": ":memory:"}, marks=pytest.mark.duckdb),
        pytest.param("datafusion", {}, marks=pytest.mark.datafusion),
        pytest.param("sqlite", {"database": None}, marks=pytest.mark.sqlite),
    ],
)
def test_get_con_lazy_con_kwargs_set_after_connect(con_name, kwargs):
    profile = Profile(con_name=con_name, kwargs_tuple=tuple(kwargs.items()))
    lazy_con = profile.get_con(lazy=True)

    _ = lazy_con.name  # trigger connection
    assert lazy_con.is_connected
    assert lazy_con._con_kwargs == kwargs


@pytest.mark.postgres
def test_get_con_lazy_postgres_con_kwargs_has_password():
    pg = xo.postgres.connect_env()
    profile = Profile.from_con(pg)
    lazy_con = profile.get_con(lazy=True)

    assert isinstance(lazy_con, LazyBackend)
    assert not lazy_con.is_connected

    _ = lazy_con.name  # trigger connection
    assert lazy_con.is_connected
    assert "password" in lazy_con._con_kwargs


@pytest.mark.parametrize(
    "con_name,kwargs",
    [
        pytest.param("duckdb", {"database": ":memory:"}, marks=pytest.mark.duckdb),
        pytest.param("sqlite", {"database": None}, marks=pytest.mark.sqlite),
    ],
)
def test_into_backend_from_xorq_lazy(con_name, kwargs, awards_players_parquet):
    source_con = xo.connect()
    source_expr = source_con.read_parquet(
        awards_players_parquet, table_name="awards_players"
    )

    profile = Profile(con_name=con_name, kwargs_tuple=tuple(kwargs.items()))
    lazy_con = profile.get_con(lazy=True)

    assert not lazy_con.is_connected

    lazy_result = source_expr.into_backend(lazy_con).execute()

    assert lazy_con.is_connected
    assert len(lazy_result) > 0

    eager_con = profile.get_con(lazy=False)
    eager_result = source_expr.into_backend(eager_con).execute()
    assert_frame_equal(eager_result, lazy_result)


@pytest.mark.postgres
def test_into_backend_from_xorq_lazy_postgres(awards_players_parquet):
    source_con = xo.connect()
    source_expr = source_con.read_parquet(
        awards_players_parquet, table_name="awards_players"
    )

    profile = Profile.from_con(xo.postgres.connect_env())
    lazy_con = profile.get_con(lazy=True)

    assert not lazy_con.is_connected

    result = source_expr.into_backend(lazy_con).execute()

    assert lazy_con.is_connected
    assert len(result) > 0


@pytest.mark.duckdb
def test_get_con_lazy_and_eager_results_match(awards_players_parquet, builds_dir):
    backend = xo.duckdb.connect()
    expr = deferred_read_parquet(
        awards_players_parquet, backend, table_name="awards_players"
    ).filter(lambda t: t.lgID == "NL")

    build_path = build_expr(expr, builds_dir=builds_dir)

    eager_result = load_expr(build_path, lazy=False).execute()
    lazy_result = load_expr(build_path, lazy=True).execute()

    assert len(lazy_result) > 0
    assert_frame_equal(eager_result, lazy_result)


@pytest.mark.datafusion
def test_get_con_lazy_and_eager_results_match_datafusion(
    awards_players_parquet, builds_dir
):
    backend = xo.datafusion.connect()
    expr = deferred_read_parquet(
        awards_players_parquet, backend, table_name="awards_players"
    )

    build_path = build_expr(expr, builds_dir=builds_dir)

    eager_result = load_expr(build_path, lazy=False).execute()
    lazy_result = load_expr(build_path, lazy=True).execute()

    assert len(lazy_result) > 0
    assert_frame_equal(eager_result, lazy_result)


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


@pytest.mark.postgres
def test_lazy_load_expr_postgres(builds_dir, lahman_parquet_dir):
    expr = _make_multi_join_expr(lahman_parquet_dir)
    expr_path = build_expr(expr, builds_dir=builds_dir)
    lazy_expr = load_expr(expr_path, lazy=True, read_only_parquet_metadata=True)

    assert lazy_expr.execute() is not None
