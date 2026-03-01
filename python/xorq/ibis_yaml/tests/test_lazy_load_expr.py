"""Tests for load_expr(..., lazy=True) — backends are LazyBackend instances."""

import xorq.api as xo
from xorq.backends.lazy import LazyBackend
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.common.utils.graph_utils import find_all_sources
from xorq.ibis_yaml.compiler import build_expr, load_expr


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

    assert eager_result.equals(lazy_result)


# ---------------------------------------------------------------------------
# load_expr(lazy=True) — datafusion
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# load_expr default — backward compat
# ---------------------------------------------------------------------------


def test_load_expr_default_is_not_lazy(builds_dir, parquet_dir):
    """load_expr without lazy= must behave as before (eager connections)."""
    backend = xo.duckdb.connect()
    parquet_path = parquet_dir / "awards_players.parquet"
    expr = deferred_read_parquet(parquet_path, backend, table_name="awards_players")

    build_path = build_expr(expr, builds_dir=builds_dir)
    loaded = load_expr(build_path)

    assert not any(isinstance(src, LazyBackend) for src in find_all_sources(loaded))
