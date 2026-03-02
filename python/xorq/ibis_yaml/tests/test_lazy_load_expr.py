"""Tests for load_expr(..., lazy=True / limit=N) options."""

import time

import pandas as pd

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


# ---------------------------------------------------------------------------
# load_expr(limit=N) — deferred_reads_to_memtables row cap
# ---------------------------------------------------------------------------


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
    """Confirm the limit is applied during the parquet read, not after a full scan.

    We use functional_alltypes (7 300 rows) as the backing data.  Reading
    10 rows must be meaningfully faster than reading all rows; the large
    row-count ratio (730×) makes this reliable even on fast hardware.
    """
    df = pd.read_parquet(parquet_dir / "functional_alltypes.parquet")
    n_rows = len(df)  # 7300
    build_path = build_expr(xo.memtable(df), builds_dir=builds_dir)

    t0 = time.perf_counter()
    small_result = load_expr(build_path, limit=10).execute()
    t_small = time.perf_counter() - t0

    t0 = time.perf_counter()
    full_result = load_expr(build_path, limit=n_rows).execute()
    t_full = time.perf_counter() - t0

    assert len(small_result) == 10
    assert len(full_result) == n_rows
    assert t_small < t_full, (
        f"Expected reading 10 rows ({t_small:.4f}s) to be faster than "
        f"reading {n_rows} rows ({t_full:.4f}s)"
    )
