from __future__ import annotations

import pkgutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

import xorq.api as xo
import xorq.backends
from xorq.caching import ParquetCache
from xorq.cli import cli, pin_command, run_command
from xorq.common.exceptions import IntegrityError
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.relations import (
    CachedNode,
    CacheTag,
    _cached_node_materialized,
    pin_cache,
)
from xorq.ibis_yaml.compiler import build_expr, load_expr
from xorq.vendor.ibis.expr import operations as ops
from xorq.vendor.ibis.expr.types.core import Expr


def caches_materialized(expr: Expr) -> bool:
    """True if every ``CachedNode`` in ``expr`` has its cache populated."""
    return all(map(_cached_node_materialized, expr.ls.cached_nodes))


def _write_parquet(path: Path) -> None:
    pq.write_table(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}), path)


def _cached_build(
    tmp_path: Path, materialize: bool = True, relocate_reads: bool = True
) -> tuple[Path, Path, Path]:
    """Build a single-cache expr; optionally materialize the cache via a run.

    Defaults to ``relocate_reads=True`` to match the ``xorq build`` default, so
    pin/unpin run in the same (default) regime they do in practice.
    """
    parquet_path = tmp_path / "data.parquet"
    _write_parquet(parquet_path)
    con = xo.connect()
    cache = ParquetCache.from_kwargs(source=con)
    t = deferred_read_parquet(parquet_path, con, table_name="t")
    expr = t.filter(t.a > 1).cache(cache=cache)
    builds_dir = tmp_path / "builds"
    cache_dir = tmp_path / "cache"
    build_path = build_expr(
        expr, builds_dir=builds_dir, cache_dir=cache_dir, relocate_reads=relocate_reads
    )
    if materialize:
        run_command(
            str(build_path), str(tmp_path / "out.parquet"), "parquet", str(cache_dir)
        )
    return build_path, builds_dir, cache_dir


def test_pin_build_roundtrip(tmp_path: Path) -> None:
    build_path, builds_dir, cache_dir = _cached_build(tmp_path)
    original = load_expr(build_path, cache_dir=cache_dir)

    pinned_path = pin_command(
        str(build_path),
        do_pin=True,
        builds_dir=str(builds_dir),
        cache_dir=str(cache_dir),
    )
    pinned = load_expr(pinned_path, cache_dir=cache_dir)

    assert walk_nodes((CacheTag,), pinned)
    assert not walk_nodes((CachedNode,), pinned)
    assert (
        pinned.execute()
        .reset_index(drop=True)
        .equals(original.execute().reset_index(drop=True))
    )


def test_ls_pin_ensure_materialized_populates_cold_cache(tmp_path: Path) -> None:
    build_path, _, cache_dir = _cached_build(tmp_path, materialize=False)
    expr = load_expr(build_path, cache_dir=cache_dir)
    assert not caches_materialized(expr)

    # without the flag a cold cache cannot be pinned
    with pytest.raises(IntegrityError):
        expr.ls.pin()

    pinned = expr.ls.pin(ensure_materialized=True)
    assert walk_nodes((CacheTag,), pinned)
    assert not walk_nodes((CachedNode,), pinned)

    # head(0) populates the *full* cache, not zero rows: the source has rows
    # a in {1, 2, 3} and the build filters a > 1, so the pinned read yields 2.
    assert len(pinned.execute()) == 2


def test_pin_ensure_materialized_skips_post_cache_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet_path = tmp_path / "data.parquet"
    _write_parquet(parquet_path)
    con = xo.connect()
    cache = ParquetCache.from_kwargs(source=con)
    t = deferred_read_parquet(parquet_path, con, table_name="t")
    # the cache is NOT the root: a filter sits downstream of it
    expr = t.cache(cache=cache).filter(xo._.a > 1)
    assert not caches_materialized(expr)

    executed: list = []
    original_execute = Expr.execute

    def recording_execute(self, *args, **kwargs):
        executed.append(self)
        return original_execute(self, *args, **kwargs)

    monkeypatch.setattr(Expr, "execute", recording_execute)

    pinned = pin_cache(expr, ensure_materialized=True)

    # exactly the cache subtree was executed -- the downstream filter never ran,
    # and it was wrapped in head(0) so no cache rows were pulled into memory
    assert executed
    for e in executed:
        assert walk_nodes((CachedNode,), e)  # the cache is materialized
        assert not walk_nodes((ops.Filter,), e)  # post-cache compute skipped
        assert isinstance(e.op(), ops.Limit) and e.op().n == 0  # head(0)
    assert caches_materialized(expr)
    assert walk_nodes((CacheTag,), pinned)


def test_caches_materialized_predicate(tmp_path: Path) -> None:
    build_path, _, cache_dir = _cached_build(tmp_path, materialize=False)
    assert not caches_materialized(load_expr(build_path, cache_dir=cache_dir))

    run_command(
        str(build_path), str(tmp_path / "out.parquet"), "parquet", str(cache_dir)
    )
    assert caches_materialized(load_expr(build_path, cache_dir=cache_dir))


def test_ensure_materialized_skips_execute_when_already_materialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_path, builds_dir, cache_dir = _cached_build(tmp_path, materialize=True)

    calls = {"n": 0}
    original_execute = Expr.execute

    def counting_execute(self, *args, **kwargs):
        calls["n"] += 1
        return original_execute(self, *args, **kwargs)

    monkeypatch.setattr(Expr, "execute", counting_execute)

    # caches are already materialized, so ensure_materialized must not re-execute
    pin_command(
        str(build_path),
        do_pin=True,
        builds_dir=str(builds_dir),
        cache_dir=str(cache_dir),
        ensure_materialized=True,
    )
    assert calls["n"] == 0


def test_pin_requires_materialized_cache_then_ensure_materialized(
    tmp_path: Path,
) -> None:
    build_path, builds_dir, cache_dir = _cached_build(tmp_path, materialize=False)

    result = CliRunner().invoke(
        cli,
        [
            "pin",
            str(build_path),
            "--cache-dir",
            str(cache_dir),
            "--builds-dir",
            str(builds_dir),
        ],
    )
    assert result.exit_code != 0
    assert "ensure-materialized" in result.output

    result_e = CliRunner().invoke(
        cli,
        [
            "pin",
            str(build_path),
            "--cache-dir",
            str(cache_dir),
            "--builds-dir",
            str(builds_dir),
            "-e",
        ],
    )
    assert result_e.exit_code == 0, result_e.output


def _backend_names() -> list[str]:
    """All backend modules, discovered from ``xorq.backends``.

    Deriving the list dynamically means a newly added backend is automatically
    swept into the contract test below -- nothing to remember to update.
    """
    return sorted(
        m.name
        for m in pkgutil.iter_modules(xorq.backends.__path__)
        if not m.name.startswith("_") and m.name not in ("tests", "conftest")
    )


@pytest.mark.slow(level=1)
@pytest.mark.parametrize("backend_name", _backend_names())
def test_ensure_materialized_full_cache_contract(
    backend_name: str, tmp_path: Path
) -> None:
    """``head(0).execute()`` must populate the FULL cache on every backend.

    ``pin(ensure_materialized=True)`` relies on cache population being a side
    effect of the execute pass, independent of the client row limit. A backend
    that pushed the limit into cache population would write an empty cache and
    pin a silently-wrong artifact. Backends that cannot host the flow at all
    (need a server/creds, or cannot ingest parquet) skip via the capability
    probe below, but any backend that *runs* the flow yet truncates the cache
    fails on the row-count assertion -- so a newly supported backend cannot
    regress pinning unnoticed.
    """
    parquet_path = tmp_path / "data.parquet"
    _write_parquet(parquet_path)  # a in {1, 2, 3}
    # Capability probe -- the ONLY code guarded by this try. It answers "can this
    # backend connect and ingest+execute parquet locally", nothing more. The pin
    # flow below is deliberately left outside the guard: a backend that clears
    # the probe but breaks in .cache()/pin must fail loudly, not skip. Do not
    # widen this try to cover the assertions, or a real pin regression would be
    # silently swallowed as a skip.
    try:
        con = getattr(xo, backend_name).connect()
        deferred_read_parquet(parquet_path, con, table_name="probe").execute()
    except Exception as exc:  # noqa: BLE001 -- needs server/creds, or no parquet ingest
        pytest.skip(f"{backend_name}: cannot host the pin flow locally ({exc!r})")

    cache = ParquetCache.from_kwargs(source=con, base_path=tmp_path / "cache")
    t = deferred_read_parquet(parquet_path, con, table_name="t")
    expr = t.cache(cache=cache).filter(xo._.a > 1)
    assert not caches_materialized(expr)

    pinned = expr.ls.pin(ensure_materialized=True)
    assert walk_nodes((CacheTag,), pinned)
    assert not walk_nodes((CachedNode,), pinned)
    # a > 1 over {1, 2, 3} -> 2 rows; 0 would mean head(0) leaked the row limit
    # into cache population.
    assert len(pinned.execute()) == 2


def test_ensure_materialized_repairs_evicted_inner_cache(tmp_path: Path) -> None:
    """An inner cache colder than its outer is still re-materialized.

    With nested caches both warm, evicting only the inner cache file leaves the
    outer warm. ``ensure_materialized`` checks every cache, so it re-materializes
    the cold inner directly (the warm outer is skipped) and pins cleanly --
    nothing relies on a warm outer implying a warm inner.
    """
    parquet_path = tmp_path / "data.parquet"
    _write_parquet(parquet_path)
    con = xo.connect()
    inner = ParquetCache.from_kwargs(source=con, base_path=tmp_path / "inner")
    outer = ParquetCache.from_kwargs(source=con, base_path=tmp_path / "outer")
    t = deferred_read_parquet(parquet_path, con, table_name="t")
    expr = t.cache(cache=inner).filter(xo._.a > 1).cache(cache=outer)
    expr.execute()  # warm both caches

    # evict only the inner cache file, leaving the outer warm
    for f in Path(inner.storage.path).glob("*.parquet"):
        f.unlink()
    assert not caches_materialized(expr)

    # the cold inner is repaired even though its outer is warm
    pinned = pin_cache(expr, ensure_materialized=True)
    assert caches_materialized(expr)
    assert walk_nodes((CacheTag,), pinned)
    assert not walk_nodes((CachedNode,), pinned)
    assert len(pinned.execute()) == 2


def test_unpin_inverts_pin(tmp_path: Path) -> None:
    build_path, builds_dir, cache_dir = _cached_build(tmp_path)

    pinned_path = pin_command(
        str(build_path),
        do_pin=True,
        builds_dir=str(builds_dir),
        cache_dir=str(cache_dir),
    )
    unpinned_path = pin_command(
        str(pinned_path),
        do_pin=False,
        builds_dir=str(builds_dir),
        cache_dir=str(cache_dir),
    )
    unpinned = load_expr(unpinned_path, cache_dir=cache_dir)

    assert walk_nodes((CachedNode,), unpinned)
    assert not walk_nodes((CacheTag,), unpinned)


def test_pin_unpin_build_hash_roundtrip(tmp_path: Path) -> None:
    build_path, builds_dir, cache_dir = _cached_build(tmp_path)

    pinned_path = pin_command(
        str(build_path),
        do_pin=True,
        builds_dir=str(builds_dir),
        cache_dir=str(cache_dir),
    )
    # pinning is intentionally not build-hash-neutral
    assert pinned_path.name != build_path.name

    unpinned_path = pin_command(
        str(pinned_path),
        do_pin=False,
        builds_dir=str(builds_dir),
        cache_dir=str(cache_dir),
    )
    # unpin restores the original unpinned build hash (relocate-on round-trip is
    # now hash-stable, so this holds in the default regime too)
    assert unpinned_path.name == build_path.name


def test_pin_partial_materialization_errors(tmp_path: Path) -> None:
    parquet_path = tmp_path / "data.parquet"
    _write_parquet(parquet_path)
    con = xo.connect()
    t = deferred_read_parquet(parquet_path, con, table_name="t")
    expr = (
        t.cache(ParquetCache.from_kwargs(source=con))
        .filter(xo._.a > 1)
        .cache(ParquetCache.from_kwargs(source=con))
    )
    builds_dir = tmp_path / "builds"
    cache_dir = tmp_path / "cache"
    build_path = build_expr(expr, builds_dir=builds_dir, cache_dir=cache_dir)
    run_command(
        str(build_path), str(tmp_path / "out.parquet"), "parquet", str(cache_dir)
    )

    cache_files = list(cache_dir.rglob("*.parquet"))
    assert len(cache_files) >= 2
    cache_files[0].unlink()

    result = CliRunner().invoke(
        cli,
        [
            "pin",
            str(build_path),
            "--cache-dir",
            str(cache_dir),
            "--builds-dir",
            str(builds_dir),
        ],
    )
    assert result.exit_code != 0


def test_unpin_missing_source_gives_clean_error(tmp_path: Path) -> None:
    """Relocating with a missing source fails cleanly, not with a raw
    FileNotFoundError.

    A lean build never bundles its source; unpinning makes that read live
    again, so relocating content-hashes a source that no longer exists.
    """
    build_path, builds_dir, cache_dir = _cached_build(tmp_path, relocate_reads=False)
    pinned = pin_command(
        str(build_path),
        do_pin=True,
        builds_dir=str(builds_dir),
        cache_dir=str(cache_dir),
        relocate_reads=False,
    )
    (tmp_path / "data.parquet").unlink()

    result = CliRunner().invoke(
        cli,
        [
            "unpin",
            str(pinned),
            "--cache-dir",
            str(cache_dir),
            "--builds-dir",
            str(builds_dir),
            "--relocate-reads",
        ],
    )
    assert result.exit_code != 0
    assert "cannot bundle reads" in result.output
    assert "--no-relocate-reads" in result.output


def test_pin_relocate_reads_is_self_contained(tmp_path: Path) -> None:
    build_path, builds_dir, cache_dir = _cached_build(tmp_path)

    pinned_path = pin_command(
        str(build_path),
        do_pin=True,
        builds_dir=str(builds_dir),
        cache_dir=str(cache_dir),
        relocate_reads=True,
    )
    reads_dir = pinned_path / "reads"
    assert reads_dir.exists()
    assert list(reads_dir.glob("*.parquet"))


def test_pin_freezes_cache_against_source_change(tmp_path: Path) -> None:
    parquet_path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}), parquet_path)
    con = xo.connect()
    cache = ParquetCache.from_kwargs(source=con)
    t = deferred_read_parquet(parquet_path, con, table_name="t")
    expr = t.filter(t.a > 1).cache(cache=cache)
    builds_dir = tmp_path / "builds"
    cache_dir = tmp_path / "cache"
    raw_build = build_expr(expr, builds_dir=builds_dir, cache_dir=cache_dir)

    def run(build: Path, name: str) -> list:
        out = tmp_path / f"{name}.parquet"
        run_command(str(build), str(out), "parquet", str(cache_dir))
        return pq.read_table(out).to_pydict()["a"]

    def caches() -> set[str]:
        return {p.name for p in cache_dir.rglob("*.parquet")}

    # materialize the cache, then freeze it
    assert run(raw_build, "raw0") == [2, 3]
    frozen_caches = caches()
    assert len(frozen_caches) == 1

    # stay in the non-relocate regime: this test exercises stat-based read
    # invalidation when the source file changes, which relocating (md5 + a
    # frozen bundled copy) would deliberately defeat.
    pinned = pin_command(
        str(raw_build),
        do_pin=True,
        builds_dir=str(builds_dir),
        cache_dir=str(cache_dir),
        relocate_reads=False,
    )
    assert walk_nodes((CacheTag,), load_expr(pinned, cache_dir=cache_dir))

    # change the source file out from under both builds: the stat-based read key
    # changes, so the raw cache is invalidated
    pq.write_table(pa.table({"a": [1, 2, 3, 4, 5], "b": [4, 5, 6, 7, 8]}), parquet_path)

    # raw build sees the change: recomputes and writes a new cache file
    assert run(raw_build, "raw1") == [2, 3, 4, 5]
    after_raw = caches()
    live_caches = after_raw - frozen_caches
    assert len(live_caches) == 1

    # pinned build is frozen: original data, reads its frozen cache, no new file
    assert run(pinned, "pin1") == [2, 3]
    assert caches() == after_raw

    # unpin restores a live cache that tracks the changed source again
    unpinned = pin_command(
        str(pinned),
        do_pin=False,
        builds_dir=str(builds_dir),
        cache_dir=str(cache_dir),
        relocate_reads=False,
    )
    assert walk_nodes((CachedNode,), load_expr(unpinned, cache_dir=cache_dir))
    assert run(unpinned, "unpin1") == [2, 3, 4, 5]
    # the unpinned form re-keys to the same live cache a fresh raw build uses,
    # which is distinct from the frozen pin's cache
    assert caches() == after_raw
    assert live_caches.isdisjoint(frozen_caches)
    assert frozen_caches <= caches()


def test_pin_noop_without_caches(tmp_path: Path) -> None:
    parquet_path = tmp_path / "data.parquet"
    _write_parquet(parquet_path)
    con = xo.connect()
    t = deferred_read_parquet(parquet_path, con, table_name="t")
    expr = t.filter(t.a > 1)
    builds_dir = tmp_path / "builds"
    cache_dir = tmp_path / "cache"
    # build relocate-on (the `xorq build` default) so the default-relocating pin
    # runs in the same regime; with no caches the pin is a structural no-op and,
    # because a relocated build is load+rebuild hash-stable, the hash is unchanged.
    build_path = build_expr(
        expr, builds_dir=builds_dir, cache_dir=cache_dir, relocate_reads=True
    )
    original = load_expr(build_path, cache_dir=cache_dir)

    result = CliRunner().invoke(
        cli,
        [
            "pin",
            str(build_path),
            "--cache-dir",
            str(cache_dir),
            "--builds-dir",
            str(builds_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    pinned_path = Path(result.output.strip().splitlines()[-1])
    pinned = load_expr(pinned_path, cache_dir=cache_dir)
    # no caches to freeze: pin is a no-op, so the build hash is unchanged
    assert not walk_nodes((CacheTag,), pinned)
    assert not walk_nodes((CachedNode,), pinned)
    assert pinned_path.name == build_path.name
    assert (
        pinned.execute()
        .reset_index(drop=True)
        .equals(original.execute().reset_index(drop=True))
    )
