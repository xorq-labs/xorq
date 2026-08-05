from __future__ import annotations

import importlib
import pathlib
import sys

import pytest

from xorq.loader import (
    _find_entry_point,
    _load_entry_points,
    load_backend,
)


def _write_fake_dist(root: pathlib.Path, con_name: str) -> None:
    """Write a minimal installed distribution declaring a xorq.backends entry
    point, so adding `root` to sys.path is indistinguishable from installing a
    backend plugin."""
    dist_info = root / f"xorq_{con_name}-0.1.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: xorq-{con_name}\nVersion: 0.1\n"
    )
    (dist_info / "entry_points.txt").write_text(
        f"[xorq.backends]\n{con_name} = xorq_{con_name}\n"
    )


def test_load_entry_points_returns_cached_tuple() -> None:
    """The value is a tuple (callers share it, so it must not be mutable in
    place) and repeated calls reuse it rather than rescanning."""
    first = _load_entry_points()
    assert isinstance(first, tuple)
    assert _load_entry_points() is first


def test_find_entry_point_resolves_an_installed_backend() -> None:
    name = _load_entry_points()[0].name
    assert _find_entry_point(name) is not None


def test_find_entry_point_refreshes_a_stale_cache(tmp_path: pathlib.Path) -> None:
    """A backend installed after the cache was populated is still resolvable.

    The cache is what makes repeated Profile construction cheap, but it means a
    distribution installed into a live process (pip install in a Jupyter kernel)
    is invisible to an already-populated cache. Resolution refreshes on a miss so
    that staleness never surfaces as an unresolvable backend.
    """
    con_name = "xorqfakebackend"
    _write_fake_dist(tmp_path, con_name)
    # populate the cache *before* the install, which is the stale-cache setup
    assert not any(ep.name == con_name for ep in _load_entry_points())
    sys.path.insert(0, str(tmp_path))
    try:
        # the cached value still predates the install ...
        assert not any(ep.name == con_name for ep in _load_entry_points())
        # ... but resolution refreshes past it
        entry_point = _find_entry_point(con_name)
        assert entry_point is not None
        assert entry_point.name == con_name
    finally:
        sys.path.remove(str(tmp_path))
        importlib.invalidate_caches()
        _load_entry_points.cache_clear()
    assert not any(ep.name == con_name for ep in _load_entry_points())


def test_find_entry_point_returns_none_for_unknown_backend() -> None:
    assert _find_entry_point("xorq-no-such-backend") is None


def test_load_backend_returns_none_for_unknown_backend() -> None:
    assert load_backend("xorq-no-such-backend") is None


@pytest.mark.parametrize("con_name", sorted(ep.name for ep in _load_entry_points()))
def test_load_backend_entry_points_are_loadable(con_name: str) -> None:
    """Every declared entry point names a module exposing a `Backend`, which is
    what both `load_backend` and the dynamic secret-key lookup assume.

    A backend whose optional dependencies aren't installed is skipped: entry
    points are declared for every backend regardless of which extras the
    environment has.
    """
    entry_point = _find_entry_point(con_name)
    assert entry_point is not None
    try:
        module = entry_point.load()
    except ImportError as e:
        pytest.skip(f"{con_name} backend not importable: {e}")
    assert hasattr(module, "Backend")
