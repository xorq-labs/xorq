from __future__ import annotations

import pathlib

import pytest

from xorq.loader import (
    _find_entry_point,
    _load_entry_points,
    load_backend,
)
from xorq.tests.util import installed_mid_process


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
    """A backend installed after the cache was populated is still resolvable:
    resolution refreshes on a miss, so staleness never surfaces as an
    unresolvable backend."""
    with installed_mid_process(tmp_path, "xorqfakebackend") as con_name:
        # resolution refreshes past the stale cache
        entry_point = _find_entry_point(con_name)
        assert entry_point is not None
        assert entry_point.name == con_name
    assert not any(ep.name == con_name for ep in _load_entry_points())


def test_find_entry_point_returns_none_for_unknown_backend() -> None:
    assert _find_entry_point("xorq-no-such-backend") is None


def test_load_backend_returns_none_for_unknown_backend() -> None:
    assert load_backend("xorq-no-such-backend") is None


@pytest.mark.parametrize("con_name", sorted(ep.name for ep in _load_entry_points()))
def test_load_backend_entry_points_are_loadable(con_name: str) -> None:
    """Every declared entry point names a module exposing a `Backend`, which both
    `load_backend` and the dynamic secret-key lookup assume. Skipped for a backend
    whose extras aren't installed -- entry points are declared regardless."""
    entry_point = _find_entry_point(con_name)
    assert entry_point is not None
    try:
        module = entry_point.load()
    except ImportError as e:
        pytest.skip(f"{con_name} backend not importable: {e}")
    assert hasattr(module, "Backend")
