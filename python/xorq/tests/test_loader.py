from __future__ import annotations

import importlib.metadata
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


def test_find_entry_point_sees_entry_points_written_after_a_lookup(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lookup that lands mid-install, after the .dist-info directory exists
    but before entry_points.txt is written, must not hide the backend once the
    file lands."""
    monkeypatch.syspath_prepend(tmp_path)
    dist_info = tmp_path / "xorq_xorqlatebackend-0.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: xorq-xorqlatebackend\nVersion: 0.1\n"
    )
    assert _find_entry_point("xorqlatebackend") is None
    (dist_info / "entry_points.txt").write_text(
        "[xorq.backends]\nxorqlatebackend = xorq_xorqlatebackend\n"
    )
    assert _find_entry_point("xorqlatebackend") is not None


def test_find_entry_point_miss_ignores_unrelated_writes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Files written into a sys.path directory (the cwd entry, notebook output)
    are not distributions, so a later miss does not rescan entry points."""
    monkeypatch.syspath_prepend(tmp_path)
    assert _find_entry_point("xorq-no-such-backend") is None
    (tmp_path / "output.csv").write_text("a\n1\n")
    scans = []
    entry_points = importlib.metadata.entry_points
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda **kwargs: scans.append(kwargs) or entry_points(**kwargs),
    )
    assert _find_entry_point("xorq-no-such-backend") is None
    assert scans == []
