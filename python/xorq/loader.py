from __future__ import annotations

import functools
import os
import pathlib
import sys
import types


try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata


BACKEND_GROUPS = ("xorq.backends", "ibis.backends")


def _env_fingerprint() -> tuple[tuple[str, int], ...]:
    """Every installed distribution's metadata directory with its mtime.

    Installing a distribution adds a .dist-info directory and writing its
    entry_points.txt moves that directory's mtime; uninstalling removes it; a new
    sys.path entry brings its own. Any of these changes the tuple, so an
    unchanged fingerprint means unchanged entry points: a lookup miss costs a
    directory listing per sys.path entry instead of a rescan, and a backend
    installed into a live process (pip install in a Jupyter kernel) is still
    found without a restart.
    """
    return tuple(
        (entry.path, entry.stat().st_mtime_ns)
        for path in sys.path
        if pathlib.Path(path or ".").is_dir()
        for entry in os.scandir(path or ".")
        if entry.name.endswith((".dist-info", ".egg-info"))
    )


@functools.lru_cache(maxsize=len(BACKEND_GROUPS))
def _scan_entry_points(
    group: str, fingerprint: tuple[tuple[str, int], ...]
) -> tuple[importlib_metadata.EntryPoint, ...]:
    # a tuple, so the shared value can't be mutated in place by a caller
    return tuple(sorted(importlib_metadata.entry_points(group=group)))


def _load_entry_points(
    group: str = "xorq.backends",
) -> tuple[importlib_metadata.EntryPoint, ...]:
    return _scan_entry_points(group, _env_fingerprint())


def _find_entry_point(name: str) -> importlib_metadata.EntryPoint | None:
    """Look up a `xorq.backends` entry point, rescanning only when distributions changed."""
    return next((ep for ep in _load_entry_points() if ep.name == name), None)


def backend_names() -> frozenset[str]:
    """Names `xorq.api` resolves as backends: xorq's and ibis's entry points."""
    return frozenset(
        ep.name for group in BACKEND_GROUPS for ep in _load_entry_points(group)
    )


def load_backend(name: str) -> types.ModuleType | None:
    if entry_point := _find_entry_point(name):
        module = entry_point.load()
        backend = module.Backend()
        backend.register_options()

        def connect(*args, **kwargs):
            return backend.connect(*args, **kwargs)

        connect.__doc__ = backend.do_connect.__doc__
        connect.__wrapped__ = backend.do_connect
        connect.__module__ = f"xorq.{name}"

        proxy = types.ModuleType(f"xorq.{name}")
        proxy.connect = connect
        proxy.compile = backend.compile
        proxy.has_operation = backend.has_operation
        proxy.name = name
        proxy._from_url = backend._from_url

        # Add any additional methods that should be exposed at the top level
        for attr in getattr(backend, "_top_level_methods", ()):
            setattr(proxy, attr, getattr(backend, attr))

        return proxy
