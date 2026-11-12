from __future__ import annotations

import functools
import pathlib
import sys
import time
import types


try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata


BACKEND_GROUPS = ("xorq.backends", "ibis.backends")
# a directory changed this recently may still be mid-install (entry_points.txt
# not yet written into its new .dist-info), so its scan is not cached
RACY_WINDOW_NS = 2_000_000_000


def _env_fingerprint() -> tuple[tuple[str, int], ...]:
    """Each sys.path directory with its mtime.

    Installing a distribution writes a .dist-info directory into a sys.path
    directory (pip and uv both), which moves that directory's mtime, and a new
    sys.path entry changes the tuple. An unchanged fingerprint therefore means no
    new entry points, so a lookup miss costs a few stats instead of a rescan, and
    a backend installed into a live process (pip install in a Jupyter kernel) is
    still found without a restart.
    """
    return tuple(
        (path, directory.stat().st_mtime_ns)
        for path in sys.path
        if (directory := pathlib.Path(path)).is_dir()
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
    fingerprint = _env_fingerprint()
    newest = max((mtime for _, mtime in fingerprint), default=0)
    scan = (
        _scan_entry_points.__wrapped__
        if time.time_ns() - newest < RACY_WINDOW_NS
        else _scan_entry_points
    )
    return scan(group, fingerprint)


def _find_entry_point(name: str) -> importlib_metadata.EntryPoint | None:
    """Look up a `xorq.backends` entry point, rescanning only when sys.path changed."""
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
