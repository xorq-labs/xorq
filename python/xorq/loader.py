from __future__ import annotations

import functools
import importlib
import types


try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata


@functools.cache
def _load_entry_points() -> tuple[importlib_metadata.EntryPoint, ...]:
    # cached: the scan costs ~15ms and validate_con_name runs per Profile. A
    # tuple, so the shared value can't be mutated in place by a caller.
    eps = importlib_metadata.entry_points(group="xorq.backends")
    return tuple(sorted(eps))


def _find_entry_point(name: str) -> importlib_metadata.EntryPoint | None:
    """Look up a `xorq.backends` entry point, refreshing the cache once on a miss.

    Resolve through here rather than scanning `_load_entry_points()`: a direct
    scan sees the cache as it was, so it rejects a backend installed into a live
    process (pip install in a Jupyter kernel) until that process restarts. Only a
    miss pays the rescan.
    """
    if entry_point := next(
        (ep for ep in _load_entry_points() if ep.name == name), None
    ):
        return entry_point
    importlib.invalidate_caches()  # a new dist postdates the import caches too
    _load_entry_points.cache_clear()
    return next((ep for ep in _load_entry_points() if ep.name == name), None)


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
