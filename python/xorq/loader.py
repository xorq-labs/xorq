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
    # cached: scanning the installed distributions costs ~15ms, and this is
    # called on paths that run per-object rather than once -- Profile
    # construction (validate_con_name) and secret validation. Returns a tuple so
    # the shared cached value can't be mutated in place by a caller.
    #
    # The cache can go stale: installing a backend distribution into a live
    # process (pip install in a Jupyter kernel) adds entry points that an
    # already-populated cache will not show. Resolution therefore refreshes on a
    # miss -- see _find_entry_point -- so the staleness is not observable as an
    # unresolvable backend.
    eps = importlib_metadata.entry_points(group="xorq.backends")
    return tuple(sorted(eps))


def _find_entry_point(name: str) -> importlib_metadata.EntryPoint | None:
    """Look up a `xorq.backends` entry point by name, refreshing once on a miss.

    A miss can mean either "no such backend" or "the cache predates a
    mid-process install", and the two are indistinguishable without rescanning.
    Rescanning only on a miss keeps the hit path free: a name that resolves
    never pays for it, and a name that genuinely doesn't exist pays a ~15ms
    rescan rather than staying unresolvable for the life of the process.

    Resolve names through here rather than by scanning `_load_entry_points()`
    directly -- a direct scan sees the cache as it was, so it would reject a
    just-installed backend (see `profiles.Profile.validate_con_name`).
    """
    if entry_point := next(
        (ep for ep in _load_entry_points() if ep.name == name), None
    ):
        return entry_point
    # a distribution installed after the cache was populated may also postdate
    # the import system's own path caches
    importlib.invalidate_caches()
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
