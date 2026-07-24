from __future__ import annotations

import functools
import types


try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata


@functools.cache
def _load_entry_points() -> tuple:
    # cached: the installed xorq.backends entry points are fixed for the life of
    # the process (nothing installs a backend distribution at runtime), and this
    # is called on hot paths like secret validation. Returns a tuple so the
    # shared cached value can't be mutated in place by a caller.
    eps = importlib_metadata.entry_points(group="xorq.backends")
    return tuple(sorted(eps))


def load_backend(name):
    if entry_point := next(
        (ep for ep in _load_entry_points() if ep.name == name), None
    ):
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
