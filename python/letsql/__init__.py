"""Initialize LetSQL module."""

from __future__ import annotations

import importlib

from letsql import examples
from letsql.config import options
from letsql.expr import api
from letsql.expr.api import *  # noqa: F403
from letsql.backends.let import Backend


try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

__all__ = [  # noqa: PLE0604
    "api",
    "examples",
    "connect",
    "options",
    *api.__all__,
]

_CUSTOM_BACKENDS = ["postgres", "snowflake"]


def _load_entry_points():
    eps = importlib_metadata.entry_points(group="letsql.backends")
    return sorted(eps)


def load_backend(name):
    if entry_point := next(
        (ep for ep in _load_entry_points() if ep.name == name), None
    ):
        import types
        import letsql

        module = entry_point.load()
        backend = module.Backend()
        backend.register_options()

        def connect(*args, **kwargs):
            return backend.connect(*args, **kwargs)

        connect.__doc__ = backend.do_connect.__doc__
        connect.__wrapped__ = backend.do_connect
        connect.__module__ = f"letsql.{name}"

        proxy = types.ModuleType(f"letsql.{name}")
        setattr(letsql, name, proxy)
        proxy.connect = connect
        proxy.compile = backend.compile
        proxy.has_operation = backend.has_operation
        proxy.name = name
        proxy._from_url = backend._from_url
        # Add any additional methods that should be exposed at the top level
        for attr in getattr(backend, "_top_level_methods", ()):
            setattr(proxy, attr, getattr(backend, attr))

        return proxy


def connect() -> Backend:
    """Create a LETSQL backend."""
    instance = Backend()
    instance.do_connect()
    return instance


def __getattr__(name):
    import ibis

    try:
        importlib.import_module(f"letsql.backends.{name}.hotfix")
    except ModuleNotFoundError:
        pass

    return load_backend(name) or ibis.__getattr__(name)


__version__ = importlib_metadata.version(__name__)
