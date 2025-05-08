"""Initialize xorq module."""

from __future__ import annotations

from xorq import examples
from xorq import udf
from xorq.udf import *  # noqa: F403
from xorq.config import options
from xorq.expr import api
from xorq.expr.api import *  # noqa: F403
from xorq import ml
from xorq.ml import *  # noqa: F403
from xorq.backends.let import Backend
from xorq.internal import SessionConfig

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

__all__ = [  # noqa: PLE0604
    "api",
    "examples",
    "flight",
    "connect",
    "options",
    "SessionConfig",
    "udf",
    *api.__all__,
    *ml.__all__,
    *udf.__all__,
]


def _load_entry_points():
    eps = importlib_metadata.entry_points(group="xorq.backends")
    return sorted(eps)


def load_backend(name):
    if entry_point := next(
        (ep for ep in _load_entry_points() if ep.name == name), None
    ):
        import types

        import xorq as xo

        module = entry_point.load()
        backend = module.Backend()
        backend.register_options()

        def connect(*args, **kwargs):
            return backend.connect(*args, **kwargs)

        connect.__doc__ = backend.do_connect.__doc__
        connect.__wrapped__ = backend.do_connect
        connect.__module__ = f"xorq.{name}"

        proxy = types.ModuleType(f"xorq.{name}")
        setattr(xo, name, proxy)
        proxy.connect = connect
        proxy.compile = backend.compile
        proxy.has_operation = backend.has_operation
        proxy.name = name
        proxy._from_url = backend._from_url
        # Add any additional methods that should be exposed at the top level
        for attr in getattr(backend, "_top_level_methods", ()):
            setattr(proxy, attr, getattr(backend, attr))

        return proxy


def connect(session_config: SessionConfig | None = None) -> Backend:
    """Create a xorq backend."""
    instance = Backend()
    instance.do_connect(session_config)
    return instance


from xorq import flight  # noqa: E402


def __getattr__(name):
    from xorq.vendor import ibis

    return load_backend(name) or ibis.load_backend(name)


__version__ = importlib_metadata.version(__package__)
