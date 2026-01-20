"""Initialize xorq module."""

from __future__ import annotations

import sys

from xorq import examples
from xorq import udf
from xorq.udf import *  # noqa: F403
from xorq import config
from xorq import expr
from xorq.config import options


from xorq.expr import api
from xorq.expr.api import *  # noqa: F403
from xorq import ml
from xorq.ml import *  # noqa: F403
from xorq.backends.xorq import Backend
from xorq.internal import SessionConfig

from xorq.loader import load_backend

# Import catalog API - both namespace and convenience functions
from xorq import catalog_api as catalog
from xorq.catalog_api import (
    get as read_catalog,
    load_expr as read_build,
    get_placeholder as get_catalog_placeholder,
)

__all__ = [  # noqa: PLE0604
    "api",
    "examples",
    "expr",
    "flight",
    "config",
    "connect",
    "options",
    "SessionConfig",
    "udf",
    "catalog",
    "read_catalog",
    "read_build",
    "get_catalog_placeholder",
    *api.__all__,
    *ml.__all__,
    *udf.__all__,
]


def connect(session_config: SessionConfig | None = None) -> Backend:
    """Create a xorq backend."""
    instance = Backend()
    instance.do_connect(session_config)
    return instance


from xorq import flight  # noqa: E402


def __getattr__(name):
    # Provide helpful error messages for common mistakes
    _suggestions = {
        "get_catalog": "Did you mean xo.catalog.get() or xo.read_catalog()?",
        "load_catalog": "Did you mean xo.catalog.get() or xo.read_catalog()?",
        "catalog_get": "Did you mean xo.catalog.get() or xo.read_catalog()?",
        "load_build": "Did you mean xo.catalog.load_expr() or xo.read_build()?",
        "read_expr": "Did you mean xo.catalog.load_expr() or xo.read_build()?",
    }

    if name in _suggestions:
        raise AttributeError(
            f"module 'xorq' has no attribute '{name}'. {_suggestions[name]}\n\n"
            f"Available catalog functions:\n"
            f"  - xo.catalog.get('alias')           # Load from catalog\n"
            f"  - xo.catalog.load_expr('builds/...')  # Load from build dir\n"
            f"  - xo.catalog.get_placeholder('alias') # Get placeholder memtable\n"
            f"  - xo.read_catalog('alias')          # Alias for catalog.get()\n"
            f"  - xo.read_build('builds/...')       # Alias for catalog.load_expr()"
        )

    from xorq.vendor import ibis

    backend = load_backend(name) or ibis.load_backend(name)

    setattr(sys.modules[__name__], name, backend)

    return backend
