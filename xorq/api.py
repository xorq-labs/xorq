"""Initialize xorq module."""

from __future__ import annotations

import sys

from xorq import config, examples, expr, ml, udf
from xorq.backends.let import Backend
from xorq.config import options
from xorq.expr import api
from xorq.expr.api import *  # noqa: F403
from xorq.internal import SessionConfig
from xorq.loader import load_backend
from xorq.ml import *  # noqa: F403
from xorq.udf import *  # noqa: F403


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
    from xorq.vendor import ibis

    backend = load_backend(name) or ibis.load_backend(name)

    setattr(sys.modules[__name__], name, backend)

    return backend
