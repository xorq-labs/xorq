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
from xorq import caching
from xorq.caching import *  # noqa: F403
from xorq import ml
from xorq.ml import *  # noqa: F403
from xorq.backends.xorq_datafusion import Backend
from xorq.internal import SessionConfig

from xorq.loader import load_backend
from xorq.ibis_yaml.compiler import (
    build_expr,
    load_expr,
)
from xorq.common.utils.graph_utils import (
    replace_sources,
)


__all__ = [  # noqa: PLE0604
    "api",
    "caching",
    "catalog",
    "examples",
    "expr",
    "flight",
    "config",
    "connect",
    "options",
    "SessionConfig",
    "udf",
    "build_expr",
    "load_expr",
    "replace_sources",
    *api.__all__,
    *caching.__all__,
    *ml.__all__,
    *udf.__all__,
]


def connect(session_config: SessionConfig | None = None) -> Backend:
    """Create a xorq backend."""
    instance = Backend()
    instance.do_connect(session_config)
    return instance


import xorq.catalog.api as catalog  # noqa: E402
from xorq import flight  # noqa: E402


def __getattr__(name):
    from xorq.vendor import ibis  # noqa: PLC0415

    backend = load_backend(name) or ibis.load_backend(name)

    setattr(sys.modules[__name__], name, backend)

    return backend
