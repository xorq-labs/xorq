"""Initialize LetSQL module."""

from __future__ import annotations

from letsql.backends.let import Backend

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata


def connect() -> Backend:
    """Create a LETSQL backend."""
    instance = Backend()
    instance.do_connect()
    return instance


__version__ = importlib_metadata.version(__name__)
