"""Initialize LetSQL module."""

from __future__ import annotations


from letsql.backends.let import Backend


def connect() -> Backend:
    """Create a LETSQL backend."""
    instance = Backend()
    instance.do_connect()
    return instance
