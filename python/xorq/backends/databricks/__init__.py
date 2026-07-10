"""Databricks backend for xorq."""

from __future__ import annotations

from xorq.backends.databricks.backend import Backend, connect


__all__ = [
    "Backend",
    "connect",
]
