from __future__ import annotations


try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum

__all__ = ["StrEnum"]
