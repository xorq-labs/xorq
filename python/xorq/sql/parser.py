from __future__ import annotations

from typing import Any

from xorq_datafusion._internal import parser


def __getattr__(name: str) -> Any:
    return getattr(parser, name)
