"""Detail strategy dispatch for kind-aware TUI panels.

Import this package to ensure all strategies are registered.
"""

import xorq.catalog.tui.detail.ml  # noqa: F401
import xorq.catalog.tui.detail.semantic  # noqa: F401

# Eagerly import strategy modules so @register_detail decorators fire.
import xorq.catalog.tui.detail.standard  # noqa: F401
from xorq.catalog.tui.detail.base import DetailStrategy
from xorq.catalog.tui.detail.dispatch import get_detail_strategy, register_detail


__all__ = [
    "DetailStrategy",
    "get_detail_strategy",
    "register_detail",
]
