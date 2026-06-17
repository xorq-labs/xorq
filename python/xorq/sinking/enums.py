from __future__ import annotations

from xorq.common.compat import StrEnum


class SinkMode(StrEnum):
    CREATE = "create"
    APPEND = "append"
