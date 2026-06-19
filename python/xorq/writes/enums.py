from __future__ import annotations

from xorq.common.compat import StrEnum


class WriteMode(StrEnum):
    CREATE = "create"
    APPEND = "append"
