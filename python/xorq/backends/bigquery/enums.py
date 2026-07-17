from __future__ import annotations

from xorq.common.compat import StrEnum


class IngestMode(StrEnum):
    """ADBC bulk-ingest modes accepted by ``adbc_ingest``."""

    CREATE = "create"
    APPEND = "append"
    REPLACE = "replace"
    CREATE_APPEND = "create_append"
