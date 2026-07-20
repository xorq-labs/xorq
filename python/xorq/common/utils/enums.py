from __future__ import annotations

from xorq.common.compat import StrEnum


class IngestMode(StrEnum):
    """ADBC bulk-ingest modes accepted by ``adbc_ingest``.

    These four modes are driver-agnostic ADBC concepts shared by every ADBC
    backend (see ``xorq.common.utils.adbc_utils.ADBCBase``), so the enum lives
    here rather than in any single backend package.
    """

    CREATE = "create"
    APPEND = "append"
    REPLACE = "replace"
    CREATE_APPEND = "create_append"
