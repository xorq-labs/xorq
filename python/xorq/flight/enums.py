from __future__ import annotations

from xorq.common.compat import StrEnum


class UploadMode(StrEnum):
    """How a Flight upload writes to the server's table.

    Flight-only: ``xorq.writes.enums.WriteMode`` (used by write-throughs and
    the backends' ``read_record_batches``) has no ``replace``.
    """

    CREATE = "create"
    APPEND = "append"
    REPLACE = "replace"
