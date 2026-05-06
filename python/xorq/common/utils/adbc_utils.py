from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ADBCBase(ABC):
    """Mixin base class for ADBC-backed ingestion helpers.

    Subclasses must implement ``get_conn()`` returning an ADBC connection
    context manager.  The ``adbc_ingest`` method is provided here so the
    identical implementation is not duplicated across every backend helper.
    """

    __slots__ = ()

    @abstractmethod
    def get_conn(self, **kwargs: Any) -> Any: ...

    def adbc_ingest(
        self,
        table_name: str,
        record_batch_reader: Any,
        mode: str = "create",
        temporary: bool = False,
        conn_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        with self.get_conn(**(conn_kwargs or {})) as conn:
            with conn.cursor() as cur:
                cur.adbc_ingest(
                    table_name,
                    record_batch_reader,
                    mode=mode,
                    temporary=temporary,
                    **kwargs,
                )
            conn.commit()
