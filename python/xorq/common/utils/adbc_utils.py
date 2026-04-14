from abc import ABC, abstractmethod


class ADBCBase(ABC):
    """Mixin base class for ADBC-backed ingestion helpers.

    Subclasses must implement ``get_conn()`` returning an ADBC connection
    context manager.  The ``adbc_ingest`` method is provided here so the
    identical implementation is not duplicated across every backend helper.
    """

    __slots__ = ()

    @abstractmethod
    def get_conn(self, **kwargs): ...

    def adbc_ingest(
        self, table_name, record_batch_reader, mode="create", temporary=False, **kwargs
    ):
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.adbc_ingest(
                    table_name,
                    record_batch_reader,
                    mode=mode,
                    temporary=temporary,
                    **kwargs,
                )
            conn.commit()
