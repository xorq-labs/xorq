from abc import ABC, abstractmethod


class ADBCBase(ABC):
    """Mixin base class for ADBC-backed ingestion helpers.

    Subclasses must implement ``get_conn()`` returning an ADBC connection
    context manager.  The ``adbc_ingest`` method is provided here so the
    identical implementation is not duplicated across every backend helper.
    """

    __slots__ = ()

    # appended to the capability error below; subclasses override with
    # backend-specific driver install instructions
    ingest_install_hint = ""

    @abstractmethod
    def get_conn(self, **kwargs): ...

    def adbc_ingest(
        self,
        table_name,
        record_batch_reader,
        mode="create",
        temporary=False,
        conn_kwargs=None,
        **kwargs,
    ):
        # deferred so importing this module doesn't require the ADBC extras
        from adbc_driver_manager import ProgrammingError  # noqa: PLC0415

        with self.get_conn(**(conn_kwargs or {})) as conn:
            with conn.cursor() as cur:
                # capability probe: a driver that cannot bulk ingest rejects
                # the standard target-table option key, locally and before any
                # data is bound or sent. Probing it ourselves keys the error
                # off the failing call rather than the driver's message text
                # (`cursor.adbc_ingest` re-sets the same option, so a passing
                # probe is harmless).
                try:
                    cur.adbc_statement.set_options(
                        **{"adbc.ingest.target_table": table_name}
                    )
                except ProgrammingError as e:
                    msg = "this ADBC driver build does not support bulk ingest"
                    if self.ingest_install_hint:
                        msg = f"{msg}; {self.ingest_install_hint}"
                    raise RuntimeError(msg) from e
                cur.adbc_ingest(
                    table_name,
                    record_batch_reader,
                    mode=mode,
                    temporary=temporary,
                    **kwargs,
                )
            conn.commit()
