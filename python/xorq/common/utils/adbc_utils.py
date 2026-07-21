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
        from adbc_driver_manager import (  # noqa: PLC0415
            NotSupportedError,
            OperationalError,
            ProgrammingError,
        )

        with self.get_conn(**(conn_kwargs or {})) as conn:
            with conn.cursor() as cur:
                # capability probe: a driver that cannot bulk ingest rejects
                # the standard target-table option key, locally and before any
                # data is bound or sent. Probing it ourselves keys the error
                # off the failing call rather than the driver's message text
                # (`cursor.adbc_ingest` re-sets the same option, so a passing
                # probe is harmless). Drivers signal the unknown option in
                # different ways: NOT_IMPLEMENTED maps to NotSupportedError,
                # while the stale bigquery build reports INVALID_ARGUMENT
                # (ProgrammingError); catch the DatabaseError siblings so none
                # escape untranslated.
                try:
                    cur.adbc_statement.set_options(
                        **{"adbc.ingest.target_table": table_name}
                    )
                except (ProgrammingError, NotSupportedError, OperationalError) as e:
                    # only a genuine capability gap translates: the driver
                    # either maps the unknown option to NOT_IMPLEMENTED
                    # (NotSupportedError) or names the rejected `adbc.ingest`
                    # key. Anything else (e.g. a supporting driver rejecting a
                    # malformed target-table value) is a different failure and
                    # must propagate untranslated.
                    if not (
                        isinstance(e, NotSupportedError) or "adbc.ingest" in str(e)
                    ):
                        raise
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
