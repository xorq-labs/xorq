from abc import ABC, abstractmethod

import sqlglot as sg
import sqlglot.expressions as sge

from xorq.common.utils.enums import IngestMode
from xorq.vendor.ibis import Schema
from xorq.vendor.ibis.util import gen_name


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


def staged_ingest(conn, dialect, table_name, record_batch_reader, mode):
    """Stream ``record_batch_reader`` into ``table_name`` over ``conn``, then
    commit it as one transaction; ``conn`` is closed after.

    The stream lands in a temporary table first, which nothing else can see
    or lock. Only after the last batch does the transaction touch
    ``table_name``, with the DDL and insert the buffered write would run: the
    table is declared from the upload's ibis schema (not the driver's type
    mapping), and the rows go in by column name with the database's
    assignment casts. So a slow upload holds no lock on the target, and a
    failure anywhere, the client going away included, leaves it untouched,
    provided the database makes DDL transactional (sqlite, postgres).

    ``mode`` is an ``IngestMode``: ``CREATE`` fails on an existing table,
    ``APPEND`` on an absent one, ``REPLACE`` drops and recreates, and
    ``CREATE_APPEND`` creates the table if it is absent.
    """
    schema = Schema.from_pyarrow(record_batch_reader.schema)
    target = sg.table(table_name, quoted=True)
    staging = sg.table(gen_name("upload"), quoted=True)
    columns = [sg.to_identifier(name, quoted=True) for name in schema.names]
    create = sge.Create(
        kind="TABLE",
        this=sge.Schema(this=target, expressions=schema.to_sqlglot(dialect)),
        exists=mode is IngestMode.CREATE_APPEND,
    )
    drop = sge.Drop(kind="TABLE", this=target, exists=True)
    ddl = {
        IngestMode.CREATE: (create,),
        IngestMode.APPEND: (),
        IngestMode.REPLACE: (drop, create),
        IngestMode.CREATE_APPEND: (create,),
    }[mode]
    insert = sge.insert(
        sg.select(*columns).from_(staging), into=target, columns=columns
    )
    with conn, conn.cursor() as cur:
        cur.adbc_ingest(
            staging.name, record_batch_reader, mode=IngestMode.CREATE, temporary=True
        )
        for statement in (*ddl, insert):
            cur.execute(statement.sql(dialect))
        conn.commit()
