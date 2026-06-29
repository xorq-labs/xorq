from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

import pyarrow as pa
import sqlglot as sg
import sqlglot.expressions as sge
import toolz
from adbc_driver_manager import ProgrammingError as ADBCProgrammingError

import xorq.vendor.ibis.expr.schema as sch
from xorq.backends.postgres.compiler import compiler
from xorq.common.utils.defer_utils import (
    read_csv_rbr,
)
from xorq.common.utils.logging_utils import get_logger
from xorq.config import default_backend
from xorq.vendor.ibis import util
from xorq.vendor.ibis.backends.postgres import Backend as IbisPostgresBackend
from xorq.vendor.ibis.expr import types as ir
from xorq.vendor.ibis.util import (
    gen_name,
)


logger = get_logger(__name__)


__all__ = [
    "Backend",
    "connect",
]


class Backend(IbisPostgresBackend):
    _top_level_methods = ("connect_examples", "connect_env")
    compiler = compiler

    @classmethod
    def connect_env(cls, **kwargs):
        from xorq.common.utils.postgres_utils import make_connection  # noqa: PLC0415

        return make_connection(**kwargs)

    @classmethod
    def connect_examples(cls, **kwargs):
        examples_kwargs = {
            "host": "examples.letsql.com",
            "user": "letsql",
            "password": "letsql",
            "database": "letsql",
        }
        return cls().connect(**(examples_kwargs | kwargs))

    def _build_insert_template(
        self,
        name,
        *,
        schema: sch.Schema,
        catalog: str | None = None,
        columns: bool = False,
        placeholder: str = "?",
    ) -> str:
        """Builds an INSERT INTO table VALUES query string with placeholders.

        Parameters
        ----------
        name
            Name of the table to insert into
        schema
            Ibis schema of the table to insert into
        catalog
            Catalog name of the table to insert into
        columns
            Whether to render the columns to insert into
        placeholder
            Placeholder string. Can be a format string with a single `{i}` spec.

        Returns
        -------
        str
            The query string
        """
        quoted = self.compiler.quoted
        return sge.insert(
            sge.Values(
                expressions=[
                    sge.Tuple(
                        expressions=[
                            sge.Var(this=placeholder.format(i=i))
                            for i in range(len(schema))
                        ]
                    )
                ]
            ),
            into=sg.table(name, catalog=catalog, quoted=quoted),
            columns=(
                map(partial(sg.to_identifier, quoted=quoted), schema.keys())
                if columns
                else None
            ),
        ).sql(self.dialect)

    @util.experimental
    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
        **_: Any,
    ) -> pa.ipc.RecordBatchReader:
        from xorq.common.utils.postgres_utils import PgADBC  # noqa: PLC0415

        def _batches(self, *, pyarrow_schema, struct_type, query):
            # Primary path: ADBC opens its own independent postgres connection
            # per call, so concurrent generators never share psycopg connection
            # state (eliminates OutOfOrderTransactionNesting).
            try:
                adbc_con = PgADBC(self).get_conn()
            except Exception:
                # A genuine config/auth error is indistinguishable here from a
                # missing ADBC URI; both fall through to the psycopg path, so log
                # at debug to keep the real cause diagnosable.
                logger.debug(
                    "ADBC connection unavailable; falling back to psycopg",
                    backend=self.name,
                    exc_info=True,
                )
                adbc_con = None

            if adbc_con is not None:
                cur = adbc_con.cursor()
                # ``fall_through`` distinguishes the one recoverable failure
                # (temp table invisible on a fresh ADBC connection) from every
                # other error. The ``finally`` always closes both the cursor
                # and connection, so a non-ADBCProgrammingError raised by
                # ``execute`` (network, syntax, permission) propagates without
                # leaking the ADBC resources.
                fall_through = False
                try:
                    try:
                        cur.execute(query)
                    except ADBCProgrammingError:
                        fall_through = True
                    if not fall_through:
                        for batch in cur.fetch_record_batch():
                            yield batch.cast(pyarrow_schema)
                finally:
                    cur.close()
                    adbc_con.close()
                if not fall_through:
                    return

            # Psycopg fallback: session-local temp tables or no ADBC URI.
            con = self.con
            with (
                con.cursor(name=util.gen_name("postgres_cursor")) as cursor,
                con.transaction(),
            ):
                cur = cursor.execute(query)
                while batch := cur.fetchmany(chunk_size):
                    yield pa.RecordBatch.from_struct_array(
                        pa.array(batch, type=struct_type)
                    )

        self._run_pre_execute_hooks(expr)

        raw_schema = expr.as_table().schema()
        query = self.compile(expr, limit=limit, params=params)
        pyarrow_schema = raw_schema.to_pyarrow()
        return pa.RecordBatchReader.from_batches(
            pyarrow_schema,
            _batches(
                self,
                pyarrow_schema=pyarrow_schema,
                struct_type=raw_schema.as_struct().to_pyarrow(),
                query=query,
            ),
        )

    def read_record_batches(
        self,
        record_batches: pa.RecordBatchReader,
        table_name: str | None = None,
        password: str | None = None,
        temporary: bool = False,
        mode: str = "create",
        **kwargs: Any,
    ) -> ir.Table:
        from xorq.common.utils.postgres_utils import (  # noqa: PLC0415
            PgADBC,
            make_table_temporary,
        )

        pgadbc = PgADBC(self)
        pgadbc.adbc_ingest(table_name, record_batches, mode=mode, **kwargs)
        if temporary:
            make_table_temporary(self, table_name)
        return self.table(table_name)

    def read_parquet(
        self,
        path: str | Path,
        table_name: str | None = None,
        password: str | None = None,
        temporary: bool = False,
        mode: str = "create",
        **kwargs: Any,
    ) -> ir.Table:
        if table_name is None:
            if not temporary:
                raise ValueError(
                    "If `table_name` is not provided, `temporary` must be True"
                )
            else:
                table_name = gen_name("ls-read-parquet")
        record_batches = default_backend().read_parquet(path).to_pyarrow_batches()
        return self.read_record_batches(
            record_batches=record_batches,
            table_name=table_name,
            password=password,
            temporary=temporary,
            mode=mode,
            **kwargs,
        )

    def read_csv(
        self,
        path,
        table_name=None,
        chunksize=10_000,
        password=None,
        temporary=False,
        mode="create",
        schema=None,
        **kwargs,
    ):
        if chunksize is None:
            raise ValueError("chunksize must not be None")
        if table_name is None:
            if not temporary:
                raise ValueError(
                    "If `table_name` is not provided, `temporary` must be True"
                )
            else:
                table_name = gen_name("ls-read-csv")
        record_batches = read_csv_rbr(path, schema=schema, **kwargs)
        return self.read_record_batches(
            record_batches=record_batches,
            table_name=table_name,
            password=password,
            temporary=temporary,
            mode=mode,
            **kwargs,
        )

    def create_catalog(self, name: str, force: bool = False) -> None:
        # https://stackoverflow.com/a/43634941
        if force:
            raise ValueError("postgres does not support force=True for create_catalog")
        quoted = self.compiler.quoted
        create_stmt = sge.Create(
            this=sg.to_identifier(name, quoted=quoted), kind="DATABASE", exists=force
        )
        (prev_autocommit, self.con.autocommit) = (self.con.autocommit, True)
        with self._safe_raw_sql(create_stmt):
            pass
        self.con.autocommit = prev_autocommit

    def clone(self, password=None, **kwargs):
        """necessary because "UnsupportedOperationError: postgres does not support creating a database in a different catalog" """
        from xorq.common.utils.postgres_utils import (  # noqa: PLC0415
            make_credential_defaults,  # noqa: PLC0415
        )

        password = password or make_credential_defaults()["password"]
        if password is None:
            raise ValueError(
                "password is required if POSTGRES_PASSWORD env var is not populated"
            )
        dsn_parameters = self.con.info.get_parameters()
        dct = {
            **toolz.dissoc(
                dsn_parameters,
                "dbname",
                "options",
            ),
            **{
                "database": dsn_parameters["dbname"],
                "password": password,
            },
            **kwargs,
        }
        return connect(**dct)


def connect(**kwargs):
    con = Backend()
    Backend.connect(**kwargs)
    return con
