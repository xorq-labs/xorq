from functools import partial
from pathlib import Path
from typing import Any

import pyarrow as pa
import sqlglot as sg
import sqlglot.expressions as sge
import toolz

import xorq as xo
import xorq.vendor.ibis.expr.schema as sch
from xorq.backends.postgres.compiler import compiler
from xorq.common.utils.defer_utils import (
    read_csv_rbr,
)
from xorq.expr.relations import replace_cache_table
from xorq.vendor.ibis.backends.postgres import Backend as IbisPostgresBackend
from xorq.vendor.ibis.expr import types as ir
from xorq.vendor.ibis.util import (
    gen_name,
)


class Backend(IbisPostgresBackend):
    _top_level_methods = ("connect_examples", "connect_env")
    compiler = compiler

    @classmethod
    def connect_env(cls, **kwargs):
        from xorq.common.utils.postgres_utils import make_connection

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

    def _to_sqlglot(
        self, expr: ir.Expr, *, limit: str | None = None, params=None, **_: Any
    ):
        op = expr.op()
        out = op.map_clear(replace_cache_table)

        return super()._to_sqlglot(out.to_expr(), limit=limit, params=params)

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

    def read_record_batches(
        self,
        record_batches: pa.RecordBatchReader,
        table_name: str | None = None,
        password: str | None = None,
        temporary: bool = False,
        mode: str = "create",
        **kwargs: Any,
    ) -> ir.Table:
        from xorq.common.utils.postgres_utils import (
            PgADBC,
            make_table_temporary,
        )

        pgadbc = PgADBC(self, password)
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
        record_batches = xo.read_parquet(path).to_pyarrow_batches()
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
            raise ValueError
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
            raise ValueError
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
        from xorq.common.utils.postgres_utils import make_credential_defaults

        password = password or make_credential_defaults()["password"]
        if password is None:
            raise ValueError(
                "password is required if POSTGRES_PASSWORD env var is not populated"
            )
        dsn_parameters = self.con.get_dsn_parameters()
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
        con = xo.postgres.connect(**dct)
        return con
