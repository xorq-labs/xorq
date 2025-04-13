import contextlib
import itertools
import warnings
from typing import Any

import pandas as pd
import pyarrow as pa
import sqlglot as sg
import sqlglot.expressions as sge

import xorq.vendor.ibis.expr.schema as sch
import xorq.vendor.ibis.expr.types as ir
from xorq.common.utils.logging_utils import get_logger
from xorq.expr.relations import replace_cache_table
from xorq.vendor.ibis.backends.snowflake import _SNOWFLAKE_MAP_UDFS
from xorq.vendor.ibis.backends.snowflake import Backend as IbisSnowflakeBackend
from xorq.vendor.ibis.expr.operations.relations import (
    Namespace,
)


logger = get_logger(__name__)


class Backend(IbisSnowflakeBackend):
    _top_level_methods = ("connect_env",)

    @classmethod
    def connect_env(cls, database="SNOWFLAKE_SAMPLE_DATA", schema="TPCH_SF1", **kwargs):
        from xorq.common.utils.snowflake_utils import make_connection

        return make_connection(database=database, schema=schema, **kwargs)

    def _to_sqlglot(
        self, expr: ir.Expr, *, limit: str | None = None, params=None, **_: Any
    ):
        op = expr.op()
        out = op.map_clear(replace_cache_table)

        return super()._to_sqlglot(out.to_expr(), limit=limit, params=params)

    def table(self, *args, **kwargs):
        table = super().table(*args, **kwargs)
        op = table.op()
        if op.namespace == Namespace(None, None):
            (catalog, database) = (self.current_catalog, self.current_database)
            table = op.copy(**{"namespace": Namespace(catalog, database)}).to_expr()
        return table

    def create_table(
        self,
        name: str,
        obj: pd.DataFrame | pa.Table | ir.Table | None = None,
        *,
        schema: sch.Schema | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
        comment: str | None = None,
    ) -> ir.Table:
        """Create a table in Snowflake.

        Parameters
        ----------
        name
            Name of the table to create
        obj
            The data with which to populate the table; optional, but at least
            one of `obj` or `schema` must be specified
        schema
            The schema of the table to create; optional, but at least one of
            `obj` or `schema` must be specified
        database
            The name of the database in which to create the table; if not
            passed, the current database is used.
        temp
            Create a temporary table
        overwrite
            If `True`, replace the table if it already exists, otherwise fail
            if the table exists
        comment
            Add a comment to the table

        """
        if obj is None and schema is None:
            raise ValueError("Either `obj` or `schema` must be specified")

        quoted = self.compiler.quoted

        if database is None:
            target = sg.table(name, quoted=quoted)
            catalog = db = database
        else:
            db = self._warn_and_create_table_loc(database=database)
            (catalog, db) = (db.catalog, db.db)
            target = sg.table(name, db=db, catalog=catalog, quoted=quoted)

        column_defs = [
            sge.ColumnDef(
                this=sg.to_identifier(name, quoted=quoted),
                kind=self.compiler.type_mapper.from_ibis(typ),
                constraints=(
                    None
                    if typ.nullable
                    else [sge.ColumnConstraint(kind=sge.NotNullColumnConstraint())]
                ),
            )
            for name, typ in (schema or {}).items()
        ]

        if column_defs:
            target = sge.Schema(this=target, expressions=column_defs)

        properties = []

        if temp:
            properties.append(sge.TemporaryProperty())

        if comment is not None:
            properties.append(sge.SchemaCommentProperty(this=sge.convert(comment)))

        if obj is not None:
            if not isinstance(obj, ir.Expr):
                import xorq as xo

                table = xo.memtable(obj)
            else:
                table = obj

            self._run_pre_execute_hooks(table)

            query = self.compiler.to_sqlglot(table)
        else:
            query = None

        create_stmt = sge.Create(
            kind="TABLE",
            this=target,
            replace=overwrite,
            properties=sge.Properties(expressions=properties),
            expression=query,
        )

        with self._safe_raw_sql(create_stmt):
            pass

        return self.table(name, database=(catalog, db))

    def _setup_session(self, *, session_parameters, create_object_udfs: bool):
        con = self.con

        # enable multiple SQL statements by default
        session_parameters.setdefault("MULTI_STATEMENT_COUNT", 0)
        # don't format JSON output by default
        session_parameters.setdefault("JSON_INDENT", 0)

        # overwrite session parameters that are required for ibis + snowflake
        # to work
        session_parameters.update(
            dict(
                # Use Arrow for query results
                PYTHON_CONNECTOR_QUERY_RESULT_FORMAT="arrow_force",
                # JSON output must be strict for null versus undefined
                STRICT_JSON_OUTPUT=True,
                # Timezone must be UTC
                TIMEZONE="UTC",
            ),
        )

        with contextlib.closing(con.cursor()) as cur:
            cur.execute(
                "ALTER SESSION SET {}".format(
                    " ".join(f"{k} = {v!r}" for k, v in session_parameters.items())
                )
            )

        if create_object_udfs:
            dialect = self.name
            create_stmt = sge.Create(
                kind="DATABASE", this="ibis_udfs", exists=True
            ).sql(dialect)
            if "/" in con.database:
                (catalog, db) = con.database.split("/")
                use_stmt = sge.Use(
                    kind="SCHEMA",
                    this=sg.table(db, catalog=catalog, quoted=self.compiler.quoted),
                ).sql(dialect)
            else:
                use_stmt = ""

            stmts = [
                create_stmt,
                # snowflake activates a database on creation, so reset it back
                # to the original database and schema
                use_stmt,
                *itertools.starmap(self._make_udf, _SNOWFLAKE_MAP_UDFS.items()),
            ]

            stmt = ";\n".join(stmts)
            with contextlib.closing(con.cursor()) as cur:
                try:
                    cur.execute(stmt)
                except Exception as e:  # noqa: BLE001
                    warnings.warn(
                        f"Unable to create Ibis UDFs, some functionality will not work: {e}"
                    )
        # without this self.current_{catalog,database} is not synchronized with con.{database,schema}
        with contextlib.closing(con.cursor()) as cur:
            try:
                cur.execute("SELECT CURRENT_TIME")
            except Exception:  # noqa: BLE001
                pass

    def read_record_batches(
        self,
        record_batches: pa.RecordBatchReader,
        table_name: str | None = None,
        password: str | None = None,
        temporary: bool = False,
        mode: str = "create",
        **kwargs: Any,
    ) -> ir.Table:
        from xorq.common.utils.snowflake_utils import SnowflakeADBC

        logger.info(
            "reading record batches with SnowflakeADBC",
            **{
                "table_name": table_name,
                "temporary": temporary,
                "mode": mode,
                **kwargs,
            },
        )

        snowflake_adbc = SnowflakeADBC(self, password)
        snowflake_adbc.adbc_ingest(table_name, record_batches, mode=mode, **kwargs)
        return self.table(table_name)
