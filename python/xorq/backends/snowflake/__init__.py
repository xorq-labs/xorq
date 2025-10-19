import contextlib
import functools
import itertools
import warnings
from typing import Any

import pandas as pd
import pyarrow as pa
import sqlglot as sg
import sqlglot.expressions as sge
import toolz

import xorq.vendor.ibis.expr.api as api
import xorq.vendor.ibis.expr.schema as sch
import xorq.vendor.ibis.expr.types as ir
from xorq.common.utils.logging_utils import get_logger
from xorq.expr.relations import (
    prepare_create_table_from_expr,
)
from xorq.vendor.ibis.backends.snowflake import _SNOWFLAKE_MAP_UDFS
from xorq.vendor.ibis.backends.snowflake import Backend as IbisSnowflakeBackend
from xorq.vendor.ibis.expr.operations.relations import (
    Namespace,
)


try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum


logger = get_logger(__name__)


class SnowflakeAuthenticator(StrEnum):
    # https://docs.snowflake.com/en/developer-guide/node-js/nodejs-driver-options#label-nodejs-auth-options
    password = "none"
    mfa = "username_password_mfa"
    keypair = "snowflake_jwt"
    sso = "externalbrowser"
    # oauth = "oauth"
    # oauth2 = "oauth_authorization_code"


@functools.wraps(IbisSnowflakeBackend.do_connect)
def wrapped_do_connect(self, create_object_udfs: bool = True, **kwargs: Any):
    from xorq.common.utils.snowflake_keypair_utils import maybe_decrypt_private_key

    if "private_key" in kwargs:
        kwargs = maybe_decrypt_private_key(kwargs)
    return IbisSnowflakeBackend.do_connect(
        self, create_object_udfs=create_object_udfs, **kwargs
    )


class Backend(IbisSnowflakeBackend):
    _top_level_methods = (
        "connect_env",
        "connect_env_mfa",
        "connect_env_password",
        "connect_env_keypair",
    )

    @staticmethod
    def connect_env(
        passcode=None,
        authenticator=None,
        database="SNOWFLAKE_SAMPLE_DATA",
        schema="TPCH_SF1",
        **kwargs,
    ):
        from xorq.common.utils.snowflake_utils import make_connection

        return make_connection(
            authenticator=authenticator,
            passcode=passcode,
            database=database,
            schema=schema,
            **kwargs,
        )

    connect_env_password = staticmethod(
        toolz.curry(connect_env, authenticator=SnowflakeAuthenticator.password)
    )

    connect_env_mfa = staticmethod(
        toolz.curry(connect_env, authenticator=SnowflakeAuthenticator.mfa)
    )

    connect_env_keypair = staticmethod(
        toolz.curry(connect_env, authenticator=SnowflakeAuthenticator.keypair)
    )

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
                table = api.memtable(obj)
            else:
                table = prepare_create_table_from_expr(self, obj)

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

    do_connect = wrapped_do_connect

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

        # snowflake activates a database on creation, so reset it back
        # to the original database and schema
        if con.database and "/" in con.database:
            (catalog, db) = con.database.split("/")
            use_stmt = sge.Use(
                kind="SCHEMA",
                this=sg.table(db, catalog=catalog, quoted=self.compiler.quoted),
            ).sql(dialect=self.name)
            with contextlib.closing(con.cursor()) as cur:
                try:
                    cur.execute(use_stmt)
                except Exception as e:  # noqa: BLE001
                    warnings.warn(f"Unable to set catalog,db: {e}")

        if create_object_udfs:
            create_stmt = sge.Create(
                kind="DATABASE", this="ibis_udfs", exists=True
            ).sql(dialect=self.name)

            stmts = [
                create_stmt,
                # snowflake activates a database on creation, so reset it back
                # to the original database and schema
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

    @property
    def adbc(self):
        from xorq.common.utils.snowflake_utils import SnowflakeADBC

        adbc = SnowflakeADBC(self)
        return adbc

    def read_record_batches(
        self,
        record_batches: pa.RecordBatchReader,
        table_name: str | None = None,
        temporary: bool = False,
        mode: str = "create",
        **kwargs: Any,
    ) -> ir.Table:
        logger.info(
            "reading record batches with SnowflakeADBC",
            **{
                "table_name": table_name,
                "temporary": temporary,
                "mode": mode,
                **kwargs,
            },
        )

        snowflake_adbc = self.adbc
        snowflake_adbc.adbc_ingest(table_name, record_batches, mode=mode, **kwargs)
        return self.table(table_name)


def connect(*args, **kwargs):
    con = Backend(*args, **kwargs)
    con.reconnect()
    return con
