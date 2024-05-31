import contextlib
import functools
import itertools
import warnings

import dask
import ibis
import sqlglot as sg
import sqlglot.expressions as sge
from ibis.backends.snowflake import _SNOWFLAKE_MAP_UDFS


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
        create_stmt = sge.Create(kind="DATABASE", this="ibis_udfs", exists=True).sql(
            dialect
        )
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


@functools.cache
def monkeypatch_setup_session():
    attrname = "_setup_session"
    _setup_session.original = getattr(ibis.backends.snowflake.Backend, attrname)
    setattr(ibis.backends.snowflake.Backend, attrname, _setup_session)


def maybe_monkeypatch_setup_session():
    tokenized = dask.base.tokenize(ibis.backends.snowflake.Backend._setup_session)
    if tokenized == '8c96093dd6f2f759ff96fd41199f06f5':
        monkeypatch_setup_session()


maybe_monkeypatch_setup_session()
