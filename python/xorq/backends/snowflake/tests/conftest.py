import types
from contextlib import contextmanager

import pytest
import sqlglot as sg
import sqlglot.expressions as sge

import xorq as xo
from xorq.vendor.ibis.util import gen_name


SU = pytest.importorskip("xorq.common.utils.snowflake_utils")


@pytest.fixture(scope="session")
def sf_con():
    return xo.snowflake.connect(
        # a database/schema we can trust exists
        database="SNOWFLAKE_SAMPLE_DATA",
        schema="TPCH_SF1",
        **SU.make_credential_defaults(),
        **SU.make_connection_defaults(),
        create_object_udfs=False,
    )


@pytest.fixture
def temp_catalog(sf_con):
    cat = gen_name("tmp_catalog")

    sf_con.create_catalog(cat)
    assert cat in sf_con.list_catalogs()

    yield cat

    sf_con.drop_catalog(cat)
    assert cat not in sf_con.list_catalogs()


@pytest.fixture
def temp_db(sf_con, temp_catalog):
    database = gen_name("tmp_database")

    sf_con.create_database(database, catalog=temp_catalog)
    assert database in sf_con.list_databases(catalog=temp_catalog)

    yield database

    sf_con.drop_database(database, catalog=temp_catalog)
    assert database not in sf_con.list_databases(catalog=temp_catalog)


@contextmanager
def inside_temp_schema(con, temp_catalog, temp_db):
    (prev_catalog, prev_db) = (con.current_catalog, con.current_database)
    con.raw_sql(
        sge.Use(
            kind="SCHEMA", this=sg.table(temp_db, db=temp_catalog, quoted=True)
        ).sql(dialect=con.name),
    )
    try:
        yield
    finally:
        con.raw_sql(
            sge.Use(
                kind="SCHEMA", this=sg.table(prev_db, db=prev_catalog, quoted=True)
            ).sql(dialect=con.name),
        )


class MockSnowflakeADBC:
    def __init__(self, snowflake_adbc, database=None, schema=None):
        self.snowflake_adbc = snowflake_adbc
        self.database = database
        self.schema = schema

    def adbc_ingest(
        self, table_name, record_batch_reader, mode="create", temporary=False, **kwargs
    ):
        with self.snowflake_adbc.get_conn(
            database=self.database, schema=self.schema
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'USE SCHEMA "{self.snowflake_adbc.con.current_catalog}"."{self.snowflake_adbc.con.current_database}"'
                )
                self.snowflake_adbc._adbc_ingest(
                    cur,
                    table_name,
                    record_batch_reader,
                    mode=mode,
                    temporary=temporary,
                    **kwargs,
                )
            # must commit!
            conn.commit()


@contextmanager
def mock_snowflake_adbc(con, database, schema):
    def read_record_batches(
        self,
        record_batches,
        table_name: str | None = None,
        password: str | None = None,
        temporary: bool = False,
        mode: str = "create",
        **kwargs,
    ):
        from xorq.common.utils.snowflake_utils import SnowflakeADBC

        snowflake_adbc = MockSnowflakeADBC(
            SnowflakeADBC(self, password), database=database, schema=schema
        )
        snowflake_adbc.adbc_ingest(table_name, record_batches, mode=mode, **kwargs)
        return self.table(table_name)

    prev_record_batches = con.read_record_batches
    con.read_record_batches = types.MethodType(read_record_batches, con)

    yield con

    con.read_record_batches = prev_record_batches
