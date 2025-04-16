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


def generate_mock_get_conn(original_get_conn, temp_catalog, temp_db):
    @contextmanager
    def mock_get_conn(self, **kwargs):
        kwargs["database"] = "SNOWFLAKE_SAMPLE_DATA"
        kwargs["schema"] = "TPCH_SF1"
        with original_get_conn(self, **kwargs) as conn:
            yield MockSnowFlakeADBCCon(conn, temp_catalog, temp_db)

    return mock_get_conn


class MockSnowFlakeADBCCon:
    def __init__(self, snowflake_adbc_con, temp_catalog, temp_db):
        self.snowflake_adbc_con = snowflake_adbc_con
        self.temp_catalog = temp_catalog
        self.temp_db = temp_db

    @contextmanager
    def cursor(self):
        with self.snowflake_adbc_con.cursor() as cur:
            cur.execute(f'USE SCHEMA "{self.temp_catalog}"."{self.temp_db}"')
            yield cur

    def commit(self):
        self.snowflake_adbc_con.commit()
