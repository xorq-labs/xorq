import pytest
import toolz

from xorq.backends.snowflake import SnowflakeAuthenticator


SU = pytest.importorskip("xorq.common.utils.snowflake_utils")


(database, schema) = ("SNOWFLAKE_SAMPLE_DATA", "TPCH_SF1")
make_connection = toolz.curry(
    SU.make_connection,
    authenticator=SnowflakeAuthenticator.keypair,
    database=database,
    schema=schema,
)


@pytest.mark.snowflake
def test_setup_session():
    con = make_connection(
        create_object_udfs=False,
    )
    dct = (
        con.raw_sql("SELECT CURRENT_WAREHOUSE(), CURRENT_DATABASE(), CURRENT_SCHEMA();")
        .fetch_pandas_all()
        .iloc[0]
        .to_dict()
    )
    expected = (database, schema)
    assert (con.current_catalog, con.current_database) == expected
    assert (con.con.database, con.con.schema) == expected
    assert dct == {
        "CURRENT_WAREHOUSE()": "COMPUTE_WH",
        "CURRENT_DATABASE()": database,
        "CURRENT_SCHEMA()": schema,
    }

    con = make_connection(
        create_object_udfs=True,
    )
    dct = (
        con.raw_sql("SELECT CURRENT_WAREHOUSE(), CURRENT_DATABASE(), CURRENT_SCHEMA();")
        .fetch_pandas_all()
        .iloc[0]
        .to_dict()
    )
    assert con.current_catalog == database
    assert con.current_database == schema
    assert con.con.database == database
    assert con.con.schema == schema
    assert dct == {
        "CURRENT_WAREHOUSE()": "COMPUTE_WH",
        "CURRENT_DATABASE()": database,
        "CURRENT_SCHEMA()": schema,
    }


@pytest.mark.snowflake
def test_table_namespace():
    con = make_connection()
    table = con.table("CUSTOMER")
    namespace = table.op().namespace
    assert namespace.catalog is not None
    assert namespace.database is not None
