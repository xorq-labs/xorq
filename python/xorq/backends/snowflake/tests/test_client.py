import pytest
import toolz

import xorq.api as xo
from xorq.backends.snowflake import SnowflakeAuthenticator
from xorq.backends.snowflake.tests.conftest import (
    inside_temp_schema,
)
from xorq.vendor.ibis.util import gen_name


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


@pytest.mark.snowflake
def test_create_table_from_expr_failure(sf_con, temp_catalog, temp_db, csv_dir):
    name = "batting"
    t = xo.deferred_read_csv(csv_dir.joinpath(f"{name}.csv"), table_name=name)
    name = gen_name(t.get_name())
    with pytest.raises(ValueError, match="expr backend must be .*, is"):
        sf_con.create_table(name, t, database=f"{temp_catalog}.{temp_db}")


@pytest.mark.snowflake
def test_create_table_from_expr_success(sf_con, temp_catalog, temp_db, csv_dir):
    with inside_temp_schema(sf_con, temp_catalog, temp_db):
        name = "batting"
        t = xo.deferred_read_csv(
            csv_dir.joinpath(f"{name}.csv"), table_name=name
        ).into_backend(sf_con)
        name = gen_name(t.get_name())
        sf_con.create_table(name, t)
        assert name in sf_con.list_tables()


@pytest.mark.snowflake
def test_create_table_from_expr_other(
    sf_con, temp_catalog, temp_db, temp_catalog2, temp_db2, csv_dir
):
    with inside_temp_schema(sf_con, temp_catalog, temp_db):
        name = "batting"
        t = xo.deferred_read_csv(
            csv_dir.joinpath(f"{name}.csv"), table_name=name
        ).into_backend(sf_con)
        name = gen_name(t.get_name())
        assert not sf_con.list_tables()
        sf_con.create_table(name, t, database=f"{temp_catalog2}.{temp_db2}")
        assert name in sf_con.list_tables(database=(temp_catalog2, temp_db2))
