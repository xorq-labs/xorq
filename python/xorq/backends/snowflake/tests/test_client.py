from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import toolz

import xorq.api as xo
from xorq.backends.snowflake import Backend as SnowflakeBackend
from xorq.backends.snowflake.enums import SnowflakeAuthenticator
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
    sf_con: SnowflakeBackend,
    temp_catalog: str,
    temp_db: str,
    temp_catalog2: str,
    temp_db2: str,
    csv_dir: Path,
) -> None:
    with inside_temp_schema(sf_con, temp_catalog, temp_db):
        name = "batting"
        t = xo.deferred_read_csv(
            csv_dir.joinpath(f"{name}.csv"), table_name=name
        ).into_backend(sf_con)
        name = gen_name(t.get_name())
        assert not sf_con.list_tables()
        sf_con.create_table(name, t, database=f"{temp_catalog2}.{temp_db2}")
        assert name in sf_con.list_tables(database=(temp_catalog2, temp_db2))


def test_create_table_requires_obj_or_schema() -> None:
    # Guard arm: neither obj nor schema raises before any SQL, so it needs no
    # live connection. Covers the `obj is None and schema is None` branch.
    with pytest.raises(ValueError, match="Either `obj` or `schema`"):
        SnowflakeBackend().create_table(gen_name("t"))


def test_create_table_rejects_unsupported_obj_type() -> None:
    # The default arm delegates to api.memtable, which rejects a bare scalar
    # before any SQL -- so memtable, not a create_table guard, owns rejection.
    with pytest.raises(ValueError, match="DataFrame constructor"):
        SnowflakeBackend().create_table(gen_name("t"), obj=42)


@pytest.mark.snowflake
def test_create_table_from_dataframe(
    sf_con: SnowflakeBackend, temp_catalog: str, temp_db: str
) -> None:
    # The `pd.DataFrame() | pa.Table()` arm wraps obj in a memtable (scope stays
    # None, so the finally is a no-op). Live-only: it issues a real CREATE.
    with inside_temp_schema(sf_con, temp_catalog, temp_db):
        name = gen_name("df_table")
        sf_con.create_table(name, pd.DataFrame({"a": [1, 2, 3]}))
        assert name in sf_con.list_tables()
        assert sf_con.table(name).count().execute() == 3


@pytest.mark.snowflake
def test_create_table_from_schema_only(
    sf_con: SnowflakeBackend, temp_catalog: str, temp_db: str
) -> None:
    # The `None` arm: schema-only create yields table=None (empty table, no
    # populating query). Live-only: it issues a real CREATE.
    with inside_temp_schema(sf_con, temp_catalog, temp_db):
        name = gen_name("schema_only")
        sf_con.create_table(name, schema=xo.schema({"a": "int64"}))
        assert name in sf_con.list_tables()
        assert sf_con.table(name).count().execute() == 0
