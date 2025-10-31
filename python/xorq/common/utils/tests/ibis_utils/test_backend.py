import pandas as pd
import pytest
import toolz

from xorq.backends.datafusion import Backend as DatafusionBackend
from xorq.backends.postgres import Backend as PostgresBackend
from xorq.backends.snowflake import Backend as SnowflakeBackend
from xorq.backends.snowflake import SnowflakeAuthenticator
from xorq.backends.sqlite import Backend as SqliteBackend
from xorq.common.utils.env_utils import maybe_substitute_env_var
from xorq.common.utils.ibis_utils import from_ibis, map_ibis


ibis = pytest.importorskip("ibis")


def connect_postgres():
    from xorq.common.utils.postgres_utils import (
        make_connection_defaults,
        make_credential_defaults,
    )

    return ibis.postgres.connect(
        **toolz.valmap(
            maybe_substitute_env_var,
            (make_credential_defaults() | make_connection_defaults()),
        )
    )


def connect_snowflake():
    from xorq.common.utils.snowflake_keypair_utils import maybe_decrypt_private_key
    from xorq.common.utils.snowflake_utils import (
        make_connection_defaults,
        make_credential_defaults,
    )

    database = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"

    connection_defaults = toolz.valmap(
        maybe_substitute_env_var,
        (
            make_credential_defaults(authenticator=SnowflakeAuthenticator.keypair)
            | make_connection_defaults()
        ),
    )

    if "private_key" in connection_defaults:
        connection_defaults = maybe_decrypt_private_key(connection_defaults)

    return ibis.snowflake.connect(
        database=f"{database}/{schema}",
        **connection_defaults,
    )


@pytest.mark.parametrize(
    "get_con,backend_type",
    (
        (connect_postgres, PostgresBackend),
        (lambda: ibis.datafusion.connect(), DatafusionBackend),
        (lambda: ibis.sqlite.connect(), SqliteBackend),
    ),
)
def test_backends(get_con, backend_type):
    conn = get_con()
    ta = conn.create_table("foo", pd.DataFrame({"id": [1, 2, 3]}))
    expr = ta.select(ibis._.id).filter(ibis._.id > 2)

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None
    assert not xorq_expr.execute().empty
    assert isinstance(xorq_expr._find_backend(), backend_type)

    try:
        conn.drop_table("foo")
    except Exception as e:
        print(e)


@pytest.mark.snowflake
def test_snowflake_backend():
    snow_conn = connect_snowflake()
    assert snow_conn.list_tables() is not None
    xorq_con = map_ibis(snow_conn, None)
    assert isinstance(xorq_con, SnowflakeBackend)
    assert xorq_con.list_tables() is not None
