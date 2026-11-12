import importlib
import importlib.metadata

import pandas as pd
import pytest
import toolz
from packaging import version

from xorq.backends.snowflake import Backend as SnowflakeBackend
from xorq.backends.snowflake.enums import SnowflakeAuthenticator
from xorq.common.utils.env_utils import maybe_substitute_env_var
from xorq.common.utils.ibis_utils import (
    from_ibis,
    ibis_utils_backend_registry_context,
    map_ibis,
)


ibis = pytest.importorskip("ibis")

# ibis registers its builtin udfs with input_types=, which datafusion 52
# renamed to input_fields
datafusion_renamed_input_types = bool(importlib.util.find_spec("datafusion")) and (
    version.parse(importlib.metadata.version("datafusion")) >= version.parse("52")
)


def connect_snowflake():
    from xorq.common.utils.snowflake_keypair_utils import (  # noqa: PLC0415
        maybe_decrypt_private_key,
    )
    from xorq.common.utils.snowflake_utils import (  # noqa: PLC0415
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


# No postgres case: from_ibis over an ibis postgres connection is not covered.
# The ibis-compatibility jobs start no postgres server.
@pytest.mark.parametrize(
    "get_con,backend_name",
    (
        pytest.param(
            lambda: ibis.datafusion.connect(),
            "datafusion",
            id="datafusion",
            marks=pytest.mark.xfail(
                datafusion_renamed_input_types,
                raises=TypeError,
                reason="ibis datafusion backend predates datafusion 52",
                strict=True,
            ),
        ),
        pytest.param(lambda: ibis.sqlite.connect(), "sqlite", id="sqlite"),
    ),
)
def test_backends(get_con, backend_name):
    conn = get_con()
    ta = conn.create_table("foo", pd.DataFrame({"id": [1, 2, 3]}))
    expr = ta.select(ibis._.id).filter(ibis._.id > 2)

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None
    assert not xorq_expr.execute().empty
    assert xorq_expr._find_backend().name == backend_name

    try:
        conn.drop_table("foo")
    except Exception as e:
        print(e)


@pytest.mark.skipif(
    importlib.util.find_spec("snowflake") is None, reason="requires snowflake"
)
@pytest.mark.snowflake
def test_snowflake_backend():
    snow_conn = connect_snowflake()
    assert snow_conn.list_tables() is not None
    with ibis_utils_backend_registry_context():
        xorq_con = map_ibis(snow_conn, None)
    assert isinstance(xorq_con, SnowflakeBackend)
    assert xorq_con.list_tables() is not None
