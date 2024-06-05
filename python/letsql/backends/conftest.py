import os

import pytest

from letsql.expr.relations import CachedNode


snowflake_credentials_varnames = (
    "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_USER",
)
have_snowflake_credentials = all(
    os.environ.get(varname) for varname in snowflake_credentials_varnames
)


def pytest_runtest_setup(item):
    if any(mark.name == "snowflake" for mark in item.iter_markers()):
        pytest.importorskip("snowflake.connector")
        if not have_snowflake_credentials:
            pytest.skip("cannot run snowflake tests without snowflake creds")


def get_storage_uncached(con, expr):
    op = expr.op()
    assert isinstance(op, CachedNode)

    def replace_table(node, _, **_kwargs):
        return con._sources.get_table_or_op(node, node.__recreate__(_kwargs))

    uncached = expr.op().replace(replace_table).parent.to_expr()
    return (op.storage, uncached)
