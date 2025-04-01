import os

import pytest


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


def get_storage_uncached(expr):
    assert expr.ls.is_cached
    storage = expr.ls.storage
    uncached = expr.ls.uncached_one
    return (storage, uncached)
