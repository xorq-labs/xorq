from __future__ import annotations

import contextlib
import uuid
from pathlib import Path

import pandas as pd
import pytest

from xorq.common.utils.postgres_utils import PostgresConfig


pytest.importorskip("pyiceberg")

from pyiceberg.exceptions import (  # noqa: E402
    NamespaceNotEmptyError,
    NoSuchNamespaceError,
    NoSuchTableError,
)
from sqlalchemy.engine import URL  # noqa: E402

from xorq.backends.pyiceberg import Backend as PyIcebergBackend  # noqa: E402


@pytest.fixture
def postgres_uri() -> str:
    config = PostgresConfig.from_env()
    return URL.create(
        "postgresql",
        username=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
        host=config.POSTGRES_HOST,
        port=int(config.POSTGRES_PORT),
        database=config.POSTGRES_DATABASE,
    ).render_as_string(hide_password=False)


def test_connect_env_with_a_postgres_catalog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, postgres_uri: str
) -> None:
    # the catalog outlives the warehouse in a reused database, so each run
    # gets its own catalog and namespace and drops both; pyiceberg's own
    # catalog tables (iceberg_tables, iceberg_namespace_properties) stay
    name = f"test_{uuid.uuid4().hex[:12]}"
    monkeypatch.setenv("ICEBERG_URI", postgres_uri)
    monkeypatch.setenv("ICEBERG_CATALOG_TYPE", "sql")
    monkeypatch.setenv("ICEBERG_WAREHOUSE_PATH", str(tmp_path / "warehouse"))
    monkeypatch.setenv("ICEBERG_NAMESPACE", name)
    monkeypatch.setenv("ICEBERG_CATALOG_NAME", name)

    con = PyIcebergBackend.connect_env()
    try:
        assert (con.catalog.name, con.catalog.properties["uri"]) == (
            name,
            postgres_uri,
        )
        t = con.create_table("trades", pd.DataFrame({"price": [1.0, 2.0]}))
        assert t.execute()["price"].tolist() == [1.0, 2.0]
    finally:
        # each step tolerates what an earlier failure left undone; closing
        # releases the catalog's pooled postgres connection
        try:
            with contextlib.suppress(NoSuchTableError):
                con.catalog.drop_table(f"{name}.trades")
            with contextlib.suppress(NoSuchNamespaceError, NamespaceNotEmptyError):
                con.catalog.drop_namespace(name)
        finally:
            con.catalog.close()
