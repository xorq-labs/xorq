import os

import pytest

from xorq.backends.pyiceberg import Backend as PyIcebergBackend


def test_can_connect(iceberg_con):
    assert iceberg_con is not None


@pytest.mark.parametrize(
    "uri",
    [
        pytest.param(None, id="default"),
        pytest.param(
            (
                f"postgresql://{os.environ['POSTGRES_USER']}:"
                f"{os.environ['POSTGRES_PASSWORD']}@"
                f"{os.environ['POSTGRES_HOST']}:"
                f"{os.environ['POSTGRES_PORT']}/"
                f"{os.environ['POSTGRES_DATABASE']}"
            ),
            id="postgresql",
        ),
    ],
)
def test_connect_env(trades_df, monkeypatch, tmp_path, uri):
    name = "test"
    warehouse_path = tmp_path / "another_warehouse"
    warehouse_path.mkdir(exist_ok=True)

    monkeypatch.setenv("ICEBERG_WAREHOUSE_PATH", str(warehouse_path))
    monkeypatch.setenv("ICEBERG_NAMESPACE", name)
    monkeypatch.setenv("ICEBERG_CATALOG_NAME", name)

    if uri:
        monkeypatch.setenv("ICEBERG_URI", str(uri))
        # assert uri is points to the right DB
        assert os.environ["ICEBERG_URI"] == uri

    # assert connection can be established
    con = PyIcebergBackend.connect_env()
    assert con is not None
    assert con.catalog.name == name
    assert con.catalog.properties["type"] == "sql"
    assert "another_warehouse" in con.catalog.properties["warehouse"]

    # assert connection can create tables
    table_name = "trades"
    t = con.create_table(table_name, trades_df, overwrite=True)
    assert table_name in con.list_tables()
    assert not t.execute().empty
