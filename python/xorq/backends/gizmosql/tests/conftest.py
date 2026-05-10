from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

import xorq.api as xo
from xorq.vendor.ibis import util


# ── Constants ────────────────────────────────────────────────────────────────
GIZMOSQL_USERNAME = "ibis"
GIZMOSQL_PASSWORD = "ibis_password"

ROOT_DIR = Path(__file__).resolve().parents[5]  # xorq repo root
DATA_DIR = ROOT_DIR / "ci" / "ibis-testing-data"

PARQUET_TABLES = (
    "functional_alltypes",
    "diamonds",
    "batting",
    "awards_players",
    "astronauts",
)


@pytest.fixture(scope="session")
def gizmosql_server():
    """Start the GizmoSQL server as a managed subprocess via the
    [gizmosql](https://pypi.org/project/gizmosql/) PyPI package.

    The package downloads + caches the matching server binary on first use,
    auto-picks a free port, and tears the server down on exit — no Docker
    is needed for the test fixture.
    """
    gizmosql = pytest.importorskip("gizmosql")
    with gizmosql.Server(
        username=GIZMOSQL_USERNAME,
        password=GIZMOSQL_PASSWORD,
    ) as srv:
        yield srv


@pytest.fixture(scope="session")
def con(gizmosql_server):
    """GizmoSQL connection with test data loaded."""
    conn = xo.gizmosql.connect(
        host=gizmosql_server.host,
        user=gizmosql_server.username,
        password=gizmosql_server.password,
        port=gizmosql_server.port,
        use_encryption=False,
    )

    # Load standard test tables from parquet
    parquet_dir = DATA_DIR / "parquet"
    for table_name in PARQUET_TABLES:
        parquet_path = parquet_dir / f"{table_name}.parquet"
        if parquet_path.exists():
            conn.read_parquet(parquet_path, table_name=table_name)

    return conn


@pytest.fixture(scope="session")
def alltypes(con):
    return con.table("functional_alltypes")


@pytest.fixture(scope="session")
def alltypes_df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope="session")
def awards_players(con):
    return con.table("awards_players")


@pytest.fixture(scope="session")
def batting(con):
    return con.table("batting")


@pytest.fixture
def temp_table(con):
    name = util.gen_name("temp_table")
    yield name
    with contextlib.suppress(Exception):
        with con._safe_raw_sql(f'DROP TABLE IF EXISTS "{name}"'):
            pass
