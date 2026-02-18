from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    pass

# ── Constants ────────────────────────────────────────────────────────────────
GIZMOSQL_PORT = 31337
GIZMOSQL_IMAGE = "gizmodata/gizmosql:latest"
GIZMOSQL_USERNAME = "ibis"
GIZMOSQL_PASSWORD = "ibis_password"
CONTAINER_NAME = "xorq-gizmosql-test"

ROOT_DIR = Path(__file__).resolve().parents[5]  # xorq repo root
DATA_DIR = ROOT_DIR / "ci" / "ibis-testing-data"

PARQUET_TABLES = (
    "functional_alltypes",
    "diamonds",
    "batting",
    "awards_players",
    "astronauts",
)


# ── Docker container management ──────────────────────────────────────────────
def _port_is_listening(port: int, host: str = "localhost") -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


def _wait_for_container_log(
    container,
    timeout=60,
    poll_interval=1,
    ready_message="GizmoSQL server - started",
):
    start_time = time.time()
    while time.time() - start_time < timeout:
        logs = container.logs().decode("utf-8")
        if ready_message in logs:
            return True
        time.sleep(poll_interval)
    raise TimeoutError(f"Container did not show '{ready_message}' within {timeout}s.")


@pytest.fixture(scope="session")
def gizmosql_server():
    """Start the GizmoSQL Docker container for testing.

    If a server is already listening on GIZMOSQL_PORT, skip Docker management
    and use that server instead.
    """
    if _port_is_listening(GIZMOSQL_PORT):
        yield None
        return

    docker = pytest.importorskip("docker")
    client = docker.from_env()
    parquet_dir = str(DATA_DIR / "parquet")

    try:
        container = client.containers.get(CONTAINER_NAME)
        if container.status == "running":
            yield container
            return
        container.remove(force=True)
    except docker.errors.NotFound:
        pass

    container = client.containers.run(
        image=GIZMOSQL_IMAGE,
        name=CONTAINER_NAME,
        detach=True,
        remove=True,
        tty=True,
        init=True,
        ports={f"{GIZMOSQL_PORT}/tcp": GIZMOSQL_PORT},
        volumes={parquet_dir: {"bind": "/data/parquet", "mode": "ro"}},
        environment={
            "GIZMOSQL_USERNAME": GIZMOSQL_USERNAME,
            "GIZMOSQL_PASSWORD": GIZMOSQL_PASSWORD,
            "TLS_ENABLED": "1",
            "PRINT_QUERIES": "0",
            "DATABASE_FILENAME": ":memory:",
        },
        stdout=True,
        stderr=True,
    )

    _wait_for_container_log(container)
    yield container
    container.stop()


@pytest.fixture(scope="session")
def con(gizmosql_server):
    """GizmoSQL connection with test data loaded."""
    import xorq.api as xo

    conn = xo.gizmosql.connect(
        host="localhost",
        user=GIZMOSQL_USERNAME,
        password=GIZMOSQL_PASSWORD,
        port=GIZMOSQL_PORT,
        use_encryption=True,
        disable_certificate_verification=True,
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
    from xorq.vendor.ibis import util

    name = util.gen_name("temp_table")
    yield name
    with contextlib.suppress(Exception):
        with con._safe_raw_sql(f'DROP TABLE IF EXISTS "{name}"'):
            pass
