from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from xorq.common.utils.env_utils import EnvConfigable


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from xorq.backends.bigquery import Backend
    from xorq.vendor.ibis.expr import types as ir


# the google client libraries are an optional (`--extra bigquery`) dependency
pytest.importorskip("google.cloud.bigquery")

import google.api_core.exceptions as gexc  # noqa: E402
import google.auth  # noqa: E402
import google.auth.exceptions  # noqa: E402

import xorq.api as xo  # noqa: E402
from xorq.vendor.ibis.backends.bigquery import EXTERNAL_DATA_SCOPES  # noqa: E402
from xorq.vendor.ibis.util import gen_name  # noqa: E402


PROJECT_ID_ENV_VAR = "GOOGLE_BIGQUERY_PROJECT_ID"
bigquery_config = EnvConfigable.subclass_from_kwargs(PROJECT_ID_ENV_VAR).from_env()


@pytest.fixture(scope="session")
def default_credentials() -> tuple[object, str]:
    try:
        credentials, project_id = google.auth.default(scopes=EXTERNAL_DATA_SCOPES)
    except google.auth.exceptions.DefaultCredentialsError as exc:
        pytest.skip(f"Could not get GCP credentials: {exc}")
    return credentials, project_id


@pytest.fixture(scope="session")
def project_id(default_credentials: tuple[object, str]) -> str:
    _, default_project_id = default_credentials
    project_id = bigquery_config[PROJECT_ID_ENV_VAR] or default_project_id
    if not project_id:
        pytest.skip(f"no project id: set ${PROJECT_ID_ENV_VAR} or configure ADC")
    return project_id


@pytest.fixture(scope="session")
def credentials(default_credentials: tuple[object, str]) -> object:
    credentials, _ = default_credentials
    return credentials


@pytest.fixture(scope="session")
def dataset_id() -> str:
    return gen_name("xorq_gbq_testing")


@pytest.fixture(scope="session")
def con(credentials: object, project_id: str, dataset_id: str) -> Iterator[Backend]:
    con = xo.bigquery.connect(
        project_id=project_id, dataset_id=dataset_id, credentials=credentials
    )
    # disable the query cache so tests observe real behavior
    con.client.default_query_job_config.use_query_cache = False
    try:
        # query_and_wait executes eagerly, so this smoke-tests access before
        # the dataset is created; any client-side or auth error (Forbidden,
        # NotFound, ServiceUnavailable, transport/credential failures) skips
        # cleanly instead of surfacing as a hard error
        con.raw_sql("SELECT 1")
        con.create_database(dataset_id, force=True)
    except (gexc.GoogleAPICallError, google.auth.exceptions.GoogleAuthError) as exc:
        pytest.skip(f"Cannot access BigQuery project {project_id}: {exc}")
    try:
        yield con
    finally:
        con.drop_database(dataset_id, force=True, cascade=True)


@pytest.fixture
def temp_table(con: Backend) -> Iterator[str]:
    # yields a unique table name and drops it on teardown, so tests need no
    # per-test cleanup (mirrors the gizmosql backend's temp_table fixture)
    name = gen_name("xorq_gbq_table")
    yield name
    with contextlib.suppress(Exception):
        con.drop_table(name, force=True)


def _persistent_table(con: Backend, parquet_dir: Path, name: str) -> Iterator[ir.Table]:
    # create the fixture in the connection's dataset (not the anonymous session
    # dataset that read_parquet lands in) so it resolves in `__TABLES__` and the
    # dasher normalizer exercises its real last_modified_time keying
    df = pd.read_parquet(parquet_dir.joinpath(f"{name}.parquet"))
    con.create_table(name, obj=df, overwrite=True)
    try:
        yield con.table(name)
    finally:
        con.drop_table(name, force=True)


@pytest.fixture(scope="session")
def batting(con: Backend, parquet_dir: Path) -> Iterator[ir.Table]:
    yield from _persistent_table(con, parquet_dir, "batting")


@pytest.fixture(scope="session")
def awards_players(con: Backend, parquet_dir: Path) -> Iterator[ir.Table]:
    yield from _persistent_table(con, parquet_dir, "awards_players")
