from __future__ import annotations

from typing import TYPE_CHECKING

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
    project_id = bigquery_config[PROJECT_ID_ENV_VAR] or default_credentials[1]
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
        # query_and_wait executes eagerly, so this smoke-tests access
        con.raw_sql("SELECT 1")
    except gexc.Forbidden:
        pytest.skip(f"Cannot access BigQuery project: {project_id}")

    con.create_database(dataset_id, force=True)
    try:
        yield con
    finally:
        con.drop_database(dataset_id, force=True, cascade=True)


@pytest.fixture(scope="session")
def batting(con: Backend, parquet_dir: Path) -> ir.Table:
    return con.read_parquet(
        parquet_dir.joinpath("batting.parquet"), table_name="batting"
    )
