from __future__ import annotations

from typing import Any

from adbc_driver_manager import ProgrammingError, dbapi
from attr import field, frozen
from attr.validators import instance_of

from xorq.backends.bigquery import Backend as BigQueryBackend
from xorq.common.utils.adbc_utils import ADBCBase


# BigQuery ADBC driver option keys. The driver itself is installed out-of-band
# via ``dbc install bigquery`` (see ci-test-bigquery.yml), so the
# ``adbc_driver_bigquery`` Python package — and its ``DatabaseOptions`` enum —
# is not a dependency; the option strings are inlined here instead.
_PROJECT_ID = "adbc.bigquery.sql.project_id"
_DATASET_ID = "adbc.bigquery.sql.dataset_id"
_AUTH_TYPE = "adbc.bigquery.sql.auth_type"
_AUTH_CLIENT_ID = "adbc.bigquery.sql.auth.client_id"
_AUTH_CLIENT_SECRET = "adbc.bigquery.sql.auth.client_secret"
_AUTH_REFRESH_TOKEN = "adbc.bigquery.sql.auth.refresh_token"
_AUTH_TYPE_DEFAULT = "adbc.bigquery.sql.auth_type.auth_bigquery"
_AUTH_TYPE_USER = "adbc.bigquery.sql.auth_type.user_authentication"


@frozen
class BigQueryADBC(ADBCBase):
    con = field(validator=instance_of(BigQueryBackend))

    @property
    def credentials(self) -> Any:
        # google.cloud.client.Client stores the auth object on the private
        # `_credentials`; it exposes no public `credentials` attribute, so
        # reading `client.credentials` would silently always be None
        return getattr(self.con.client, "_credentials", None)

    @property
    def project_id(self) -> str:
        # prefer data_project: read paths (`self.table()` after ingest) resolve
        # via `current_catalog`, which is `data_project`, so ingest must target
        # the same project or the freshly-ingested table won't resolve
        project_id = self.con.data_project or self.con.billing_project
        if not project_id:
            raise ValueError("BigQuery backend has no resolvable project id")
        return project_id

    @property
    def db_kwargs(self) -> dict[str, str]:
        dataset_id = self.con.current_database
        if not dataset_id:
            raise ValueError(
                "BigQuery ADBC ingest requires a dataset; pass dataset_id to connect()"
            )
        db_kwargs = {_PROJECT_ID: self.project_id, _DATASET_ID: dataset_id}

        # reuse the backend's credentials when they are user credentials (the
        # `gcloud auth application-default login` case); otherwise let the
        # driver discover Application Default Credentials, which is how the
        # backend authenticates service accounts too
        credentials = self.credentials
        client_id = getattr(credentials, "client_id", None)
        client_secret = getattr(credentials, "client_secret", None)
        refresh_token = getattr(credentials, "refresh_token", None)
        if client_id and client_secret and refresh_token:
            db_kwargs |= {
                _AUTH_TYPE: _AUTH_TYPE_USER,
                _AUTH_CLIENT_ID: client_id,
                _AUTH_CLIENT_SECRET: client_secret,
                _AUTH_REFRESH_TOKEN: refresh_token,
            }
        else:
            db_kwargs[_AUTH_TYPE] = _AUTH_TYPE_DEFAULT
        return db_kwargs

    def get_conn(self, **kwargs: Any) -> dbapi.Connection:
        try:
            return dbapi.connect(
                driver="bigquery", db_kwargs={**self.db_kwargs, **kwargs}
            )
        except ProgrammingError as e:
            # the driver manager prefixes load/resolution failures with
            # "[Driver Manager]"; the bigquery driver's own runtime errors
            # (auth, bad args) are prefixed "[bq]", so this only intercepts a
            # genuinely-missing driver and lets everything else propagate
            if "[Driver Manager]" not in str(e):
                raise
            raise RuntimeError(
                "could not load the BigQuery ADBC driver; install it with "
                "`dbc install bigquery` (https://dbc.columnar.tech)"
            ) from e
