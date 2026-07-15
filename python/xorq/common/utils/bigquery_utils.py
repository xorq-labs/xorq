from __future__ import annotations

from typing import TYPE_CHECKING, Any

from adbc_driver_manager import dbapi
from attr import field, frozen
from attr.validators import instance_of

from xorq.backends.bigquery import Backend as BigQueryBackend


if TYPE_CHECKING:
    import pyarrow as pa


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
class BigQueryADBC:
    con = field(validator=instance_of(BigQueryBackend))

    @property
    def credentials(self) -> Any:
        # google.cloud.bigquery.Client stores the auth object here
        return getattr(self.con.client, "_credentials", None)

    @property
    def project_id(self) -> str:
        return self.con.billing_project or self.con.data_project

    @property
    def db_kwargs(self) -> dict[str, str]:
        db_kwargs = {_PROJECT_ID: self.project_id}
        if self.con.current_database:
            db_kwargs[_DATASET_ID] = self.con.current_database

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
        return dbapi.connect(driver="bigquery", db_kwargs={**self.db_kwargs, **kwargs})

    def adbc_ingest(
        self,
        table_name: str,
        record_batch_reader: pa.RecordBatchReader | pa.Table,
        mode: str = "create",
        **kwargs: Any,
    ) -> None:
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.adbc_ingest(table_name, record_batch_reader, mode=mode, **kwargs)
            conn.commit()
