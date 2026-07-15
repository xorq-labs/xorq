from __future__ import annotations

from typing import TYPE_CHECKING, Any

from adbc_driver_bigquery import DatabaseOptions, dbapi
from attr import field, frozen
from attr.validators import instance_of

from xorq.backends.bigquery import Backend as BigQueryBackend


if TYPE_CHECKING:
    import pyarrow as pa


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
    def dataset_id(self) -> str | None:
        return self.con.current_database

    @property
    def db_kwargs(self) -> dict[str, str]:
        db_kwargs = {DatabaseOptions.PROJECT_ID.value: self.project_id}
        if self.dataset_id:
            db_kwargs[DatabaseOptions.DATASET_ID.value] = self.dataset_id

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
                DatabaseOptions.AUTH_TYPE.value: DatabaseOptions.AUTH_VALUE_USER_AUTHENTICATION.value,
                DatabaseOptions.AUTH_CLIENT_ID.value: client_id,
                DatabaseOptions.AUTH_CLIENT_SECRET.value: client_secret,
                DatabaseOptions.AUTH_REFRESH_TOKEN.value: refresh_token,
            }
        else:
            db_kwargs[DatabaseOptions.AUTH_TYPE.value] = (
                DatabaseOptions.AUTH_VALUE_BIGQUERY.value
            )
        return db_kwargs

    def get_conn(self, **kwargs: Any) -> dbapi.Connection:
        return dbapi.connect(db_kwargs={**self.db_kwargs, **kwargs})

    def adbc_ingest(
        self,
        table_name: str,
        record_batch_reader: pa.RecordBatchReader | pa.Table,
        mode: str = "create",
        **kwargs: Any,
    ) -> None:
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.adbc_ingest(
                    table_name,
                    record_batch_reader,
                    mode=mode,
                    **kwargs,
                )
            conn.commit()
