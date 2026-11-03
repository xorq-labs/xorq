from __future__ import annotations

from adbc_driver_manager import dbapi
from attr import field, frozen
from attr.validators import instance_of

from xorq.backends.databricks import Backend as DatabricksBackend
from xorq.common.utils.env_utils import EnvConfigable, env_templates_dir


DatabricksConfig = EnvConfigable.subclass_from_env_file(
    env_templates_dir.joinpath(".env.databricks.template")
)
databricks_config = DatabricksConfig.from_env()


def make_connection_params():
    """Create connection parameters from environment variables."""
    return {
        "server_hostname": databricks_config["DATABRICKS_SERVER_HOSTNAME"],
        "http_path": databricks_config["DATABRICKS_HTTP_PATH"],
        "access_token": databricks_config["DATABRICKS_TOKEN"],
    }


def make_connection(**kwargs):
    """Create a Databricks connection using environment variables."""
    con = DatabricksBackend()
    con = con.connect(
        **{
            **make_connection_params(),
            **kwargs,
        }
    )
    return con


@frozen
class DatabricksADBC:
    con = field(validator=instance_of(DatabricksBackend))

    @property
    def default_kwargs(self) -> dict:
        return {
            "hostname": self.con._server_hostname,
            "token": self.con._access_token,
            "http_path": self.con._http_path,
            "port": databricks_config["DATABRICKS_PORT"],
        }

    @property
    def uri(self):
        return self.get_uri()

    def get_uri(self, **kwargs):
        params = {**self.default_kwargs, **kwargs}
        return f"databricks://token:{params['token']}@{params['hostname']}:{params['port']}/{params['http_path']}"

    def get_conn(self, **kwargs):
        return dbapi.connect(
            driver="databricks",
            uri=self.get_uri(**kwargs),
        )

    def adbc_ingest(
        self,
        table_name: str,
        record_batch_reader,
        mode: str = "create",
        catalog_name: str | None = None,
        db_schema_name: str | None = None,
        **kwargs,
    ) -> None:
        # the connection's current catalog and schema unless given; resolving
        # them queries ``self.con``
        catalog_name = catalog_name or self.con.current_catalog
        db_schema_name = db_schema_name or self.con.current_database
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.adbc_ingest(
                    table_name,
                    record_batch_reader,
                    mode=mode,
                    db_schema_name=db_schema_name,
                    catalog_name=catalog_name,
                    **kwargs,
                )
            conn.commit()
