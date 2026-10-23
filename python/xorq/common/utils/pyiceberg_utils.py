from xorq.backends.pyiceberg import (
    Backend as PyIcebergBackend,
)
from xorq.common.utils.env_utils import EnvConfigable, env_templates_dir


PyIcebergConfig = EnvConfigable.subclass_from_env_file(
    env_templates_dir.joinpath(".env.pyiceberg.template")
)


def make_connection_defaults():
    # read on every call: connect_env connects with the environment as it is now
    config = PyIcebergConfig.from_env()
    return {
        "uri": config["ICEBERG_URI"],
        "warehouse_path": config["ICEBERG_WAREHOUSE_PATH"],
        "namespace": config["ICEBERG_NAMESPACE"],
        "catalog_name": config["ICEBERG_CATALOG_NAME"],
        "catalog_type": config["ICEBERG_CATALOG_TYPE"],
    }


def make_connection(**kwargs):
    con = PyIcebergBackend()
    con = con.connect(
        **{
            **make_connection_defaults(),
            **kwargs,
        }
    )
    return con


def get_iceberg_snapshots_ids(dt):
    database = dt.source.namespace
    catalog = dt.source.catalog
    table_name = dt.name

    table = catalog.load_table(f"{database}.{table_name}")

    return tuple(snapshot.snapshot_id for snapshot in table.metadata.snapshots)
