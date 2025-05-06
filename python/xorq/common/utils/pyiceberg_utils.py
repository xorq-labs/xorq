from xorq.backends.pyiceberg import (
    Backend as PyIcebergBackend,
)
from xorq.common.utils.env_utils import EnvConfigable, env_templates_dir


PyIcebergConfig = EnvConfigable.subclass_from_env_file(
    env_templates_dir.joinpath(".env.pyiceberg.template")
)
pyiceberg_config = PyIcebergConfig.from_env()


def make_connection_defaults():
    return {
        "uri": pyiceberg_config["ICEBERG_URI"],
        "warehouse_path": pyiceberg_config["ICEBERG_WAREHOUSE_PATH"],
        "namespace": pyiceberg_config["ICEBERG_NAMESPACE"],
        "catalog_name": pyiceberg_config["ICEBERG_CATALOG_NAME"],
        "catalog_type": pyiceberg_config["ICEBERG_CATALOG_TYPE"],
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
