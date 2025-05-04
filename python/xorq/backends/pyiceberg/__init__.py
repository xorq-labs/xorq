from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import pandas as pd
import pyarrow as pa
from pyiceberg.catalog import load_catalog
from pyiceberg.io.pyarrow import pyarrow_to_schema
from pyiceberg.table import ALWAYS_TRUE
from pyiceberg.table import Table as IcebergTable
from pyiceberg.table.name_mapping import MappedField, NameMapping

import xorq as xo
from xorq.vendor import ibis
from xorq.vendor.ibis.backends.sql import SQLBackend
from xorq.vendor.ibis.expr import schema as sch
from xorq.vendor.ibis.expr import types as ir
from xorq.vendor.ibis.util import gen_name


# we use the PyIceberg's default connection
# TODO: See if there is anything to be done for creating connection profile for PyIceberg backend
# TODO: needs tests


def parse_url(url: str) -> Dict[str, Any]:
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(url)
    warehouse_path = (
        parsed.netloc + parsed.path if parsed.netloc else parsed.path.lstrip("/")
    )
    params = {k: v[0] if len(v) == 1 else v for k, v in parse_qs(parsed.query).items()}

    return {
        **{
            "namespace": "default",
            "catalog_name": "default",
            "catalog_type": "sql",
            "warehouse_path": warehouse_path,
        },
        **params,
    }


class Backend(SQLBackend):
    name = "pyiceberg"
    version = "0.0.0"

    def __init__(self):
        super().__init__()
        self.catalog = None
        self.namespace = "default"
        self.warehouse_path = None
        self.duckdb_con = None

    @staticmethod
    def _from_url(url: str) -> Dict[str, Any]:
        return parse_url(url)

    def do_connect(
        self,
        warehouse_path="warehouse",
        namespace="default",
        catalog_name="default",
        catalog_type="sql",
    ) -> None:
        self.warehouse_path = Path(warehouse_path).absolute()
        self.namespace = namespace

        Path(self.warehouse_path).mkdir(exist_ok=True)

        self.catalog_params = {
            "type": catalog_type,
            "uri": f"sqlite:///{self.warehouse_path}/pyiceberg_catalog.db",
            "warehouse": f"file://{self.warehouse_path}",
        }

        self.catalog = load_catalog(catalog_name, **self.catalog_params)

        if self.namespace not in [n[0] for n in self.catalog.list_namespaces()]:
            self.catalog.create_namespace(self.namespace)

        self.duckdb_con = xo.duckdb.connect()
        self._setup_duckdb_connection()
        self._reflect_views()

    def _setup_duckdb_connection(self):
        """Configure DuckDB connection with required settings"""
        commands = [
            "INSTALL iceberg;",
            "LOAD iceberg;",
            "SET unsafe_enable_version_guessing=true;",
        ]
        for cmd in commands:
            self.duckdb_con.raw_sql(cmd)

    def _reflect_views(self):
        # required for duckdb backend but for PyIceberg backend this will not
        # be necessary
        table_names = [t[1] for t in self.catalog.list_tables(self.namespace)]

        for table_name in table_names:
            table_path = f"{self.warehouse_path}/{self.namespace}.db/{table_name}"
            self._setup_duckdb_connection()

            escaped_path = table_path.replace("'", "''")
            safe_name = f'"{table_name}"' if "-" in table_name else table_name

            self.duckdb_con.raw_sql(f"""
                CREATE OR REPLACE VIEW {safe_name} AS
                SELECT * FROM iceberg_scan(
                    '{escaped_path}', 
                    version='?',
                    allow_moved_paths=true
                )
            """)

    def create_table(
        self,
        name: str,
        obj: Optional[Union[pd.DataFrame, pa.Table, ir.Table]] = None,
        *,
        schema: Optional[ibis.Schema] = None,
        database: Optional[str] = None,
        temp: bool = False,
        overwrite: bool = False,
    ) -> ir.Table:
        database = database or self.namespace
        full_table_name = f"{database}.{name}"

        if self.catalog.table_exists(full_table_name):
            if not overwrite:
                raise ValueError(f"Table {full_table_name} already exists")
            self.drop_table(name, database=database)

        if obj is None and schema is not None:
            raise NotImplementedError("Creating empty tables is not implemented yet")

        if isinstance(obj, pd.DataFrame):
            obj = pa.Table.from_pandas(obj)
        elif isinstance(obj, ir.Table):
            obj = self.to_pyarrow(obj)

        mapped_fields = []
        for i, field in enumerate(obj.schema, 1):
            mapped_fields.append(
                MappedField(
                    field_id=i,
                    names=[field.name],
                    fields=[],  # nested?
                )
            )

        name_mapping = NameMapping(mapped_fields)

        iceberg_schema = pyarrow_to_schema(obj.schema, name_mapping=name_mapping)
        iceberg_table = self.catalog.create_table(
            identifier=full_table_name, schema=iceberg_schema
        )
        iceberg_table.append(obj)

        self._reflect_views()
        return self.table(name)

    def insert(
        self,
        table_name: str,
        data: Union[pd.DataFrame, pa.Table, ir.Table],
        database: Optional[str] = None,
        mode: str = "append",
    ) -> bool:
        database = database or self.namespace
        full_table_name = f"{database}.{table_name}"

        if not self.catalog.table_exists(full_table_name):
            raise ValueError(f"Table {full_table_name} does not exist")

        if isinstance(data, pd.DataFrame):
            data = pa.Table.from_pandas(data)
        elif isinstance(data, ir.Table):
            data = self.to_pyarrow(data)

        iceberg_table = self.catalog.load_table(full_table_name)
        iceberg_table.refresh()

        if mode == "overwrite":
            self._overwrite_table_data(iceberg_table, data)
        else:
            with iceberg_table.transaction() as transaction:
                transaction.append(data)

        self._reflect_views()
        return True

    def _overwrite_table_data(self, iceberg_table: IcebergTable, data: pa.Table):
        tx = iceberg_table.transaction()
        tx.delete(delete_filter=ALWAYS_TRUE)

        update_snapshot = tx.update_snapshot()
        with update_snapshot.fast_append() as append_files:
            for data_file in iceberg_table.writer().write_table(data):
                append_files.append_data_file(data_file)

        tx.commit_transaction()

    def to_pyarrow(self, expr: ir.Expr) -> pa.Table:
        batches = self.to_pyarrow_batches(expr)
        return pa.Table.from_batches(batches.read_all())

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        params: Optional[Mapping[ir.Scalar, Any]] = None,
        limit: Optional[Union[int, str]] = None,
        chunk_size: int = 10_000,
        **_: Any,
    ) -> pa.ipc.RecordBatchReader:
        self._reflect_views()
        # FIXME: this should just use pyiceberg's scan operator respecting any
        # predicate and/or projection pushdowns
        # TODO: add a ibis select expression to pyiceberg's scan operator converter
        # Check with dan if the utils already exist.
        # Raise NotImplementedError if not a selection
        return self.duckdb_con.to_pyarrow_batches(
            expr, params=params, limit=limit, chunk_size=chunk_size
        )

    def list_tables(
        self,
        like: Optional[str] = None,
        database: Optional[Union[Tuple[str, str], str]] = None,
    ) -> List[str]:
        database = database or self.namespace
        table_names = [t[1] for t in self.catalog.list_tables(database)]

        if like is not None:
            import fnmatch

            return [t for t in table_names if fnmatch.fnmatch(t, like)]

        return table_names

    def drop_table(
        self,
        name: str,
        database: Optional[str] = None,
        force: bool = False,
    ) -> None:
        database = database or self.namespace
        full_table_name = f"{database}.{name}"

        if self.catalog.table_exists(full_table_name):
            self.catalog.drop_table(full_table_name)

    def get_schema(
        self,
        table_name: str,
        *,
        catalog: Optional[str] = None,
        database: Optional[str] = None,
    ) -> sch.Schema:
        database = database or self.namespace

        sql = f"SELECT * FROM '{table_name}' LIMIT 0"
        result = self.duckdb_con.sql(sql)
        pa_table = result.to_pyarrow()
        return sch.Schema.from_pyarrow(pa_table.schema)

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        limit_query = f"SELECT * FROM ({query}) AS t LIMIT 0"
        result = self.duckdb_con.sql(limit_query)

        pa_table = result.to_pyarrow()
        return sch.Schema.from_pyarrow(pa_table.schema)

    def read_record_batches(
        self,
        reader: Union[pa.RecordBatchReader, pa.ChunkedArray],
        table_name: Optional[str] = None,
    ) -> ir.Table:
        table_name = table_name or gen_name("read_record_batches")
        table = pa.Table.from_batches([batch for batch in reader])

        self.create_table(name=table_name, obj=table, database=self.namespace)
        self._reflect_views()
        return self.table(table_name)
