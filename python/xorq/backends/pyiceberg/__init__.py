from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import pandas as pd
import pyarrow as pa
from pyiceberg.catalog import load_catalog
from pyiceberg.table import ALWAYS_TRUE
from pyiceberg.table import Table as IcebergTable

import xorq.vendor.ibis.expr.operations as ops
from xorq.backends.postgres.compiler import compiler as postgres_compiler
from xorq.backends.pyiceberg.compiler import PyIceberg, translate
from xorq.backends.pyiceberg.relations import PyIcebergTable
from xorq.vendor import ibis
from xorq.vendor.ibis.backends.sql import SQLBackend
from xorq.vendor.ibis.expr import schema as sch
from xorq.vendor.ibis.expr import types as ir
from xorq.vendor.ibis.util import gen_name


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


def _overwrite_table_data(iceberg_table: IcebergTable, data: pa.Table):
    tx = iceberg_table.transaction()
    tx.delete(delete_filter=ALWAYS_TRUE)

    update_snapshot = tx.update_snapshot()
    with update_snapshot.fast_append() as append_files:
        for data_file in iceberg_table.writer().write_table(data):
            append_files.append_data_file(data_file)

    tx.commit_transaction()


class Backend(SQLBackend):
    name = "pyiceberg"
    dialect = PyIceberg
    compiler = postgres_compiler

    @property
    def version(self):
        import importlib.metadata

        return importlib.metadata.version("pyiceberg")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.catalog = None
        self.namespace = "default"
        self.warehouse_path = None
        self.duckdb_con = None
        self.catalog_params = None
        self.uri = None

    @staticmethod
    def _from_url(url: str) -> Dict[str, Any]:
        return parse_url(url)

    @classmethod
    def connect_env(cls, **kwargs):
        from xorq.common.utils.pyiceberg_utils import make_connection

        return make_connection(**kwargs)

    def do_connect(
        self,
        uri=None,
        warehouse_path="warehouse",
        namespace="default",
        catalog_name="default",
        catalog_type="sql",
    ) -> None:
        self.warehouse_path = Path(warehouse_path).absolute()
        self.namespace = namespace
        self.warehouse_path.mkdir(exist_ok=True)
        self.uri = uri or f"sqlite:///{self.warehouse_path}/pyiceberg_catalog.db"

        self.catalog_params = {
            "type": catalog_type,
            "uri": self.uri,
            "warehouse": f"file://{self.warehouse_path}",
        }

        self.catalog = load_catalog(catalog_name, **self.catalog_params)

        if self.namespace not in [n[0] for n in self.catalog.list_namespaces()]:
            self.catalog.create_namespace(self.namespace)

    def table(
        self,
        name: str,
        schema: str | None = None,
        database: tuple[str, str] | str | None = None,
        snapshot_id: str | None = None,
    ) -> ir.Table:
        table_schema = self.get_schema(name, catalog=schema, database=database)

        return PyIcebergTable(
            name=name,
            schema=table_schema,
            source=self,
            namespace=ops.Namespace(catalog=schema, database=database),
            snapshot_id=snapshot_id,
        ).to_expr()

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

        # NEEDS this to reset the field_id
        obj = obj.cast(
            pa.schema(
                [
                    pa.field(name, type=typ)
                    for name, typ in zip(obj.schema.names, obj.schema.types)
                ]
            )
        )

        iceberg_table = self.catalog.create_table(
            identifier=full_table_name, schema=obj.schema
        )
        iceberg_table.append(obj)

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
            _overwrite_table_data(iceberg_table, data)
        else:
            with iceberg_table.transaction() as transaction:
                transaction.append(data)

        return True

    def upsert(
        self,
        table_name: str,
        data: Union[pd.DataFrame, pa.Table, ir.Table],
        database: Optional[str] = None,
        join_cols=None,
        when_matched_update_all=True,
        when_not_matched_insert_all=True,
        case_sensitive=True,
    ):
        """Wrapper around upsert"""
        database = database or self.namespace
        full_table_name = f"{database}.{table_name}"

        if isinstance(data, pd.DataFrame):
            data = pa.Table.from_pandas(data)
        elif isinstance(data, ir.Table):
            data = self.to_pyarrow(data)

        iceberg_table = self.catalog.load_table(full_table_name)
        iceberg_table.upsert(
            data,
            join_cols=join_cols,
            when_matched_update_all=when_matched_update_all,
            when_not_matched_insert_all=when_not_matched_insert_all,
            case_sensitive=case_sensitive,
        )

        return self.table(table_name)

    def to_pyarrow(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pa.Table:
        batches = self.to_pyarrow_batches(expr, params=params, limit=limit, **kwargs)
        return batches.read_all()

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        params: Optional[Mapping[ir.Scalar, Any]] = None,
        limit: Optional[Union[int, str]] = None,
        chunk_size: int = 10_000,
        **_: Any,
    ) -> pa.ipc.RecordBatchReader:
        query = translate(expr.op(), namespace=self.namespace, catalog=self.catalog)
        return query.to_arrow_batch_reader()

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
        catalog = (
            self.catalog
            if catalog is None
            else load_catalog(catalog, **self.catalog_params)
        )

        table = catalog.load_table(f"{database}.{table_name}")
        return sch.Schema.from_pyarrow(table.schema().as_arrow())

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        raise NotImplementedError("_get_schema_using_query")

    def read_record_batches(
        self,
        reader: Union[pa.RecordBatchReader, pa.ChunkedArray],
        table_name: Optional[str] = None,
    ) -> ir.Table:
        table_name = table_name or gen_name("read_record_batches")
        table = pa.Table.from_batches(reader, reader.schema)

        self.create_table(name=table_name, obj=table, database=self.namespace)
        return self.table(table_name)

    def list_snapshots(self, database=None) -> dict[str, int]:
        database = database or self.namespace
        table_names = [t[1] for t in self.catalog.list_tables(database)]

        snapshots = {}
        for table_name in table_names:
            ice_table = self.catalog.load_table(f"{database}.{table_name}")
            ice_table.inspect.snapshots()
            snapshots[table_name] = (
                ice_table.inspect.snapshots().column("snapshot_id").to_pylist()
            )

        return snapshots
