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
from xorq.writes.enums import WriteMode


__all__ = [
    "Backend",
    "parse_url",
]


def parse_url(url: str) -> Dict[str, Any]:
    from urllib.parse import parse_qs, urlparse  # noqa: PLC0415

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
        import importlib.metadata  # noqa: PLC0415

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
        from xorq.common.utils.pyiceberg_utils import make_connection  # noqa: PLC0415

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
            import fnmatch  # noqa: PLC0415

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

    @staticmethod
    def _stream_reader_to_parquet(
        ice: IcebergTable, reader: pa.RecordBatchReader
    ) -> str:
        import pyarrow.parquet as pq  # noqa: PLC0415

        file_path = f"{ice.location()}/data/{gen_name('add_files')}.parquet"
        # add_files references files in place, so write under the table location.
        output = ice.io.new_output(file_path).create(overwrite=True)
        try:
            with pq.ParquetWriter(output, reader.schema) as writer:
                for batch in reader:
                    writer.write_batch(batch)
        finally:
            output.close()
        return file_path

    def read_record_batches(
        self,
        reader: Union[pa.RecordBatchReader, pa.ChunkedArray],
        table_name: Optional[str] = None,
        mode: WriteMode | str = WriteMode.CREATE,
        branch: Optional[str] = None,
        schema: Optional[pa.Schema] = None,
    ) -> ir.Table:
        # Callers on the write-through path hand us the wire string (mode.value);
        # normalize at the boundary so the body always works with the enum.
        mode = WriteMode(mode)
        table_name = table_name or gen_name("read_record_batches")
        full_name = f"{self.namespace}.{table_name}"
        # ``schema`` lets a direct caller override the table schema; on the
        # remote-table path it is omitted and we fall back to the reader's own
        # schema (already projected and cast upstream of the cache).
        schema = schema or reader.schema
        exists = self.catalog.table_exists(full_name)

        if branch is None:
            match mode:
                case WriteMode.CREATE:
                    if exists:
                        raise ValueError(f"Table {full_name} already exists")
                    ice = self.catalog.create_table(identifier=full_name, schema=schema)
                case WriteMode.APPEND:
                    if not exists:
                        raise ValueError(f"Table {full_name} does not exist")
                    ice = self.catalog.load_table(full_name)
                case _:
                    raise ValueError(f"Unsupported write mode: {mode!r}")
        else:
            if not exists:
                # A branch must point at a snapshot, and a freshly created table
                # has none. Seed one empty snapshot so create_branch below has a
                # base; the seed carries no rows and becomes ancestor history once
                # the branch is fast-forwarded into main at publish.
                ice = self.catalog.create_table(identifier=full_name, schema=schema)
                ice.append(schema.empty_table())
                ice = self.catalog.load_table(full_name)
            else:
                ice = self.catalog.load_table(full_name)

            if mode == WriteMode.CREATE and branch in ice.refs():
                raise ValueError(
                    f"Branch {branch!r} already exists on table {full_name}"
                )

            if branch not in ice.refs():
                current = ice.current_snapshot()
                ice.manage_snapshots().create_branch(
                    current.snapshot_id, branch
                ).commit()
                ice = self.catalog.load_table(full_name)

        file_path = self._stream_reader_to_parquet(ice, reader)
        # File names are unique per call, so the duplicate scan is unneeded and
        # also raises on tables whose only snapshot has zero data files.
        with ice.transaction() as tx:
            if branch is None:
                tx.add_files([file_path], check_duplicate_files=False)
            else:
                tx.add_files([file_path], check_duplicate_files=False, branch=branch)

        return self.table(table_name)

    def publish_branch(self, table_name: str, branch: str) -> None:
        # Repoint main at the staging branch tip, then drop the branch. This is a
        # fast-forward, NOT a merge: set_current_snapshot overwrites the main
        # pointer, so any commits made to main since the branch was cut would be
        # discarded. Safe under WAP because main only ever advances via publish.
        full_name = f"{self.namespace}.{table_name}"
        # An empty stream never opens the sink writer, so the table itself is
        # never created (it is created lazily inside read_record_batches).
        missing = not self.catalog.table_exists(full_name)
        ice = None if missing else self.catalog.load_table(full_name)
        if missing or branch not in ice.refs():
            raise RuntimeError(
                f"staging branch {branch!r} missing at publish on {full_name}. "
                "The sink opens its writer on the first batch, so either the "
                "audited input was empty (no batch, no artifact) or the staging "
                "write has not committed yet (async sink?)."
            )
        staging_snap = ice.refs()[branch].snapshot_id
        ice.manage_snapshots().set_current_snapshot(staging_snap).commit()
        ice = self.catalog.load_table(full_name)
        ice.manage_snapshots().remove_branch(branch).commit()

    def publish_staging_table(self, staging: str, final: str) -> None:
        # Repoint metadata instead of copying rows: register staging's parquet
        # data files into final, then drop staging's catalog entry. add_files is
        # metadata-only (reads footers, not rows). This relies on pyiceberg's
        # Catalog.drop_table contract — it removes only the catalog entry and does
        # NOT purge data files (purge_table is the separate, opt-in API) — so the
        # files now live under final without being rewritten. A catalog that
        # purged on drop would delete the files out from under final.
        full_staging = f"{self.namespace}.{staging}"
        full_final = f"{self.namespace}.{final}"
        if not self.catalog.table_exists(full_staging):
            raise RuntimeError(
                f"staging table {full_staging!r} missing at publish. The sink "
                "opens its writer on the first batch, so either the audited "
                "input was empty (no batch, no artifact) or the staging write "
                "has not committed yet (async sink?)."
            )
        staged_tbl = self.catalog.load_table(full_staging)
        data_files = [task.file.file_path for task in staged_tbl.scan().plan_files()]
        if not self.catalog.table_exists(full_final):
            self.catalog.create_table(identifier=full_final, schema=staged_tbl.schema())
        if data_files:
            self.catalog.load_table(full_final).add_files(data_files)
        self.drop_table(staging)

    def list_snapshots(self, database: Optional[str] = None) -> dict[str, int]:
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
