from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import pyarrow as pa

from xorq.common.utils.rbr_utils import (
    make_filtered_reader,
)
from xorq.flight.action import (
    DropTableAction,
    DropViewAction,
    GetSchemaQueryAction,
    ListTablesAction,
    ReadParquetAction,
    TableInfoAction,
    VersionAction,
)
from xorq.flight.client import FlightClient
from xorq.vendor import ibis
from xorq.vendor.ibis import util
from xorq.vendor.ibis.backends.sql import SQLBackend
from xorq.vendor.ibis.expr import schema as sch
from xorq.vendor.ibis.expr import types as ir


class Backend(SQLBackend):
    @property
    def name(self):
        return "xorq_flight"

    @property
    def version(self) -> str:
        return self.con.do_action(VersionAction.name, options=self.con._options)[0]

    def create_table(
        self,
        name: str,
        obj: pd.DataFrame | pa.Table | ir.Table | None = None,
        *,
        schema: ibis.Schema | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ) -> ir.Table:
        if isinstance(obj, pd.DataFrame):
            obj = pa.Table.from_pandas(obj)

        if isinstance(obj, pa.Table):
            obj = obj.to_batches()

        if isinstance(obj, ir.Table):
            obj = obj.to_pyarrow_batches()

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        return self.con.do_action_one(
            GetSchemaQueryAction.name, action_body=query, options=self.con._options
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.con = None

    def do_connect(
        self,
        host="localhost",
        port=8815,
        username=None,
        password=None,
        tls_roots=None,
    ) -> None:
        self.con = FlightClient(
            host=host,
            port=port,
            username=username,
            password=password,
            tls_roots=tls_roots,
        )

    def get_schema(
        self,
        table_name: str,
        *,
        catalog: str | None = None,
        database: str | None = None,
    ) -> sch.Schema:
        args = {
            "table_name": table_name,
            "catalog": catalog,
            "database": database,
        }
        return self.con.do_action_one(
            TableInfoAction.name, action_body=args, options=self.con._options
        )

    def read_in_memory(
        self,
        source: pd.DataFrame | pa.Table | pa.RecordBatchReader,
        table_name: str | None = None,
    ) -> ir.Table:
        table_name = table_name or util.gen_name("read_in_memory")

        if isinstance(source, pa.Table):
            self.con.upload_data(table_name, source)
        elif isinstance(source, pa.RecordBatchReader):
            self.con.upload_batches(table_name, source)
        return self.table(table_name)

    def read_parquet(
        self,
        source_list: str | Iterable[str],
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        args = {
            "source_list": source_list,
            "table_name": table_name,
        }
        table_name = self.con.do_action_one(
            ReadParquetAction.name, action_body=args, options=self.con._options
        )
        return self.table(table_name)

    def register(
        self,
        source: str | Path | Any,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        if isinstance(source, pd.DataFrame):
            source = pa.Table.from_pandas(source)
            source = pa.RecordBatchReader.from_batches(
                source.schema, source.to_batches()
            )

        if isinstance(source, pa.RecordBatchReader):
            self.con.upload_batches(table_name, source)

        return self.table(table_name)

    def list_tables(
        self, like: str | None = None, database: tuple[str, str] | str | None = None
    ) -> list[str]:
        kwargs = {
            "like": like,
            "database": database,
        }

        return self.con.do_action_one(
            ListTablesAction.name, action_body=kwargs, options=self.con._options
        )

    def drop_table(
        self,
        name: str,
        database: tuple[str, str] | str | None = None,
        force: bool = False,
    ) -> None:
        self.con.do_action(
            DropTableAction.name, action_body=name, options=self.con._options
        )

    def drop_view(
        self,
        name: str,
        *,
        database: str | None = None,
        schema: str | None = None,
        force: bool = False,
    ) -> None:
        self.con.do_action(
            DropViewAction.name, action_body=name, options=self.con._options
        )

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 10_000,
        **_: Any,
    ) -> pa.ipc.RecordBatchReader:
        batches = self.con.execute_batches(
            expr, params=params, limit=limit, chunk_size=chunk_size
        )
        batches = make_filtered_reader(batches)
        return batches
