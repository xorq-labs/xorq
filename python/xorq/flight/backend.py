from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import pandas as pd
import pyarrow as pa
import toolz

from xorq.common.utils.rbr_utils import (
    make_filtered_reader,
)
from xorq.expr.relations import FlightUDXF
from xorq.flight.action import (
    DropTableAction,
    DropViewAction,
    GetExchangeAction,
    GetSchemaQueryAction,
    ListExchangesAction,
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
from xorq.vendor.ibis.util import gen_name


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
            self.con.upload_table(name, obj)
            return self.table(name)
        if isinstance(obj, pa.RecordBatchReader):
            self.con.upload_batches(name, obj)
            return self.table(name)
        raise TypeError(f"Unsupported type for create_table: {type(obj)}")

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
        cert_chain=None,
        private_key=None,
        tls_root_certs=None,
    ) -> None:
        self.con = FlightClient(
            host=host,
            port=port,
            username=username,
            password=password,
            cert_chain=cert_chain,
            private_key=private_key,
            tls_root_certs=tls_root_certs,
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
            self.con.upload_table(table_name, source)
        elif isinstance(source, pd.DataFrame):
            self.con.upload_batches(
                table_name, pa.RecordBatchReader.from_stream(source)
            )
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

        return self.read_record_batches(source, table_name=table_name)

    def read_record_batches(self, source, table_name=None):
        table_name = table_name or gen_name("read_record_batches")
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
        kwargs = {"name": name, "database": database, "force": force}
        self.con.do_action(
            DropTableAction.name, action_body=kwargs, options=self.con._options
        )

    def drop_view(
        self,
        name: str,
        *,
        database: str | None = None,
        schema: str | None = None,
        force: bool = False,
    ) -> None:
        kwargs = {"name": name, "database": database, "schema": schema, "force": force}
        self.con.do_action(
            DropViewAction.name, action_body=kwargs, options=self.con._options
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

    def get_flight_udxf(self, command):
        udxf = self.con.do_action_one(GetExchangeAction.name, action_body=command)

        @toolz.curry
        def flight_udxf(
            expr,
            con=None,
            name=None,
            inner_name=None,
            **kwargs,
        ):
            @contextmanager
            def make_server():
                try:
                    yield SimpleNamespace(client=self.con)
                finally:
                    pass

            return (
                FlightUDXF.from_expr(
                    input_expr=expr,
                    udxf=udxf,
                    make_server=make_server,
                    **kwargs,
                )
                .to_expr()
                .into_backend(con=con or expr._find_backend(), name=name)
            )

        return flight_udxf

    def list_udxfs(self):
        return self.con.do_action_one(ListExchangesAction.name)
