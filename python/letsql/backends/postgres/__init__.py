from functools import partial
from pathlib import Path
from typing import Any, Mapping

import ibis.expr.schema as sch
import pyarrow as pa
import pyarrow.parquet as pq
import sqlglot as sg
import sqlglot.expressions as sge
from ibis.backends.postgres import Backend as IbisPostgresBackend
from ibis.expr import types as ir
from ibis.util import (
    gen_name,
)

from letsql.backends.postgres.compiler import compiler
from letsql.common.utils.defer_utils import (
    read_csv_rbr,
)
from letsql.common.utils.graph_utils import replace_fix
from letsql.expr.relations import CachedNode, replace_cache_table


class Backend(IbisPostgresBackend):
    _top_level_methods = ("connect_examples", "connect_env")
    compiler = compiler

    @classmethod
    def connect_env(cls, **kwargs):
        from letsql.common.utils.postgres_utils import make_connection

        return make_connection(**kwargs)

    @classmethod
    def connect_examples(cls):
        return cls().connect(
            host="examples.letsql.com",
            user="letsql",
            password="letsql",
            database="letsql",
        )

    @staticmethod
    def _register_and_transform_cache_tables(expr):
        """This function will sequentially execute any cache node that is not already cached"""

        def fn(node, _, **kwargs):
            node = node.__recreate__(kwargs)
            if isinstance(node, CachedNode):
                uncached, storage = node.parent, node.storage
                node = storage.set_default(uncached, uncached.op())
            return node

        op = expr.op()
        out = op.replace(replace_fix(fn))

        return out.to_expr()

    def execute(
        self,
        expr: ir.Expr,
        params: Mapping | None = None,
        limit: str | None = "default",
        **kwargs: Any,
    ) -> Any:
        expr = self._register_and_transform_cache_tables(expr)
        return super().execute(expr, params=params, limit=limit, **kwargs)

    def _to_sqlglot(
        self, expr: ir.Expr, *, limit: str | None = None, params=None, **_: Any
    ):
        op = expr.op()
        out = op.map_clear(replace_cache_table)

        return super()._to_sqlglot(out.to_expr(), limit=limit, params=params)

    def _build_insert_template(
        self,
        name,
        *,
        schema: sch.Schema,
        catalog: str | None = None,
        columns: bool = False,
        placeholder: str = "?",
    ) -> str:
        """Builds an INSERT INTO table VALUES query string with placeholders.

        Parameters
        ----------
        name
            Name of the table to insert into
        schema
            Ibis schema of the table to insert into
        catalog
            Catalog name of the table to insert into
        columns
            Whether to render the columns to insert into
        placeholder
            Placeholder string. Can be a format string with a single `{i}` spec.

        Returns
        -------
        str
            The query string
        """
        quoted = self.compiler.quoted
        return sge.insert(
            sge.Values(
                expressions=[
                    sge.Tuple(
                        expressions=[
                            sge.Var(this=placeholder.format(i=i))
                            for i in range(len(schema))
                        ]
                    )
                ]
            ),
            into=sg.table(name, catalog=catalog, quoted=quoted),
            columns=(
                map(partial(sg.to_identifier, quoted=quoted), schema.keys())
                if columns
                else None
            ),
        ).sql(self.dialect)

    def read_record_batches(
        self,
        record_batches: pa.RecordBatchReader,
        table_name: str | None = None,
        password: str | None = None,
        temporary: bool = False,
        mode: str = "create",
        **kwargs: Any,
    ) -> ir.Table:
        from letsql.common.utils.postgres_utils import (
            PgADBC,
            make_table_temporary,
        )

        pgadbc = PgADBC(self, password)
        pgadbc.adbc_ingest(table_name, record_batches, mode=mode, **kwargs)
        if temporary:
            make_table_temporary(self, table_name)
        return self.table(table_name)

    def read_parquet(
        self,
        path: str | Path,
        table_name: str | None = None,
        password: str | None = None,
        temporary: bool = False,
        mode: str = "create",
        **kwargs: Any,
    ) -> ir.Table:
        if table_name is None:
            if not temporary:
                raise ValueError(
                    "If `table_name` is not provided, `temporary` must be True"
                )
            else:
                table_name = gen_name("ls-read-parquet")
        record_batches = pq.ParquetFile(path).iter_batches()
        return self.read_record_batches(
            record_batches=record_batches,
            table_name=table_name,
            password=password,
            temporary=temporary,
            mode=mode,
            **kwargs,
        )

    def read_csv(
        self,
        path,
        table_name=None,
        chunksize=10_000,
        password=None,
        temporary=False,
        mode="create",
        schema=None,
        **kwargs,
    ):
        if chunksize is None:
            raise ValueError
        if table_name is None:
            if not temporary:
                raise ValueError(
                    "If `table_name` is not provided, `temporary` must be True"
                )
            else:
                table_name = gen_name("ls-read-csv")
        record_batches = read_csv_rbr(path, schema=schema, **kwargs)
        return self.read_record_batches(
            record_batches=record_batches,
            table_name=table_name,
            password=password,
            temporary=temporary,
            mode=mode,
            **kwargs,
        )
