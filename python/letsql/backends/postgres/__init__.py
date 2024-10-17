from functools import partial
from pathlib import Path
from typing import Mapping, Any

import ibis.expr.schema as sch
import sqlglot as sg
import sqlglot.expressions as sge
from ibis.backends.postgres import Backend as IbisPostgresBackend
from ibis.expr import types as ir
from ibis.util import (
    gen_name,
)

from letsql.expr.relations import CachedNode, replace_cache_table
from letsql.common.utils.defer_utils import (
    read_csv_rbr,
)


class Backend(IbisPostgresBackend):
    _top_level_methods = ("connect_examples", "connect_env")

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
                uncached_to_expr = uncached.to_expr()
                node = storage.set_default(uncached_to_expr, uncached)
            return node

        op = expr.op()
        out = op.replace(fn)

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

    def read_parquet(
        self,
        path: str | Path,
        table_name: str | None = None,
        password: str | None = None,
        temporary: bool = False,
        **kwargs: Any,
    ) -> ir.Table:
        import pyarrow.parquet as pq
        from letsql.common.utils.postgres_utils import (
            PgADBC,
            make_table_temporary,
        )

        if table_name is None:
            if not temporary:
                raise ValueError(
                    "If `table_name` is not provided, `temporary` must be True"
                )
            else:
                table_name = gen_name("ls-read-parquet")
        rbr = pq.ParquetFile(path).iter_batches()
        pgadbc = PgADBC(self, password)
        pgadbc.adbc_ingest(table_name, rbr)
        if temporary:
            make_table_temporary(self, table_name)
        return self.table(table_name)

    def read_csv(
        self,
        path,
        table_name=None,
        chunksize=10_000,
        password=None,
        temporary=False,
        **kwargs,
    ):
        from letsql.common.utils.postgres_utils import (
            PgADBC,
            make_table_temporary,
        )

        if chunksize is None:
            raise ValueError
        if table_name is None:
            if not temporary:
                raise ValueError(
                    "If `table_name` is not provided, `temporary` must be True"
                )
            else:
                table_name = gen_name("ls-read-csv")
        pgadbc = PgADBC(self, password)
        pd_rbr = read_csv_rbr(path, **kwargs)
        pgadbc.adbc_ingest(table_name, pd_rbr)
        if temporary:
            make_table_temporary(self, table_name)
        return self.table(table_name)
