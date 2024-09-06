from functools import partial
from pathlib import Path
from typing import Mapping, Any

import ibis
import ibis.common.exceptions as exc
import ibis.expr.schema as sch
import numpy as np
import sqlglot as sg
import sqlglot.expressions as sge
from ibis.backends.postgres import Backend as IbisPostgresBackend
from ibis.expr import types as ir
from pandas.api.types import is_float_dtype

from letsql.expr.relations import CachedNode, replace_cache_table


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

    def read_parquet(
        self, path: str | Path, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        import pyarrow.parquet as pq
        from psycopg2.extras import execute_batch

        pyarrow_schema = pq.read_schema(path, memory_map=True)
        schema = ibis.Schema.from_pyarrow(pyarrow_schema)

        if null_columns := [col for col, dtype in schema.items() if dtype.is_null()]:
            raise exc.IbisTypeError(
                f"{self.name} cannot yet reliably handle `null` typed columns; "
                f"got null typed columns: {null_columns}"
            )

        if table_name not in self.list_tables():
            quoted = self.compiler.quoted
            column_defs = [
                sg.exp.ColumnDef(
                    this=sg.to_identifier(colname, quoted=quoted),
                    kind=self.compiler.type_mapper.from_ibis(typ),
                    constraints=(
                        None
                        if typ.nullable
                        else [
                            sg.exp.ColumnConstraint(
                                kind=sg.exp.NotNullColumnConstraint()
                            )
                        ]
                    ),
                )
                for colname, typ in schema.items()
            ]

            create_stmt = sg.exp.Create(
                kind="TABLE",
                this=sg.exp.Schema(
                    this=sg.to_identifier(table_name, quoted=quoted),
                    expressions=column_defs,
                ),
                properties=sg.exp.Properties(expressions=[sge.TemporaryProperty()]),
            )
            create_stmt_sql = create_stmt.sql(self.dialect)

            df = pq.read_table(path).to_pandas()
            # nan gets compiled into 'NaN'::float which throws errors in non-float columns
            # In order to hold NaN values, pandas automatically converts integer columns
            # to float columns if there are NaN values in them. Therefore, we need to convert
            # them to their original dtypes (that support pd.NA) to figure out which columns
            # are actually non-float, then fill the NaN values in those columns with None.
            convert_df = df.convert_dtypes()
            for col in convert_df.columns:
                if not is_float_dtype(convert_df[col]):
                    df[col] = df[col].replace(np.nan, None)

            data = df.itertuples(index=False)
            sql = self._build_insert_template(
                table_name, schema=schema, columns=True, placeholder="%s"
            )

            with self.begin() as cur:
                cur.execute(create_stmt_sql)
                execute_batch(cur, sql, data, 128)

        return self.table(table_name)

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
