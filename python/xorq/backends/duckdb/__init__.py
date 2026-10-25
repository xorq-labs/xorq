from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pyarrow as pa
import sqlglot as sg
import sqlglot.expressions as sge
from batchcorder import StreamCache
from parsy import ParseError

import xorq.vendor.ibis.expr.datatypes as dt
from xorq.vendor.ibis.backends.duckdb import Backend as IbisDuckDBBackend
from xorq.vendor.ibis.common.annotations import SignatureValidationError
from xorq.vendor.ibis.expr import types as ir
from xorq.vendor.ibis.util import gen_name


__all__ = [
    "Backend",
]


def materialize_ctes_reading(
    query: sge.Expression, names: frozenset[str]
) -> sge.Expression:
    # duckdb inlines a CTE referenced twice under a LIMIT, scanning its body once
    # per reference; a registered stream is sized for (or only survives) one
    # scan per CTE, so the CTEs that read one are forced MATERIALIZED
    def mark(node: sge.Expression) -> sge.Expression:
        if isinstance(node, sge.CTE) and any(
            table.name in names for table in node.this.find_all(sge.Table)
        ):
            node.set("materialized", True)
        return node

    # transform copies the tree before marking it: to_sqlglot's tree is a DAG
    # and must not be edited in place
    return query.transform(mark) if names and query.ctes else query


def parse_column_type(name, typ, type_mapper) -> tuple[str, dt.DataType]:
    """DuckDB type string and ibis dtype for `typ`: an ibis dtype, an ibis type
    string, or a DuckDB type string ibis can represent. Ibis spellings win; a
    string ibis can't parse goes to DuckDB verbatim, so `TIMETZ` survives."""
    try:
        dtype, verbatim = dt.dtype(typ), None
    except (ParseError, TypeError, ValueError, SignatureValidationError):
        dtype, verbatim = None, typ
    try:
        if dtype is None:
            dtype = type_mapper.from_string(typ)
        # KeyError on `unknown` at any depth, e.g. BIT, BIT[], STRUCT(x BIT)
        sql_type = type_mapper.to_string(dtype)
    except (
        KeyError,
        ValueError,
        TypeError,
        SignatureValidationError,
        sg.errors.SqlglotError,
    ):
        raise ValueError(
            f"Column {name!r} has type {typ!r}, which is not an ibis type"
            " or a DuckDB type ibis can represent"
        ) from None
    return verbatim or sql_type, dtype


class Backend(IbisDuckDBBackend):
    # names read_record_batches registered a stream under; a pa.Table can be
    # scanned any number of times and is not recorded
    _stream_names: frozenset[str] = frozenset()

    def _to_sqlglot(
        self,
        expr: ir.Expr,
        *,
        limit: str | None = None,
        params: Mapping[ir.Expr, Any] | None = None,
    ) -> sge.Expression:
        query = super()._to_sqlglot(expr, limit=limit, params=params)
        return materialize_ctes_reading(query, self._stream_names)

    def execute(
        self,
        expr: ir.Expr,
        params: Mapping | None = None,
        limit: str | None = "default",
        **_: Any,
    ) -> Any:
        batch_reader = self.to_pyarrow_batches(expr, params=params, limit=limit)
        return expr.__pandas_result__(
            batch_reader.read_pandas(timestamp_as_object=True)
        )

    def read_record_batches(
        self,
        source: pa.Table | pa.RecordBatchReader | StreamCache,
        table_name: str | None = None,
    ) -> ir.Table:
        # duckdb registers ``source`` (typically a StreamCache) directly so it
        # can replay the stream across scans; a casting wrapper would not be
        # replayable, so casting to the logical schema happens upstream, before
        # the StreamCache, in the remote pass (REMOTE_PASS / _make_remote_replacer).
        table_name = table_name or gen_name("read_record_batches")
        self.con.register(table_name, source)
        if not isinstance(source, pa.Table):
            self._stream_names = self._stream_names | {table_name}
        return self.table(table_name)

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 10_000,
        **_: Any,
    ) -> pa.ipc.RecordBatchReader:
        return self._to_duckdb_relation(
            expr, params=params, limit=limit
        ).fetch_arrow_reader(chunk_size)

    def _type_struct(self, mapping: Mapping[str, str | dt.DataType]) -> sge.Struct:
        type_mapper = self.compiler.type_mapper
        parsed = {
            name: parse_column_type(name, typ, type_mapper)
            for name, typ in mapping.items()
        }
        if any(dtype.is_geospatial() for _, dtype in parsed.values()):
            self._load_extensions(["spatial"])
        return sge.Struct(
            expressions=[
                sge.PropertyEQ(
                    this=sge.to_identifier(name), expression=sge.convert(sql_type)
                )
                for name, (sql_type, _) in parsed.items()
            ]
        )

    def _type_options(self, **mappings) -> dict[str, sge.Struct]:
        return {
            key: self._type_struct(mapping)
            for key, mapping in mappings.items()
            if mapping is not None
        }

    def read_csv(
        self,
        paths: str | list[str] | tuple[str],
        /,
        *,
        table_name: str | None = None,
        columns: Mapping[str, str | dt.DataType] | None = None,
        types: Mapping[str, str | dt.DataType] | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """`columns` (all columns) and `types` (a subset) map names to ibis
        dtypes, ibis type strings or DuckDB type strings. A name valid in both
        is read as ibis (`INT8` is int8, not BIGINT); `T[]` is read as DuckDB.
        Types ibis cannot represent raise a `ValueError` naming the column."""
        # uppercase COLUMNS bypasses the parent's `columns`, which re-parses
        # and emits types unquoted; DuckDB option names are case-insensitive
        return super().read_csv(
            paths,
            table_name=table_name,
            **{
                "auto_detect": columns is None,
                **kwargs,
                **self._type_options(COLUMNS=columns, dtypes=types),
            },
        )

    def read_json(
        self,
        paths: str | list[str] | tuple[str],
        /,
        *,
        table_name: str | None = None,
        columns: Mapping[str, str | dt.DataType] | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """`columns` follows the `read_csv` rules: `INT8` is int8, not DuckDB's
        BIGINT, and types ibis cannot represent (`UHUGEINT`, `BIT`) raise a
        `ValueError` naming the column."""
        return super().read_json(
            paths,
            table_name=table_name,
            **{**kwargs, **self._type_options(COLUMNS=columns or None)},
        )
