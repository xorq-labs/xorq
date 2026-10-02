from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import sqlglot as sg
import sqlglot.expressions as sge
from batchcorder import StreamCache
from parsy import ParseError

import xorq.vendor.ibis.expr.datatypes as dt
from xorq.vendor.ibis import util
from xorq.vendor.ibis.backends.duckdb import Backend as IbisDuckDBBackend
from xorq.vendor.ibis.backends.sql.compilers.base import STAR, C
from xorq.vendor.ibis.common.annotations import SignatureValidationError
from xorq.vendor.ibis.expr import types as ir
from xorq.vendor.ibis.util import gen_name


if TYPE_CHECKING:
    from xorq.vendor.ibis.backends.sql.datatypes import SqlglotType


__all__ = [
    "Backend",
]


def _contains_unknown(dtype: dt.DataType) -> bool:
    if dtype.is_array():
        return _contains_unknown(dtype.value_type)
    if dtype.is_map():
        return _contains_unknown(dtype.key_type) or _contains_unknown(dtype.value_type)
    if dtype.is_struct():
        return any(map(_contains_unknown, dtype.types))
    return dtype.is_unknown()


def parse_column_type(
    name: str, typ: str | dt.DataType, type_mapper: type[SqlglotType]
) -> tuple[str, dt.DataType]:
    """Return the DuckDB type string for `typ` and its ibis dtype.

    Ibis spellings win; a string ibis can't parse is passed to DuckDB
    verbatim, so types ibis maps lossily (e.g. `TIMETZ`) survive. DuckDB
    types ibis maps to unknown anywhere (e.g. `BIT`, `BIT[]`) are rejected.
    """
    verbatim = None
    try:
        dtype = dt.dtype(typ)
    except (ParseError, ValueError, SignatureValidationError):
        verbatim = typ
        try:
            dtype = type_mapper.from_string(typ)
        except (
            KeyError,
            ValueError,
            TypeError,
            SignatureValidationError,
            sg.errors.SqlglotError,
        ):
            dtype = dt.unknown
    if _contains_unknown(dtype):
        raise ValueError(
            f"Column {name!r} has type {typ!r}, which is not an ibis type or a DuckDB type ibis can represent"
        )
    return verbatim or type_mapper.to_string(dtype), dtype


class Backend(IbisDuckDBBackend):
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

    def _column_type_strings(
        self, mapping: Mapping[str, str | dt.DataType]
    ) -> dict[str, str]:
        parsed = {
            name: parse_column_type(name, typ, self.compiler.type_mapper)
            for name, typ in mapping.items()
        }
        if any(dtype.is_geospatial() for _, dtype in parsed.values()):
            self._load_extensions(["spatial"])
        return {name: sql_type for name, (sql_type, _) in parsed.items()}

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
        """Register a CSV file as a table in the current database.

        `columns` (all columns) and `types` (a subset) map column names to
        ibis types, ibis type strings, or DuckDB type strings ibis can
        represent (not e.g. `BIT`). A string valid in both keeps its ibis
        meaning: `INT`, `FLOAT` and `INT8` are int64, float64 and int8; spell
        DuckDB widths `INTEGER`, `REAL` and `BIGINT`. `T[]` is always a DuckDB
        type (`int[]` is an array of int32); use `array<T>` for the ibis
        element meaning. See the parent method for the rest.
        """
        paths = util.normalize_filenames(paths)
        table_name = table_name or util.gen_name("read_csv")

        if any(source.startswith(("http://", "https://", "s3://")) for source in paths):
            self._load_extensions(["httpfs"])

        kwargs.setdefault("header", True)
        # auto_detect and columns collide, so auto_detect defaults to True
        # unless columns is given
        kwargs["auto_detect"] = kwargs.pop("auto_detect", columns is None)
        options = [C[key].eq(sge.convert(value)) for key, value in kwargs.items()]
        for key, mapping in (("columns", columns), ("types", types)):
            if mapping is not None:
                # DuckDB wants a STRUCT of type strings; sqlglot makes a MAP of a dict
                struct = sge.Struct(
                    expressions=[
                        sge.PropertyEQ(
                            this=sge.to_identifier(name), expression=sge.convert(t)
                        )
                        for name, t in self._column_type_strings(mapping).items()
                    ]
                )
                options.append(C[key].eq(struct))
        self._create_temp_view(
            table_name,
            sg.select(STAR).from_(self.compiler.f.read_csv(paths, *options)),
        )
        return self.table(table_name)

    def read_json(
        self,
        paths: str | list[str] | tuple[str],
        /,
        *,
        table_name: str | None = None,
        columns: Mapping[str, str | dt.DataType] | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Read newline-delimited JSON into an ibis table.

        `columns` accepts the same type spellings as `read_csv`'s `columns`,
        including its rule that a name valid in both ibis and DuckDB keeps
        its ibis meaning. That differs from passing the string to DuckDB
        directly: `INT8` is int8 (TINYINT), not DuckDB's 64-bit BIGINT, so
        `columns={"id": "INT8"}` fails to cast an id over 127; likewise `INT`
        and `FLOAT` are int64 and float64. Spell DuckDB widths `BIGINT`,
        `INTEGER` and `REAL`. See the parent method for the rest.
        """
        if columns:
            columns = self._column_type_strings(columns)
        return super().read_json(
            paths, table_name=table_name, columns=columns, **kwargs
        )
