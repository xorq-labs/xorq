from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

import pyarrow as pa
import sqlglot as sg
import sqlglot.expressions as sge
from attr import field, frozen
from batchcorder import StreamCache
from parsy import ParseError

import xorq.vendor.ibis.expr.datatypes as dt
from xorq.common.exceptions import XorqError
from xorq.expr.remote_table_exec import bind_scope_to_reader
from xorq.vendor.ibis.backends.duckdb import Backend as IbisDuckDBBackend
from xorq.vendor.ibis.common.annotations import SignatureValidationError
from xorq.vendor.ibis.expr import types as ir
from xorq.vendor.ibis.util import gen_name


__all__ = [
    "Backend",
]


# a fresh cursor of the connection sees none of its TEMP objects and LOCAL
# settings, so it streams only what reads persistent relations alone
RELATIONS_SQL = """
SELECT * FROM (
    SELECT lower(table_name) AS name, temporary, NULL AS sql FROM duckdb_tables()
    UNION ALL
    SELECT lower(view_name), temporary, sql FROM duckdb_views() WHERE NOT internal
) WHERE list_contains($names, name)
"""
LOCAL_SETTINGS_SQL = (
    "SELECT name, value FROM duckdb_settings()"
    " WHERE scope = 'LOCAL' AND value IS NOT NULL"
)


@frozen
class ConnectionStream:
    """A remote stream left on the connection, open until
    ``bind_scope_to_reader`` closes it on read-through or collection."""

    closed: threading.Event = field(factory=threading.Event)

    def close(self) -> None:
        self.closed.set()


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
    _connection_stream: ConnectionStream | None = None

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
        if isinstance(source, pa.RecordBatchReader):
            # A reader is read-once; computing the cache key reads the rows, so
            # register a StreamCache that later scans can read again.
            source = StreamCache(source)
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
        self._check_no_connection_stream()
        return self._to_duckdb_relation(
            expr, params=params, limit=limit
        ).fetch_arrow_reader(chunk_size)

    def _to_isolated_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 10_000,
        **_: Any,
    ) -> pa.ipc.RecordBatchReader:
        # duckdb keeps one pending result per connection: a later statement on
        # it empties a stream there without an error (GH #1733), so a remote
        # stream goes on its own cursor when the cursor reads alike
        self._check_no_connection_stream()
        sql = self._to_duckdb_sql(expr, params=params, limit=limit)
        if not (self._reads_persistent_relations(sql) and not self._in_transaction()):
            self._connection_stream = ConnectionStream()
            reader = self.con.sql(sql).fetch_arrow_reader(chunk_size)
            return bind_scope_to_reader(self._connection_stream, reader)
        cursor = self.con.cursor()
        defaults = frozenset(cursor.execute(LOCAL_SETTINGS_SQL).fetchall())
        settings = frozenset(self.con.execute(LOCAL_SETTINGS_SQL).fetchall())
        for name, value in settings - defaults:
            cursor.execute(f"SET {name} = {sge.convert(value).sql(self.dialect)}")
        reader = cursor.sql(sql).fetch_arrow_reader(chunk_size)
        return bind_scope_to_reader(cursor, reader)

    def _check_no_connection_stream(self) -> None:
        match self._connection_stream:
            case ConnectionStream(closed=closed) if not closed.is_set():
                raise XorqError(
                    "DuckDB streams one result per connection, and a remote"
                    " stream on this connection is still open: it reads"
                    " relations other than persistent tables and views, which a"
                    " separate cursor cannot see. Another statement on the"
                    " connection would silently empty that stream. Read"
                    " persistent tables or views, or execute one side first."
                )

    def _reads_persistent_relations(
        self, sql: str, seen: frozenset[str] = frozenset()
    ) -> bool:
        try:
            parsed = sg.parse_one(sql, read=self.dialect)
        except sg.errors.ParseError:
            return False
        # a view's SQL is its CREATE statement; only its query reads relations
        query = parsed.expression if isinstance(parsed, sge.Create) else parsed
        tables = tuple(query.find_all(sge.Table))
        ctes = frozenset(cte.alias_or_name.lower() for cte in query.find_all(sge.CTE))
        names = frozenset(table.name.lower() for table in tables) - ctes
        # a qualified name still matches every relation of that name, so a
        # TEMP object named like a persistent one keeps the stream here
        found = self.con.execute(RELATIONS_SQL, {"names": list(names)}).fetchall()
        # a persistent view may read TEMP objects, which a cursor cannot see
        return (
            all(isinstance(table.this, sge.Identifier) for table in tables)
            and frozenset(name for name, *_ in found) == names
            and not any(temporary for _, temporary, _ in found)
            and all(
                self._reads_persistent_relations(view_sql, seen | {name})
                for name, _, view_sql in found
                if view_sql is not None and name not in seen
            )
        )

    def _in_transaction(self) -> bool:
        # each autocommit statement gets a new id; BEGIN ... COMMIT shares one
        first, second = (
            self.con.execute("SELECT txid_current()").fetchall() for _ in range(2)
        )
        return first == second

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
