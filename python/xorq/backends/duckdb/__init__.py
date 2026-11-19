from __future__ import annotations

import functools
import threading
import weakref
from collections.abc import Mapping, Sequence, Set
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


# what each relation a query names is on the connection; a fresh cursor of it
# sees the persistent ones only
CATALOG_SQL = """
SELECT * FROM (
    SELECT 'table' AS kind, database_name, schema_name, table_name AS name,
        table_oid AS oid, temporary
    FROM duckdb_tables()
    UNION ALL
    SELECT 'view', database_name, schema_name, view_name, view_oid, temporary
    FROM duckdb_views() WHERE NOT internal
) WHERE list_contains($names, lower(name))
"""
# a registered object is a TEMP view too, with a new oid when replaced
TEMP_VIEW_OID_SQL = (
    "SELECT view_oid FROM duckdb_views() WHERE temporary AND view_name = $name"
)
SESSION_SETTINGS_SQL = (
    "SELECT name, value FROM duckdb_settings()"
    " WHERE scope = 'LOCAL' AND value IS NOT NULL"
)


@frozen
class CatalogEntry:
    kind: str
    catalog: str
    schema: str
    name: str
    oid: int
    temporary: bool

    def is_named_by(self, table: sge.Table) -> bool:
        # duckdb names are case-insensitive
        return (
            table.name.lower() == self.name.lower()
            and table.db.lower() in ("", self.schema.lower())
            and table.catalog.lower() in ("", self.catalog.lower())
        )


def cursor_blocker(
    table: sge.Table,
    entries: tuple[CatalogEntry, ...],
    ctes: frozenset[str],
    xorq_oids: Set[int],
) -> str | None:
    """Why a fresh cursor can't read ``table`` as the connection does, or None:
    it reads a persistent table, or a TEMP object xorq created and replays."""
    if not isinstance(table.this, sge.Identifier):
        return f"table function {table.this.sql('duckdb')}"
    found = tuple(entry for entry in entries if entry.is_named_by(table))
    # a TEMP object shadows every persistent one of its name
    match tuple(entry for entry in found if entry.temporary) or found:
        case () if table.name.lower() in ctes:
            return None
        case ():
            return f"unknown relation {'.'.join(part.name for part in table.parts)}"
        case (CatalogEntry(temporary=True) as entry, *_) if table.name.lower() in ctes:
            return f"CTE named like TEMP {entry.kind} {entry.name}"
        case (entry,) if entry.oid in xorq_oids:
            return None
        case (CatalogEntry(kind="table", temporary=True) as entry, *_):
            return f"TEMP table {entry.name}"
        case _ if all(entry.kind == "table" for entry in found):
            return None
        case (entry, *_):
            return f"{'TEMP ' * entry.temporary}view {entry.name} not created by xorq"


@frozen
class ConnectionStream:
    """A stream on the connection itself, live until read through or
    collected: ``close`` is the hook ``bind_scope_to_reader`` calls."""

    blockers: tuple[str, ...]
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
        self._register(table_name, source)
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
        # duckdb keeps one pending result per connection: any later statement
        # on ``self.con`` empties a stream on it without an error (GH #1733),
        # so a stream goes on its own cursor whenever the cursor reads alike.
        # Building any stream runs statements on the connection, so none may
        # start while a stream left on it is open.
        match self._connection_stream:
            case ConnectionStream(blockers=blockers, closed=closed) if (
                not closed.is_set()
            ):
                raise XorqError(
                    "DuckDB streams one result per connection, and a stream on"
                    " this connection is still open: it reads"
                    f" {', '.join(blockers)}, which a separate cursor cannot see."
                    " Another statement on the connection would silently empty"
                    " that stream. Read persistent tables, create the data"
                    " through xorq (read_* methods, memtables), or execute one"
                    " side first."
                )
        sql = self._to_duckdb_sql(expr, params=params, limit=limit)
        sources, blockers = self._cursor_sources(sql)
        if blockers:
            reader = self.con.sql(sql).fetch_arrow_reader(chunk_size)
            self._connection_stream = ConnectionStream(blockers)
            return bind_scope_to_reader(self._connection_stream, reader)
        cursor = self.con.cursor()
        for name, value in self._session_settings():
            cursor.execute(f"SET {name} = {sge.convert(value).sql(self.dialect)}")
        for name, source in sources:
            match source:
                case str():
                    cursor.execute(source)
                case _:
                    cursor.register(name, source)
        reader = cursor.sql(sql).fetch_arrow_reader(chunk_size)
        return bind_scope_to_reader(cursor, reader)

    def _post_connect(self, extensions: Sequence[str] | None = None) -> None:
        # what xorq put in the TEMP catalog: the oid and CREATE statement of a
        # read_* view by name (a re-read replaces it), or the object
        # registered by oid (weakly: duckdb holds it)
        self._xorq_views = {}
        self._xorq_objects = weakref.WeakValueDictionary()
        self._connection_stream = None
        super()._post_connect(extensions)

    def _register(self, name: str, obj: Any) -> None:
        super()._register(name, obj)
        self._xorq_objects[self._temp_view_oid(name)] = obj

    def _create_temp_view(self, table_name, source) -> None:
        super()._create_temp_view(table_name, source)
        self._xorq_views[table_name] = (
            self._temp_view_oid(table_name),
            self._get_temp_view_definition(table_name, source).sql(self.name),
        )

    def _temp_view_oid(self, name: str) -> int:
        ((oid,),) = self.con.execute(TEMP_VIEW_OID_SQL, {"name": name}).fetchall()
        return oid

    @functools.cached_property
    def _cursor_settings(self) -> frozenset[tuple[str, str]]:
        with self.con.cursor() as cursor:
            return frozenset(cursor.execute(SESSION_SETTINGS_SQL).fetchall())

    def _session_settings(self) -> frozenset[tuple[str, str]]:
        """The connection's settings a fresh cursor lacks, e.g. ``TimeZone``."""
        settings = self.con.execute(SESSION_SETTINGS_SQL).fetchall()
        return frozenset(settings) - self._cursor_settings

    def _in_transaction(self) -> bool:
        # each autocommit statement gets a new id; BEGIN ... COMMIT shares one
        first, second = (
            self.con.execute("SELECT txid_current()").fetchall() for _ in range(2)
        )
        return first == second

    def _cursor_sources(
        self, sql: str
    ) -> tuple[tuple[tuple[str, Any], ...], tuple[str, ...]]:
        """The xorq TEMP objects a cursor replays to read ``sql``, as (name,
        CREATE statement or object), and why it can't read ``sql``: relations
        it can't see or an open transaction."""
        try:
            parsed = sg.parse_one(sql, read=self.dialect)
        except sg.errors.ParseError:
            return (), ("SQL sqlglot cannot parse",)
        tables = tuple(parsed.find_all(sge.Table))
        ctes = frozenset(cte.alias_or_name.lower() for cte in parsed.find_all(sge.CTE))
        rows = self.con.execute(
            CATALOG_SQL, {"names": [table.name.lower() for table in tables]}
        ).fetchall()
        # the name filter in SQL ignores qualifiers: a TEMP view named like a
        # qualified persistent table is neither read nor replayed
        entries = tuple(
            entry
            for entry in (CatalogEntry(*row) for row in rows)
            if any(entry.is_named_by(table) for table in tables)
        )
        views = dict(self._xorq_views.values())
        sources = {
            entry.oid: (entry.name, source)
            for entry in entries
            if (source := views.get(entry.oid, self._xorq_objects.get(entry.oid)))
            is not None
        }
        blockers = (
            *("an open transaction",) * self._in_transaction(),
            *(cursor_blocker(table, entries, ctes, sources.keys()) for table in tables),
        )
        return tuple(sources.values()), tuple(dict.fromkeys(filter(None, blockers)))

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
