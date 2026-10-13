from __future__ import annotations

import importlib.util
from typing import Any

import psycopg
import pyarrow as pa
import sqlglot as sg
import sqlglot.expressions as sge
from psycopg.adapt import Loader

import xorq.common.exceptions as exc
import xorq.vendor.ibis.expr.operations as ops
import xorq.vendor.ibis.expr.schema as sch
from xorq.backends.postgres import Backend as PostgresBackend
from xorq.backends.redshift.compiler import compiler
from xorq.common.utils.logging_utils import get_logger
from xorq.common.utils.redshift_utils import session_temp_schema_of
from xorq.vendor.ibis.expr import datatypes as dt
from xorq.vendor.ibis.expr import types as ir


logger = get_logger(__name__)


__all__ = [
    "Backend",
]

# Redshift listens on 5439; the postgres backend defaults to 5432.
DEFAULT_PORT = 5439

# The modes ``adbc_ingest`` accepts -- and so the postgres backend's
# ``read_record_batches`` -- restated so this psycopg ingest accepts exactly
# the same set: callers such as ``defer_utils`` pass ``mode`` to either backend
# alike, so the two must not disagree about what it means.
INGEST_MODES = ("create", "append", "replace", "create_append")

# Modes that append to a table this call did not create, so ``temporary`` has
# nothing to apply to.
APPEND_ONLY_MODES = ("append", "create_append")

# Rows per ``executemany`` for a ``pa.Table``, which carries no batch size.
INGEST_CHUNKSIZE = 10_000

# Redshift's OID for ``VARBYTE`` in a result description.
VARBYTE_OID = 6551

# Alias for the derived table that ``_get_schema_using_query`` probes through.
# Fixed rather than generated: it names a subquery, which is scoped to the
# statement and cannot collide with anything in the catalog, and a deterministic
# alias makes the emitted SQL assertable.
PROBE_ALIAS = "redshift_probe"

# What ``sg.parse`` may yield for trailing noise rather than for a statement.
# ``sge.Semicolon`` is absent from sqlglot 23.6.3 -- within the supported range,
# and the version CI's lowest-direct job installs -- and present by 23.17.0.
# Where it is absent, a trailing comment attaches to the statement before it,
# so there is nothing to drop and an empty tuple matches nothing.
_NOT_A_STATEMENT = tuple(filter(None, (getattr(sge, "Semicolon", None),)))


class VarbyteLoader(Loader):
    """Decode ``VARBYTE``'s unprefixed hex text; see ``Backend._post_connect``."""

    def load(self, data: bytes | bytearray | memoryview) -> bytes:
        return bytes.fromhex(bytes(data).decode("ascii"))


class Backend(PostgresBackend):
    """Redshift Serverless, over the PostgreSQL wire protocol.

    Subclasses the *xorq* postgres backend rather than the vendored ibis one:
    the ADBC/psycopg ``to_pyarrow_batches`` and ``read_record_batches`` live on
    the xorq subclass, and the vendored base has no ``read_record_batches`` at
    all.
    """

    name = "redshift"
    compiler = compiler

    # ``_secret_keys`` is inherited, not restated: a literal copy drifts from
    # the ``con_name_to_secret_keys`` mirror, and ``()`` would narrow
    # ``check_for_exposed_secrets`` to just ``password``.

    # Deliberately empty. The postgres backend exposes ``connect_env`` and
    # ``connect_examples``, and both are inherited by a plain subclass:
    # ``connect_env`` is backed by ``PostgresConfig`` and the postgres env
    # template, and ``connect_examples`` is hardcoded to a public postgres box.
    # Exposing either on the Redshift namespace would offer a method that does
    # not connect to Redshift. Redshift's own auth modes arrive with the
    # authenticator work.
    _top_level_methods = ()

    def _clone_credential_default_password(self) -> str | None:
        """Redshift has no environment default, and must not borrow one.

        The same reasoning that empties ``_top_level_methods`` above: the
        inherited fallback is ``$POSTGRES_PASSWORD``, so a Redshift connection
        with no password in ``_con_kwargs`` -- which is every connection built
        by ``from_connection`` -- either refused with a message naming a
        service the caller never used, or, on a developer machine where
        ``POSTGRES_PASSWORD`` happens to be set, dialled the *warehouse* with
        a local postgres password.

        ``None`` rather than a ``$REDSHIFT_PASSWORD`` of our own invention:
        that would be a new public convention, and nothing else in xorq reads
        such a variable. Redshift's own auth modes arrive with the
        authenticator work, and this hook is where they will attach.
        """
        return None

    def do_connect(
        self,
        host: str | None = None,
        user: str | None = None,
        password: str | None = None,
        port: int = DEFAULT_PORT,
        database: str | None = None,
        schema: str | None = None,
        autocommit: bool = True,
        **kwargs: Any,
    ) -> None:
        """Connect to Redshift.

        Identical to the postgres backend except for the default port and for
        ``client_encoding``, which is mandatory rather than a nicety: Redshift
        reports its encoding as ``UNICODE``, the PostgreSQL 8.x alias, which is
        absent from psycopg3's codec map. Without this every query -- not just
        non-ASCII ones -- raises ``NotSupportedError``.

        Defaulting it here rather than in the caller keeps it out of
        ``_con_kwargs``, which is captured from the caller's arguments, so it
        never reaches the profile or the build hash; a caller's explicit value
        still wins. ``prepare_threshold`` is set in ``_post_connect``, which
        every construction path reaches.
        """
        kwargs.setdefault("client_encoding", "utf8")
        return super().do_connect(
            host=host,
            user=user,
            password=password,
            port=port,
            database=database,
            schema=schema,
            autocommit=autocommit,
            **kwargs,
        )

    def _post_connect(self) -> None:
        """Make the connection safe for Redshift, then run postgres's setup.

        The one hook ``connect``, ``from_connection`` and ``clone`` all reach,
        so the invariants live here rather than in ``do_connect``, which
        ``from_connection`` skips.

        ``prepare_threshold`` becomes ``None`` unless the caller passed one:
        Redshift has no ``DEALLOCATE ALL``, which psycopg sends to clear its
        prepared statements whenever a transaction rolls back, and there the
        syntax error rolls back a ``drop_table`` and leaves its table behind.
        With no threshold psycopg never prepares, so it has nothing to
        deallocate. A ``from_connection`` backend has no caller kwargs, so its
        connection gets ``None`` whatever it carried.

        The encoding is checked first because it cannot be repaired here:
        setting it takes a query, and a connection Redshift reports as
        ``UNICODE`` cannot run one. Such a connection is refused before any
        SQL, naming the setting to open it with.

        It then teaches the connection to read ``VARBYTE``. A result
        description reports ``VARBYTE`` as OID 6551, which psycopg's registry
        does not know, so it hands the value back as the text Redshift sends:
        hex digits with no prefix, ``'ab'`` for ``b"\\xab"``. The column is
        typed binary on both introspection paths, so the Arrow cast then
        turned that text into its *ASCII* bytes -- ``b"ab"`` -- silently, on
        every read the psycopg baseline served. Measured on a live warehouse:
        the ADBC path returned ``b"\\xab"`` and psycopg ``b"ab"`` for the same
        row. The loader decodes the hex.
        """
        con = self.con
        try:
            con.info.encoding
        except psycopg.NotSupportedError as e:
            raise ValueError(
                "this connection's client encoding is not one psycopg can "
                "decode; open it with client_encoding='utf8'"
            ) from e
        if "prepare_threshold" not in self._con_kwargs:
            con.prepare_threshold = None
        super()._post_connect()
        con.adapters.register_loader(VARBYTE_OID, VarbyteLoader)

    @property
    def current_database(self) -> str:
        """The current schema.

        Overridden because the inherited implementation emits bare
        ``SELECT CURRENT_SCHEMA``, which Redshift rejects with
        ``UndefinedColumn: column "current_schema" does not exist``. Redshift
        requires the parenthesised call.

        No dialect swap fixes this: at sqlglot 28.6.0,
        ``sg.func("current_schema")`` renders without parentheses under
        sqlglot's Postgres *and* Redshift dialects, and only the default
        generator parenthesises it. That is a *version-qualified* claim, not a
        standing one -- at 23.6.3, the floor ``--resolution lowest-direct``
        picks under the declared ``sqlglot>=23.4``, ``sg.func`` parenthesises
        everywhere, and there this override is redundant rather than wrong.
        ``Anonymous`` forces the call form at both versions and under every
        dialect measured, which is why the test asserts what this method emits
        rather than how ``sg.func`` renders.

        ``current_catalog`` needs no such treatment -- ``CURRENT_DATABASE()``
        already renders with parentheses.
        """
        sql = sg.select(sge.Anonymous(this="current_schema")).sql(self.dialect)
        con = self.con
        with con.cursor() as cursor, con.transaction():
            [(schema,)] = cursor.execute(sql).fetchall()
        return schema

    # ------------------------------------------------------------------
    # Introspection.
    #
    # Both inherited implementations are PostgreSQL-only in ways that stay
    # invisible until a statement reaches the server: each compiles cleanly
    # under the postgres dialect and then fails on Redshift.
    # ------------------------------------------------------------------

    # ``svv_all_columns`` is Redshift's own union of native, datashare and
    # external columns. Its ``data_type`` is the *unparameterised* SQL name
    # (``numeric``, ``character varying``), with the modifiers split out into
    # ``numeric_precision``/``numeric_scale``, so a decimal's type string has
    # to be reassembled -- see ``_type_string``.
    #
    # ``character_maximum_length`` is deliberately *not* selected. It looks like
    # the char-type counterpart of the numeric pair and is not usable: ibis's
    # ``dt.String`` carries no length, so ``from_string("character varying(256)")``
    # and ``from_string("character varying")`` return the identical dtype. A
    # length read here could only be discarded one call later.
    #
    # Every value is bound rather than interpolated. Notably this is *not*
    # ``schema_name = ANY(%(dbs)s)``, which is the form the inherited query
    # uses: Redshift has no array type.
    #
    # The catalog and schema defaults are resolved *server side* with
    # ``COALESCE(%(x)s, current_database())`` rather than by reading
    # ``self.current_catalog``/``self.current_database`` first. Those are two
    # extra round trips on every single ``con.table()``, and on Serverless each
    # one is visible latency; the COALESCE costs nothing and cannot disagree
    # with the session the catalog query itself runs in.
    _COLUMN_FIELDS = """\
  column_name,
  data_type,
  is_nullable,
  numeric_precision,
  numeric_scale"""

    _SVV_ALL_COLUMNS_QUERY = f"""\
SELECT
{_COLUMN_FIELDS}
FROM svv_all_columns
WHERE database_name = COALESCE(%(catalog)s, current_database())
  AND schema_name = COALESCE(%(schema)s, current_schema())
  AND table_name = %(table)s
ORDER BY ordinal_position ASC"""

    # The temporary-table counterpart. ``svv_columns`` rather than
    # ``svv_all_columns`` because only the former lists temporary tables --
    # measured on a live warehouse, see ``_temp_table_rows`` -- and the two
    # expose the same column names, so one row-to-schema conversion serves both.
    #
    # ``LIKE 'pg^_temp^_%%' ESCAPE '^'`` separates temporary from permanent --
    # Redshift puts every session's temporary tables in a ``pg_temp_<N>``
    # schema -- but it is deliberately *not* what scopes this to one session,
    # and the two must not be confused: the pattern matches every session's
    # temp schema, so if the view exposed other sessions' rows this query,
    # which filters only on ``table_name``, could return another session's
    # table or two rows for one column name.
    #
    # It cannot, because ``svv_columns`` is itself session-filtered. Measured
    # 2026-09-25 with two concurrent connections as the same superuser: each
    # saw exactly one temp schema and it was its own (``pg_temp_6`` vs
    # ``pg_temp_8``); a table created only in the first was invisible to the
    # second; and with the *same* temporary table name created in both, this
    # query returned a single row, carrying the querying session's own column.
    # A superuser seeing only its own bounds every lesser-privileged user too.
    #
    # The escape character is ``^`` rather than the SQL default
    # backslash because a backslash inside a psycopg-bound statement has to
    # survive two layers of quoting; ``%%`` is a literal ``%`` to psycopg.
    _SVV_TEMP_COLUMNS_QUERY = f"""\
SELECT
{_COLUMN_FIELDS}
FROM svv_columns
WHERE table_schema LIKE 'pg^_temp^_%%' ESCAPE '^'
  AND table_name = %(table)s
ORDER BY ordinal_position ASC"""

    # Types whose ``svv_all_columns`` row carries a meaningful modifier. The
    # exclusions matter more than the inclusions: the reference's own worked
    # example shows ``numeric_precision`` populated as 32 for an ``integer``
    # and 16 for a ``smallint``, so appending the precision unconditionally
    # would build ``integer(32)``, which is not a type.
    _DECIMAL_TYPES = frozenset({"numeric", "decimal"})

    @classmethod
    def _type_string(
        cls,
        data_type: str | None,
        precision: int | None,
        scale: int | None,
    ) -> str:
        """Reassemble a type string from one ``svv_all_columns`` row."""
        base = (data_type or "").strip()
        if base.lower() in cls._DECIMAL_TYPES and precision is not None:
            return f"{base}({precision},{scale or 0})"
        return base

    @staticmethod
    def _is_nullable(flag: Any) -> bool:
        """``is_nullable`` is a ``varchar(3)``, not a boolean.

        The reference documents the values as ``yes``/``no`` and prints them as
        ``YES``/``NO`` in its own worked example, so neither case is worth
        betting on. Passing the string straight through as ``nullable=`` would
        mark every column nullable, because both spellings are truthy.

        It tests for ``no`` rather than for ``yes`` because the column has a
        documented *third* value: the ``SVV_REDSHIFT_COLUMNS`` reference gives
        the possible values as ``yes``, ``no``, and ``" "`` -- an empty string
        meaning no information, which external and datashare rows carry.
        Deciding that unknown case by asking ``== "yes"`` answers ``NOT NULL``,
        which is the unsafe direction: a column wrongly marked non-nullable
        makes ``project_and_cast_reader`` raise ``Casting field 'x' with null
        values to non-nullable`` on the first batch that carries a null, while
        a column wrongly marked nullable costs at most a cast.
        """
        if isinstance(flag, bool):
            return flag
        return str(flag).strip().lower() != "no"

    def get_schema(
        self,
        name: str,
        *,
        catalog: str | None = None,
        database: str | None = None,
    ) -> sch.Schema:
        """Read a table's schema from Redshift's own catalog.

        The inherited implementation reads ``pg_catalog`` and, purely to label
        enum columns, joins ``pg_catalog.pg_enum`` -- which Redshift does not
        have. Every ``con.table`` therefore fails with ``UndefinedTable``,
        including a plain single-schema ``con.table("t", database="s")``; this
        is not only a three-part-naming problem.

        Dropping just the enum arm would not be enough. The rest of that query
        reads ``pg_attribute``/``pg_class``/``pg_namespace`` and filters with
        ``= ANY(<array>)``, and Redshift has no array type. ``svv_all_columns``
        is the view Redshift documents for this, and the one the customer's own
        workaround used.

        The query is scoped to one database by default, not only when a
        ``catalog`` is passed. ``svv_all_columns`` is the union of
        ``SVV_REDSHIFT_COLUMNS`` -- which the reference states includes "the
        columns from datashares provided by remote clusters" -- and every
        external column, so it genuinely spans databases. ``catalog`` is
        ``None`` on every ordinary ``con.table("t")`` and
        ``con.table("t", database="s")``; only a 2-tuple or a dotted three-part
        name ever supplies one. Scoping only in that rare case left the common
        one able to match ``public.sales`` in two databases at once, and since
        ``ORDER BY ordinal_position`` has no tiebreaker across them the rows
        interleave and same-named columns silently overwrite each other. The
        compiled query still reads ``FROM "public"."sales"``, which resolves in
        the *current* database -- so the reported schema would describe a
        different table than the one queried, with no error anywhere.

        A table in another database therefore needs the three-part form. That
        is the deliberate trade: a ``TableNotFound`` naming a table you can
        re-address is recoverable, and a silently wrong schema is not.

        **The cross-database half of that argument is reasoned, not measured.**
        It rests on the ``SVV_REDSHIFT_COLUMNS`` reference text and on reading
        ``pg_get_viewdef('svv_all_columns')``; no external or datashare row was
        ever seen. The warehouse this backend was verified against carries
        none -- 0 external schemas, 0 external tables, 0 datashares, and
        ``svv_all_columns`` spanning exactly one database (measured
        2026-09-25) -- so
        the scoping predicate was never exercised against a row it would
        actually exclude. Read the rest of this docstring as measured and this
        paragraph's subject as inferred. The filter is still the safe
        direction: it costs a recoverable error if the reasoning is wrong.

        Temporary tables are **not** in this view at all -- measured on a live
        warehouse, not inferred -- so an unqualified lookup reads
        ``_temp_table_rows``, the one catalog that does list them, and it
        reads it *first*. That order is SQL's: measured on a live warehouse, a
        temporary table shadows a permanent one of the same name for
        unqualified SQL, and the compiled query names the table unqualified.
        Checking the permanent catalog first bound the permanent table's
        columns to a query that then read the temporary one -- a schema for a
        different table than the one queried, with no error. The cost is a
        second round trip for an unqualified lookup of a permanent table.

        Two behaviours are inherited rather than introduced, and neither is a
        regression: Redshift folds unquoted identifiers to lower case, so
        ``con.table("SALES")`` binds ``'SALES'`` and raises ``TableNotFound``;
        and the reference states a regular user sees only the rows it has
        access to, so a permission problem also surfaces as ``TableNotFound``.
        """
        if (
            catalog is None
            and database is None
            and (rows := self._temp_table_rows(name))
        ):
            return self._schema_from_catalog_rows(rows)

        con = self.con
        with con.cursor() as cursor, con.transaction():
            rows = cursor.execute(
                self._SVV_ALL_COLUMNS_QUERY,
                {"catalog": catalog, "schema": database, "table": name},
            ).fetchall()

        if rows:
            return self._schema_from_catalog_rows(rows)
        raise exc.TableNotFound(name)

    @classmethod
    def _schema_from_catalog_rows(cls, rows: list) -> sch.Schema:
        """Build a schema from ``_COLUMN_FIELDS``-shaped rows.

        Shared by the permanent and temporary paths so the two cannot drift on
        type mapping or on nullability -- the failure this backend has already
        had once, between its own two introspection entry points.

        ``from_tuples`` rather than a dict comprehension: it raises
        ``IntegrityError`` on a duplicate column name instead of keeping the
        last one, so a lookup that somehow matched two tables loses loudly
        rather than returning a schema short a column.

        A column the type mapper refuses -- ``SUPER`` above all -- is typed
        ``dt.NamedUnknown``, keeping its catalog spelling and nullability,
        rather than failing the whole table, and ``_raise_on_unmappable_columns``
        refuses any read or create that touches it. Dropping it instead would
        bind a schema silently short a column.
        """
        return sch.Schema.from_tuples(
            [
                (
                    column_name,
                    cls._column_dtype(
                        cls._type_string(data_type, precision, scale),
                        nullable=cls._is_nullable(is_nullable),
                    ),
                )
                for (
                    column_name,
                    data_type,
                    is_nullable,
                    precision,
                    scale,
                ) in rows
            ]
        )

    @classmethod
    def _column_dtype(cls, type_string: str, nullable: bool) -> dt.DataType:
        """One column's dtype, or ``dt.NamedUnknown`` if it has none.

        Built here rather than taken from the mapper's own ``unknown``
        fallback, which drops ``nullable=`` and the type's name.
        """
        try:
            return cls.compiler.type_mapper.from_string(type_string, nullable=nullable)
        except exc.UnsupportedBackendType:
            logger.debug(
                "binding an unmappable column as unknown", type_string=type_string
            )
            return dt.NamedUnknown(raw_type=type_string, nullable=nullable)

    def _raise_on_unmappable_columns(
        self, columns: dict[str, dt.DataType], action: str
    ) -> None:
        """Refuse an operation that touches an unmappable column.

        This is where an unmappable column fails, since binding no longer
        does. It must be explicit: nothing downstream refuses in a way that
        names the column. ``Schema.to_pyarrow`` raises ``NotImplementedError``
        naming only the type, and ``Schema.to_sqlglot`` fails with a bare
        ``KeyError`` naming a Python class.
        """
        mapper = self.compiler.type_mapper
        unmappable = {
            name: part
            for name, dtype in columns.items()
            if (part := mapper.unmappable_part(dtype)) is not None
        }
        if unmappable:
            described = ", ".join(
                f"{name!r} ({self.name} type {part.raw_type!r})"
                if isinstance(part, dt.NamedUnknown)
                else repr(name)
                for name, part in unmappable.items()
            )
            raise exc.UnmappableColumnError(
                f"cannot {action} column(s) {described}: the type has no xorq "
                f"equivalent, so the column is bound as unknown. Select the "
                f"other columns -- before into_backend, which moves every "
                f"column it is given -- or convert it to text server side "
                f"through con.sql (for SUPER, JSON_SERIALIZE).",
                columns=tuple(unmappable),
            )

    def _derived_unmappable_values(self, expr: ir.Expr) -> dict[str, dt.DataType]:
        """The unmappable columns ``expr`` casts or computes from, returned or
        not.

        Two kinds pass the returned-schema check and still have to be refused.

        A cast returns a mappable type, but not the data: measured on a live
        warehouse, ``CAST`` of a ``SUPER`` object or array to ``VARCHAR`` is
        ``NULL``, while a string or number scalar casts to its text. So a
        column of objects would read back as all ``NULL``, silently.
        ``JSON_SERIALIZE``, which the refusal names instead, returned the
        object's text.

        A value built *from* such a column -- a struct holding it, say, later
        unpacked and dropped -- is compiled with its type, which this backend
        cannot spell, so it fails in the compiler with a bare ``KeyError``
        naming no column.

        A column reference is neither, and neither is an alias of one, so a
        filter on the column is not refused.

        Each is reported by the unmappable columns it is built from, which is
        what a caller can act on, and by its own name only if it has none.
        """
        unmappable = self.compiler.type_mapper.unmappable_part
        derived = (
            op.arg if isinstance(op, (ops.Cast, ops.TryCast)) else op
            for op in expr.op().find(ops.Value)
            if isinstance(op, (ops.Cast, ops.TryCast))
            or not isinstance(op, (ops.Field, ops.Alias))
        )
        found = {}
        for value in derived:
            if unmappable(value.dtype) is None:
                continue
            # A bound column that cannot be mapped is a top-level unknown; a
            # field whose type merely *contains* one, like the struct above
            # once projected, is derived and would name the wrong thing.
            fields = {
                field.name: field.dtype
                for field in value.find(ops.Field)
                if isinstance(field.dtype, dt.Unknown)
            }
            found.update(fields or {value.name: value.dtype})
        return found

    def refuse_before_execute(self, expr: ir.Expr) -> None:
        """Refuse a read before any work is done for it if it would return an
        unmappable column, or cast or compute one.

        Called by xorq's execution entry points before their transform passes
        run, and by ``_run_pre_execute_hooks`` for the backend's own entry
        points. A column referenced but not returned, as in a filter on it,
        is not refused: that is evaluated server side on Redshift's own type,
        and the data returned is still correctly typed.
        """
        self._raise_on_unmappable_columns(
            dict(expr.as_table().schema().items()), "read"
        )
        self._raise_on_unmappable_columns(
            self._derived_unmappable_values(expr), "cast or compute"
        )

    def _run_pre_execute_hooks(self, expr: ir.Expr) -> None:
        """Refuse a read before it is issued if it returns an unmappable column.

        Every inherited read entry point -- ``execute``, ``to_pyarrow``,
        ``to_pyarrow_batches``, and so a ``RemoteTable`` or cache read drawing
        on this backend -- calls this before compiling, so one check covers
        them all. ``compile`` does not, so an expression carrying such a column
        still compiles.
        """
        self.refuse_before_execute(expr)
        super()._run_pre_execute_hooks(expr)

    def create_table(
        self,
        name: str,
        /,
        obj: Any = None,
        *,
        schema: sch.SchemaLike | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Refuse a table that would carry an unmappable column, before any DDL.

        Refused even from an expression on this backend, which Redshift could
        copy server side: the inherited implementation spells the new table's
        columns out through ``Schema.to_sqlglot``, and an unmappable column has
        no type to spell.
        """
        if schema is not None:
            self._raise_on_unmappable_columns(
                dict(sch.schema(schema).items()), "create"
            )
        if isinstance(obj, ir.Expr):
            self._raise_on_unmappable_columns(
                dict(obj.as_table().schema().items()), "create"
            )
        return super().create_table(name, obj, schema=schema, **kwargs)

    def insert(
        self,
        table_name: str,
        obj: Any,
        schema: str | None = None,
        database: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Refuse an insert that would carry an unmappable column, before the
        target is touched.

        The inherited implementation truncates first for ``overwrite=True``
        and only then runs ``_run_pre_execute_hooks``, where the read check
        lives -- so the refusal left the target empty, and Redshift's
        ``TRUNCATE`` commits on its own. The check therefore runs here, ahead
        of the truncate.

        An ``INSERT ... SELECT`` of such a column is a server-side copy that
        Redshift could run. It is refused anyway, for the reason
        ``create_table`` is: this backend cannot tell a server-side copy from a
        read of the same expression, and a column whose type it cannot name is
        one it cannot vouch for at either end.
        """
        if isinstance(obj, ir.Expr):
            self._raise_on_unmappable_columns(
                dict(obj.as_table().schema().items()), "insert"
            )
        return super().insert(
            table_name,
            obj,
            schema=schema,
            database=database,
            overwrite=overwrite,
        )

    def _temp_table_rows(self, name: str) -> list:
        """The catalog rows of a *temporary* table, which ``svv_all_columns``
        omits, or none if the session has no temporary table of that name.

        This exists because this backend's own ingest path depends on it.
        ``read_parquet``/``read_csv``/``read_record_batches`` with no
        ``table_name`` force ``temporary=True``, generate a name, create a
        ``TEMPORARY`` table and end in ``self.table(table_name)``, which
        forwards ``catalog=None, database=None``. The inherited postgres
        implementation covered this by appending ``_session_temp_db`` to its
        schema list; dropping that fold made every temporary ingest raise
        ``TableNotFound`` *after* the data had been written.

        Three facts, all measured on a live warehouse (2026-09-24), decide the
        shape of this:

        1. ``svv_all_columns`` does not list temporary tables -- 0 rows for one
           that exists. Restoring the inherited ``_session_temp_db`` fold could
           therefore never have worked: no ``schema_name`` predicate finds a row
           the view does not carry.
        2. ``svv_columns`` **does** list them, in a ``pg_temp_<N>`` schema, with
           the same column names and with ``is_nullable`` intact. So this path
           keeps ``NOT NULL``, and does not have to widen every column the way a
           ``cursor.description`` probe would.
        3. ``pg_my_temp_schema()`` does not exist on Redshift, so the schema
           cannot be resolved first and must be matched by pattern.

        A ``SELECT * ... LIMIT 0`` probe was the previous implementation and is
        wrong here, not merely slower: it resolves through ``search_path``,
        which on Redshift defaults to ``"$user", public``. An unqualified
        ``con.table("t")`` that missed the catalog would find a *permanent*
        ``alice.t`` through the probe and report it with every column widened to
        nullable -- a different table than the caller named, silently. Scoping
        to ``pg_temp_%`` is what makes the lookup mean "temporary table"
        rather than "anything the session can see".
        """
        con = self.con
        with con.cursor() as cursor, con.transaction():
            return cursor.execute(
                self._SVV_TEMP_COLUMNS_QUERY, {"table": name}
            ).fetchall()

    # Redshift's own types carry OIDs psycopg's registry does not know, so a
    # result description names them only by number. Read from ``pg_type`` on
    # a live warehouse (2026-09-28), and then from live result descriptions,
    # which differ in two places: a ``GEOMETRY`` value is described as 3999,
    # not ``pg_type``'s 3000, and the two interval column types are 1188 and
    # 1190, which arrive as text (``'1 mon'``) for want of a loader. Those two
    # are named as ``svv_columns`` spells them, so both paths bind one name.
    _REDSHIFT_TYPE_OIDS = {
        1188: "intervaly2m",
        1190: "intervald2s",
        2935: "hllsketch",
        3000: "geometry",
        3001: "geography",
        3999: "geometry",
        4000: "super",
        VARBYTE_OID: "varbyte",
    }

    @classmethod
    def _column_dtype_from_description(cls, column: Any) -> dt.DataType:
        """The dtype of one ``psycopg.Column`` of a result description.

        ``type_code`` is a PostgreSQL type OID. Redshift is a PostgreSQL 8.0
        derivative and reports the standard OIDs, which psycopg's builtin
        registry resolves without a round trip.

        Redshift's own types are named from ``_REDSHIFT_TYPE_OIDS`` and then
        mapped exactly as the catalog path maps their names, so ``VARBYTE`` is
        ``dt.Binary`` on both paths and ``SUPER`` binds as ``dt.NamedUnknown``
        on both. Any other OID the registry does not know binds as
        ``dt.NamedUnknown`` named by its number, and so does an OID the registry
        *can* name but the type mapper cannot use; either is refused where it is
        used. Nullability is not in a result description, so every column is
        nullable.

        The type string itself comes from psycopg's own ``Column.type_display``
        rather than from ``info.name``. They differ in two ways that matter:
        ``type_display`` applies the type modifier, so a ``numeric(8,2)``
        arrives parameterised without this method restating the decimal rule
        that ``_type_string`` already owns; and it renders an array as
        ``date[]``, where ``info.name`` on an array OID gives the *element*
        name and silently turns every array column into its element type.
        """
        if (name := cls._REDSHIFT_TYPE_OIDS.get(column.type_code)) is not None:
            return cls._column_dtype(name, nullable=True)
        if psycopg.postgres.types.get(column.type_code) is None:
            return dt.NamedUnknown(
                raw_type=f"type OID {column.type_code}", nullable=True
            )
        return cls._column_dtype(column.type_display, nullable=True)

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        """Infer a query's schema from the result description, issuing no DDL.

        The inherited implementation wraps the query in a
        ``CREATE TEMPORARY VIEW`` and introspects that. Redshift has temporary
        *tables* but not temporary *views*, so ``con.sql`` dies with a syntax
        error at ``VIEW``.

        The obvious repair -- swap the temporary view for a temporary table
        created ``LIMIT 0`` and dropped in a ``finally`` -- was rejected rather
        than merely passed over. It would have to be introspected back through
        ``get_schema`` above, which reaches a temporary table only through its
        ``svv_columns`` fallback, so the probe would round-trip a catalog it
        does not need. Reading ``cursor.description`` consults no catalog at
        all. It also needs no create privilege and leaves nothing behind, so
        there is no cleanup path to get wrong.

        ``get_schema`` does *not* borrow this method for a table its catalog
        query missed: a probe resolves through ``search_path`` and would
        report a permanent table of the same name. See ``_temp_table_rows``.

        The query is *wrapped* in a derived table rather than suffixed with
        ``LIMIT 0``. Not because appending would be a syntax error -- sqlglot's
        ``.limit(0)`` replaces an existing ``LIMIT`` and binds to a whole
        ``UNION`` rather than to its last branch, so appending through the AST
        is safe, and other backends do exactly that. The wrap earns its place
        by giving every probe the same shape regardless of what was passed,
        including the statement forms that are not ``Query`` at all. The bound
        itself is not cosmetic: psycopg buffers the whole result client side,
        so an unbounded probe would read the table it selects from.

        Anything that is not a ``Query`` is refused here rather than wrapped.
        ``VALUES (1, 2)`` and ``TABLE t`` parse to ``sge.Values`` and
        ``sge.Alias``, and an earlier version wrapped them by hand so that
        ``.subquery()`` would not raise ``AttributeError``. That was effort
        spent on statements Redshift cannot run: measured on a live warehouse,
        ``VALUES (1, 2)``, ``SELECT * FROM (VALUES (1, 2)) AS p LIMIT 0``,
        ``TABLE t`` and its wrapped form are all syntax errors there. Refusing
        locally turns a round trip that was going to fail into a clear message,
        and also covers the empty string, which parses to ``[None]``.

        Every column comes back nullable, because a result description carries
        no nullability -- there is nothing to read, so this is a widening
        rather than a guess. It is why ``con.sql`` reports all-nullable where
        ``con.table`` reports ``NOT NULL`` for the same column; the cost is one
        Python-side cast per batch in ``project_and_cast_reader``, and the only
        way to close it would be to invent nullability the server did not send.
        """
        # ``sg.parse`` yields one element per statement, but a trailing
        # separator or comment contributes an element that is not a statement:
        # ``SELECT 1; -- note`` parses to ``[Select, Semicolon]`` and
        # ``SELECT 1;;`` to ``[Select, None]``. Counting those as statements
        # rejected valid single-statement SQL, which is why they are dropped
        # before the count rather than after it.
        statements = [
            stmt
            for stmt in sg.parse(query, read=self.dialect)
            if stmt is not None and not isinstance(stmt, _NOT_A_STATEMENT)
        ]
        if len(statements) != 1:
            # ``ops.SQLQueryResult`` keeps the whole string, and compiling it
            # takes ``parse_one``'s first statement and drops the rest, so a
            # multi-statement ``con.sql`` would silently run less than it said.
            raise exc.XorqError(
                f"expected a single statement to introspect, got {len(statements)}"
            )
        (parsed,) = statements

        if not isinstance(parsed, sge.Query):
            raise exc.XorqError(
                f"cannot introspect {type(parsed).__name__} statements on "
                f"{self.name}; only queries have a result description to read"
            )

        probe = (
            sg.select(sge.Star())
            .from_(parsed.subquery(PROBE_ALIAS))
            .limit(0)
            .sql(self.dialect)
        )

        con = self.con
        with con.cursor() as cursor, con.transaction():
            description = list(cursor.execute(probe).description)

        # ``from_tuples`` rather than a dict comprehension: a result set may
        # legitimately repeat a column name (``SELECT t1.id, t2.id``, or
        # ``SELECT 1, 2``, whose columns are both ``?column?``), and keying a
        # dict on the name would hand back a schema with fewer columns than the
        # cursor returns -- which ``_fetch_from_cursor`` then misaligns instead
        # of rejecting. The inherited temporary-view path failed loudly here
        # (``column "id" specified more than once``); this keeps that.
        return sch.Schema.from_tuples(
            [
                (column.name, self._column_dtype_from_description(column))
                for column in description
            ]
        )

    def _adbc_unavailable_reason(self) -> str | None:
        """Why the ADBC accelerator cannot be used, or ``None`` if it can.

        Serves the read path alone -- ``_open_adbc_conn_or_none`` is its only
        caller -- and is the reason that path needs no catch-all. Ingest used
        to share it and must not: see ``read_record_batches``. Availability is
        decided from local facts *before* anything is dialled, so every
        exception raised by the subsequent connect is a real failure and
        propagates -- which is what separates "no driver installed" from
        "these credentials were rejected".
        Under a rotating IAM credential that distinction is the difference
        between a quiet fallback and silence about an expired password.

        Both checks mirror what ``PgADBC`` would actually do, rather than
        approximating it: it imports ``adbc_driver_postgresql`` and it reads
        ``_con_kwargs["password"]`` to interpolate into a URI. A ``password``
        that is absent *or* ``None`` is disqualifying, and the ``None`` case is
        the one worth stating: it does not raise, it formats into the URI as
        the literal string ``"None"`` and fails later as an auth error against
        a password nobody set.

        This method is also the seam the accelerator work extends, but it is
        not the whole of it: swapping accelerators changes a clause here, the
        extras, and ``PgADBC``, which ``_open_adbc_conn_or_none`` below dials
        and which hardcodes ``adbc_driver_postgresql``.

        ADR-2332 settled the open question this docstring used to carry:
        measured against a live endpoint, ``adbc_driver_postgresql`` connects
        to Redshift and passed every read that was exercised (the ADR lists
        them; none carried an auto-generated alias, which is a known failure
        of its own), and the feared ``pg_catalog``
        failures are in xorq's own psycopg path instead. A "no
        reason" answer still means only *installed and credentialed* -- and for
        ingest it is the wrong question entirely, since neither ADBC driver can
        ingest into Redshift. That is why ingest no longer asks it.
        """
        # Uncached: the tests simulate an absent driver by patching
        # ``find_spec``, and a cached answer would outlive the patch.
        if importlib.util.find_spec("adbc_driver_postgresql") is None:
            return "adbc_driver_postgresql is not installed"
        if self._con_kwargs.get("password") is None:
            return "no password in _con_kwargs for PgADBC to build a URI from"
        return None

    def _open_adbc_conn_or_none(self):
        """Open an ADBC connection for the Arrow read path, or ``None``.

        Overrides the postgres implementation to drop its ``except Exception``.
        That catch-all is right for postgres, where an unbuildable URI is an
        ordinary consequence of a ``.pgpass`` connection, but here it would
        swallow a rejected temporary credential and quietly downgrade to
        psycopg -- reporting nothing while the IAM path is broken.

        The URI, including the caller's libpq settings and the search path,
        is ``PgADBC``'s; see ``PgADBC.settings``.
        """
        if (reason := self._adbc_unavailable_reason()) is not None:
            logger.debug(
                "ADBC accelerator unavailable; using the psycopg baseline",
                backend=self.name,
                reason=reason,
            )
            return None

        # Below the probe: ``postgres_utils`` imports
        # ``adbc_driver_postgresql`` at module scope, so an import above it
        # raises in exactly the case the probe exists to detect.
        from xorq.common.utils.postgres_utils import PgADBC  # noqa: PLC0415

        return PgADBC(self).get_conn()

    def read_record_batches(
        self,
        record_batches: pa.RecordBatchReader,
        table_name: str | None = None,
        password: str | None = None,
        temporary: bool = False,
        mode: str = "create",
        **kwargs: Any,
    ) -> ir.Table:
        """Ingest Arrow record batches over psycopg. There is no ADBC branch.

        The postgres implementation is unconditional ADBC, and inheriting it
        made this backend claim a psycopg baseline it did not have. This one
        is unconditional psycopg, which is not a fallback but the only ingest
        Redshift accepts: measured against a live endpoint, *neither* ADBC
        driver can ingest here, because both ingest by ``COPY`` and Redshift's
        ``COPY`` reads from S3 only. For ``adbc_driver_postgresql`` that is a
        parse failure (``COPY ... FROM STDIN``, SQLSTATE 42601), not a missing
        setting, so no configuration reaches it and the driver has no
        ``INSERT`` fallback.

        **Deliberately not dispatched on ``_adbc_unavailable_reason()``.** That
        predicate answers "is the accelerator installed and credentialed",
        which is the right question for ``to_pyarrow_batches`` and the wrong
        one here: it returns ``None`` for every connection given a password, so
        dispatching on it selected the branch that cannot run and left the one
        that works as dead code. The two paths' correct answers are inversely
        correlated, so they must not share a predicate -- and after this method
        stopped calling it, they no longer can.

        No ingest-side predicate replaces it, because nothing would ever flip
        one: ``adbc_driver_postgresql`` will not ingest here in any release,
        and the Columnar driver is rejected by ADR-2332 on packaging grounds.
        The genuine future path is ``COPY``-from-S3, which is psycopg plus a
        staging upload and would branch on whether a bucket is configured --
        inside this method, never on driver availability. That work is out of
        scope; it needs no seam held open here.

        ``password`` is unused and kept because it is the inherited signature:
        ``read_csv`` and ``read_parquet`` both forward it down this call.
        ``kwargs`` are likewise accepted and dropped -- those two callers
        forward their *own* reader kwargs here, so rejecting unknown ones would
        break them.

        The return value is ``self.table(table_name)``, bound over the same
        connection the ingest wrote through. A ``temporary=True`` table is
        absent from ``svv_all_columns``, so the unqualified bind finds it
        through ``get_schema``'s ``svv_columns`` lookup, which is scoped to the
        session's temporary schemas.
        """
        if table_name is None:
            raise ValueError("table_name is required")
        if mode not in INGEST_MODES:
            raise ValueError(f"mode must be one of {INGEST_MODES}, got {mode!r}")
        if temporary and mode in APPEND_ONLY_MODES:
            # ``append`` emits no ``CREATE`` for ``TEMPORARY`` to mark, and
            # ``create_append`` would render ``CREATE TEMPORARY TABLE IF NOT
            # EXISTS``, which resolves against ``pg_temp`` and shadows a
            # permanent table.
            raise ValueError(
                f"temporary=True is not supported with mode={mode!r}: "
                f"{APPEND_ONLY_MODES} append to a table this call does not "
                "create, so there is nothing for temporary to apply to"
            )

        # Unguarded, a null column renders as the column type ``NULL``, which
        # no server accepts.
        null_columns = [
            name
            for name, dtype in sch.Schema.from_pyarrow(record_batches.schema).items()
            if dtype.is_null()
        ]
        if null_columns:
            raise exc.XorqTypeError(
                f"{self.name} cannot yet reliably handle `null` typed columns; "
                f"got null typed columns: {null_columns}"
            )

        return self._read_record_batches_psycopg(
            record_batches,
            table_name,
            temporary=temporary,
            mode=mode,
        )

    def _read_record_batches_psycopg(
        self,
        record_batches: pa.RecordBatchReader,
        table_name: str,
        *,
        temporary: bool = False,
        mode: str = "create",
    ) -> ir.Table:
        """``CREATE TABLE`` + parameterised ``INSERT`` over the live psycopg
        connection.

        ``INSERT`` rather than ``COPY`` because Redshift has no
        ``COPY ... FROM STDIN``: its ``COPY`` reads from S3, which would make
        the baseline require a bucket, an IAM role to assume and a staging
        lifecycle. That is the deferred ``redshift.ingest.bucket`` work, and
        keeping it out is why ``COPY``-from-S3 is out of scope here.

        ``TEMPORARY`` is applied to the ``CREATE`` directly, where the ADBC
        path creates a permanent table and converts it afterwards via
        ``make_table_temporary``. That rename-and-copy dance is not overhead
        ADBC failed to avoid -- it opens its own connection, so a temp table
        created there would be invisible to this one. Sharing the psycopg
        connection is what makes the direct form correct here.

        The reader's own batch size bounds each ``executemany``, so a caller
        that wants smaller transactions controls it upstream; the whole ingest
        is one transaction, so a failure part-way leaves no half-filled table.

        A ``Table`` is normalised to a reader rather than iterated: iterating a
        ``pa.Table`` yields *columns*, so it would fail here on a missing
        ``num_rows`` for an input ``adbc_ingest`` accepts. Callers were written
        against that signature and still pass tables, so this path has to take
        them too.
        """
        if isinstance(record_batches, pa.Table):
            # Unbounded, a one-chunk table becomes one batch holding every row,
            # materialised again as the tuples ``executemany`` binds.
            record_batches = record_batches.to_reader(max_chunksize=INGEST_CHUNKSIZE)

        schema = sch.Schema.from_pyarrow(record_batches.schema)
        quoted = self.compiler.quoted
        table = sg.table(table_name, quoted=quoted)

        statements = []
        if mode == "replace" and temporary:
            # An unqualified ``DROP`` resolves through ``search_path``, so with
            # no temporary table of this name in the session yet it would land
            # on a PERMANENT one and destroy it. Drop only the session's own
            # temporary table, named by the schema it was found in; with none,
            # there is nothing to replace. Measured on a live warehouse with a
            # temporary table shadowing a permanent one of the same name: this
            # qualified ``DROP`` is accepted and removes only the temporary one.
            if temp_schema := session_temp_schema_of(self.con, table_name):
                statements.append(
                    sge.Drop(
                        this=sg.table(table_name, db=temp_schema, quoted=quoted),
                        kind="TABLE",
                    )
                )
        elif mode == "replace":
            statements.append(sge.Drop(this=table, kind="TABLE", exists=True))
        if mode != "append":
            statements.append(
                sge.Create(
                    kind="TABLE",
                    this=sge.Schema(
                        this=sg.to_identifier(table_name, quoted=quoted),
                        expressions=schema.to_sqlglot(self.dialect),
                    ),
                    exists=mode == "create_append",
                    properties=sge.Properties(
                        expressions=[sge.TemporaryProperty()] if temporary else []
                    ),
                )
            )

        insert = self._build_insert_template(
            table_name, schema=schema, columns=True, placeholder="%s"
        )

        con = self.con
        with con.cursor() as cursor, con.transaction():
            for statement in statements:
                cursor.execute(statement.sql(self.dialect))
            for batch in record_batches:
                if batch.num_rows:
                    cursor.executemany(insert, zip(*batch.to_pydict().values()))
        return self.table(table_name)
