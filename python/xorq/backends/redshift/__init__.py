from __future__ import annotations

import importlib.util
import re
import urllib.parse
from typing import Any

import pyarrow as pa
import sqlglot as sg
import sqlglot.expressions as sge

import xorq.common.exceptions as exc
import xorq.vendor.ibis.expr.schema as sch
from xorq.backends.postgres import Backend as PostgresBackend
from xorq.backends.redshift.compiler import compiler
from xorq.common.utils.logging_utils import get_logger
from xorq.vendor.ibis.expr import types as ir


logger = get_logger(__name__)


__all__ = [
    "Backend",
]

# Redshift listens on 5439; the postgres backend defaults to 5432.
DEFAULT_PORT = 5439

# The modes ``adbc_ingest`` accepts, restated so the psycopg path accepts
# exactly the same set: whichever branch runs must be an implementation detail,
# and it stops being one the moment the two disagree about what ``mode`` means.
INGEST_MODES = ("create", "append", "replace", "create_append")

# Modes that append to a table this call did not create, so ``temporary`` has
# nothing to apply to.
APPEND_ONLY_MODES = ("append", "create_append")

# Rows per ``executemany`` for a ``pa.Table``, which carries no batch size.
INGEST_CHUNKSIZE = 10_000

# Alias for the derived table that ``_get_schema_using_query`` probes through.
# Fixed rather than generated: it names a subquery, which is scoped to the
# statement and cannot collide with anything in the catalog, and a deterministic
# alias makes the emitted SQL assertable.
PROBE_ALIAS = "redshift_probe"


def _search_path_option(schema: str) -> str:
    r"""``schema`` as a percent-encoded libpq ``options`` value setting
    ``search_path``.

    libpq splits ``options`` on whitespace unless it is backslash-escaped,
    and reads ``\\`` as one backslash, so both are escaped before encoding.
    The value is otherwise passed as ``_post_connect`` passes it to
    ``set_config``: a ``search_path`` string, so a comma list keeps its
    meaning.
    """
    escaped = re.sub(r"([\\\s])", r"\\\1", schema)
    return urllib.parse.quote(f"-csearch_path={escaped}", safe="")


class Backend(PostgresBackend):
    """Redshift Serverless, over the PostgreSQL wire protocol.

    Subclasses the *xorq* postgres backend rather than the vendored ibis one:
    the ADBC/psycopg ``to_pyarrow_batches`` and ``read_record_batches`` live on
    the xorq subclass, and the vendored base has no ``read_record_batches`` at
    all.
    """

    name = "redshift"
    compiler = compiler

    # ``do_connect`` defaults ``client_encoding`` below the caller's kwargs so
    # it never reaches ``_con_kwargs``, the profile or the build hash. The live
    # DSN reports it regardless, so ``clone`` has to be told to drop it or the
    # clone hashes differently from its source over a setting nobody passed.
    _clone_drop_dsn_params = ("client_encoding",)

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

        ``prepare_threshold`` is defaulted to ``None`` for the same kind of
        reason: Redshift has no ``DEALLOCATE ALL``, which psycopg sends to
        clear its server-side prepared statements whenever a transaction
        rolls back. There it fails with a syntax error, so a rolled-back
        ``drop_table`` leaves the table behind. With the threshold ``None``
        psycopg never prepares a statement, so it never has one to
        deallocate.

        Defaulting both here rather than in the caller keeps them out of
        ``_con_kwargs``, which is captured from the caller's arguments, so
        they never reach the profile or the build hash; a caller's explicit
        value still wins.
        """
        kwargs.setdefault("client_encoding", "utf8")
        kwargs.setdefault("prepare_threshold", None)
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
    # measured on a live warehouse, see ``_temp_table_schema`` -- and the two
    # expose the same column names, so one row-to-schema conversion serves both.
    #
    # ``LIKE 'pg^_temp^_%%' ESCAPE '^'`` is the whole scoping, and it is exact:
    # Redshift puts every session's temporary tables in a ``pg_temp_<N>``
    # schema. The escape character is ``^`` rather than the SQL default
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

        Temporary tables are **not** in this view at all -- measured on a live
        warehouse, not inferred -- so an unqualified lookup that finds nothing
        falls through to ``_temp_table_schema``, which reads the one catalog
        that does list them.

        Two behaviours are inherited rather than introduced, and neither is a
        regression: Redshift folds unquoted identifiers to lower case, so
        ``con.table("SALES")`` binds ``'SALES'`` and raises ``TableNotFound``;
        and the reference states a regular user sees only the rows it has
        access to, so a permission problem also surfaces as ``TableNotFound``.
        """
        con = self.con
        with con.cursor() as cursor, con.transaction():
            rows = cursor.execute(
                self._SVV_ALL_COLUMNS_QUERY,
                {"catalog": catalog, "schema": database, "table": name},
            ).fetchall()

        if rows:
            return self._schema_from_catalog_rows(rows)
        if catalog is None and database is None:
            return self._temp_table_schema(name)
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
        """
        type_mapper = cls.compiler.type_mapper
        return sch.Schema.from_tuples(
            [
                (
                    column_name,
                    type_mapper.from_string(
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

    def _temp_table_schema(self, name: str) -> sch.Schema:
        """The schema of a *temporary* table, which ``svv_all_columns`` omits.

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
        to ``pg_temp_%`` is what makes the fallback mean "temporary table"
        rather than "anything the session can see".
        """
        con = self.con
        with con.cursor() as cursor, con.transaction():
            rows = cursor.execute(
                self._SVV_TEMP_COLUMNS_QUERY, {"table": name}
            ).fetchall()

        if not rows:
            raise exc.TableNotFound(name)
        return self._schema_from_catalog_rows(rows)

    @classmethod
    def _type_string_from_column(cls, column: Any) -> str:
        """The type name for one ``psycopg.Column`` of a result description.

        ``type_code`` is a PostgreSQL type OID. Redshift is a PostgreSQL 8.0
        derivative and reports the standard OIDs, which psycopg's builtin
        registry resolves without a round trip.

        An OID the registry does not know -- Redshift's own ``SUPER``,
        ``VARBYTE`` and ``GEOMETRY`` are the expected cases -- raises rather
        than degrading to ``unknown``. A schema that is quietly wrong is the
        failure mode this backend's tests exist to prevent, and the OID goes in
        the message so a live session can map it. An OID the registry *can*
        name but the type mapper cannot use raises too, one layer down, in
        ``RedshiftType.from_string``.

        The type string itself comes from psycopg's own ``Column.type_display``
        rather than from ``info.name``. They differ in two ways that matter:
        ``type_display`` applies the type modifier, so a ``numeric(8,2)``
        arrives parameterised without this method restating the decimal rule
        that ``_type_string`` already owns; and it renders an array as
        ``date[]``, where ``info.name`` on an array OID gives the *element*
        name and silently turns every array column into its element type.

        ``psycopg`` is imported here rather than at module level because it is
        an optional extra: this module is imported when the backend entry point
        is resolved, and a module-level import would make that fail wherever
        the postgres extra is not installed. ``test_core_module_imports_are_
        declared`` enforces exactly this.
        """
        import psycopg  # noqa: PLC0415

        if psycopg.postgres.types.get(column.type_code) is None:
            raise exc.UnsupportedBackendType(
                f"{cls.name} returned column {column.name!r} with type OID "
                f"{column.type_code}, which psycopg cannot name; it is most "
                f"likely a Redshift-specific type (SUPER, VARBYTE, GEOMETRY)"
            )
        return column.type_display

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        """Infer a query's schema from the result description, issuing no DDL.

        The inherited implementation wraps the query in a
        ``CREATE TEMPORARY VIEW`` and introspects that. Redshift has temporary
        *tables* but not temporary *views*, so ``con.sql`` dies with a syntax
        error at ``VIEW``.

        The obvious repair -- swap the temporary view for a temporary table
        created ``LIMIT 0`` and dropped in a ``finally`` -- was rejected rather
        than merely passed over. It would have to be introspected back through
        ``get_schema`` above, and ``svv_all_columns`` is not documented to list
        temporary tables; that repair would trade a syntax error for a
        ``TableNotFound`` while looking like a fix in every offline test.
        Reading ``cursor.description`` consults no catalog at all, so it does
        not depend on that unsettled question. It also needs no create
        privilege and leaves nothing behind, so there is no cleanup path to get
        wrong.

        That property is why ``get_schema`` above borrows this method for a
        table its catalog query did not find: the same "no catalog at all" that
        makes this the right probe for a *query* makes it the only reliable way
        to reach a *temporary table*. See ``_temp_table_schema``.

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
            if stmt is not None and not isinstance(stmt, sge.Semicolon)
        ]
        if len(statements) != 1:
            # ``parse_one`` would silently probe the first statement while
            # ``ops.SQLQueryResult`` stores and executes the whole string.
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

        type_mapper = self.compiler.type_mapper
        # ``from_tuples`` rather than a dict comprehension: a result set may
        # legitimately repeat a column name (``SELECT t1.id, t2.id``, or
        # ``SELECT 1, 2``, whose columns are both ``?column?``), and keying a
        # dict on the name would hand back a schema with fewer columns than the
        # cursor returns -- which ``_fetch_from_cursor`` then misaligns instead
        # of rejecting. The inherited temporary-view path failed loudly here
        # (``column "id" specified more than once``); this keeps that.
        return sch.Schema.from_tuples(
            [
                (
                    column.name,
                    type_mapper.from_string(
                        self._type_string_from_column(column), nullable=True
                    ),
                )
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
        extras, and the connection factory below (``PgADBC``, which hardcodes
        ``adbc_driver_postgresql``).

        ADR-2332 settled the open question this docstring used to carry:
        measured against a live endpoint, ``adbc_driver_postgresql`` connects
        to Redshift and passed every read that was exercised (the ADR lists
        them, and one known alias failure), and the feared ``pg_catalog``
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

        It also carries ``schema`` onto the ADBC connection. psycopg gets it
        from ``_post_connect``'s ``set_config('search_path', ...)``; ADBC is a
        second connection, and ``PgADBC``'s URI names no schema, so it ran
        with the server default ``'$user, public'``. The compiler emits
        unqualified table names, so every table-bound read raised "relation
        does not exist" there and was silently re-run on psycopg by the
        inherited execute-stage catch: correct rows, and an accelerator that
        never served a table-bound read. The search path therefore goes in the
        URI's libpq ``options``, taken from the same ``_con_kwargs["schema"]``
        ``_post_connect`` reads, so the two connections agree.
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
        from xorq.common.utils.postgres_utils import (  # noqa: PLC0415
            PgADBC,
            adbc_driver_postgresql,
        )

        uri = PgADBC(self).get_uri()
        if schema := self._con_kwargs.get("schema"):
            uri = f"{uri}?options={_search_path_option(schema)}"
        return adbc_driver_postgresql.dbapi.connect(uri)

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
        one here: it returns ``None`` on every credentialed install, so
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
        scope and is tracked separately; it needs no seam held open here.

        ``password`` is unused and kept because it is the inherited signature:
        ``read_csv`` and ``read_parquet`` both forward it down this call.
        ``kwargs`` are likewise accepted and dropped -- those two callers
        forward their *own* reader kwargs here, so rejecting unknown ones would
        break them.
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
        if temporary and mode == "replace":
            # ``replace`` emits an unqualified ``DROP TABLE IF EXISTS`` before
            # the ``CREATE``, and it resolves through ``search_path``: with no
            # temporary table of that name in the session yet, it drops the
            # PERMANENT one, then replaces it with a table that disappears at
            # disconnect. The guard above refuses shadowing, which ends with
            # the session; this refuses destruction, which does not.
            raise ValueError(
                "temporary=True is not supported with mode='replace': the "
                "DROP it emits is unqualified, so it would resolve to a "
                "permanent table of the same name and destroy it"
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
        keeping it out is exactly why ``COPY``-from-S3 is off the v1 list.

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
        if mode == "replace":
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
