from __future__ import annotations

import importlib.util
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
    "connect",
]

# Redshift listens on 5439; the postgres backend defaults to 5432.
DEFAULT_PORT = 5439

# The modes ``adbc_ingest`` accepts, restated so the psycopg path accepts
# exactly the same set: whichever branch runs must be an implementation detail,
# and it stops being one the moment the two disagree about what ``mode`` means.
INGEST_MODES = ("create", "append", "replace", "create_append")

# Alias for the derived table that ``_get_schema_using_query`` probes through.
# Fixed rather than generated: it names a subquery, which is scoped to the
# statement and cannot collide with anything in the catalog, and a deterministic
# alias makes the emitted SQL assertable.
PROBE_ALIAS = "redshift_probe"


class Backend(PostgresBackend):
    """Redshift Serverless, over the PostgreSQL wire protocol.

    Subclasses the *xorq* postgres backend rather than the vendored ibis one:
    the ADBC/psycopg ``to_pyarrow_batches`` and ``read_record_batches`` live on
    the xorq subclass, and the vendored base has no ``read_record_batches`` at
    all.
    """

    name = "redshift"
    compiler = compiler

    # Inherited from the postgres backend, restated so this backend's exposed
    # secrets are visible here and stay in step with the
    # ``con_name_to_secret_keys`` mirror, which is compared for equality by
    # ``test_declared_secret_keys_are_mirrored``. Declaring nothing would not
    # skip that test -- ``getattr`` finds the inherited tuple -- and declaring
    # ``()`` would narrow ``check_for_exposed_secrets`` to just ``password``,
    # letting a literal ``sslkey`` or ``passfile`` through where postgres
    # raises.
    _secret_keys = (
        "password",
        "sslcert",
        "sslkey",
        "sslrootcert",
        "sslcrl",
        "options",
        "passfile",
    )

    # Deliberately empty. The postgres backend exposes ``connect_env`` and
    # ``connect_examples``, and both are inherited by a plain subclass:
    # ``connect_env`` is backed by ``PostgresConfig`` and the postgres env
    # template, and ``connect_examples`` is hardcoded to a public postgres box.
    # Exposing either on the Redshift namespace would offer a method that does
    # not connect to Redshift. Redshift's own auth modes arrive with the
    # authenticator work.
    _top_level_methods = ()

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
        never reaches the profile or the build hash.
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

    @property
    def current_database(self) -> str:
        """The current schema.

        Overridden because the inherited implementation emits bare
        ``SELECT CURRENT_SCHEMA``, which Redshift rejects with
        ``UndefinedColumn: column "current_schema" does not exist``. Redshift
        requires the parenthesised call.

        No dialect swap fixes this: ``sg.func("current_schema")`` renders
        without parentheses under sqlglot's Postgres *and* Redshift dialects,
        and only the default generator parenthesises it. ``Anonymous`` forces
        the call form under any dialect.

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
    # ``numeric_precision``/``numeric_scale`` and ``character_maximum_length``,
    # so the type string has to be reassembled -- see ``_type_string``.
    #
    # Predicates are appended rather than written inline because the catalog
    # one is conditional, and every value is bound rather than interpolated.
    # Notably this is *not* ``schema_name = ANY(%(dbs)s)``, which is the form
    # the inherited query uses: Redshift has no array type.
    _SVV_ALL_COLUMNS_SELECT = """\
SELECT
  column_name,
  data_type,
  is_nullable,
  character_maximum_length,
  numeric_precision,
  numeric_scale
FROM svv_all_columns
WHERE """

    # Types whose ``svv_all_columns`` row carries a meaningful modifier. The
    # exclusions matter more than the inclusions: the reference's own worked
    # example shows ``numeric_precision`` populated as 32 for an ``integer``
    # and 16 for a ``smallint``, so appending the precision unconditionally
    # would build ``integer(32)``, which is not a type.
    _DECIMAL_TYPES = frozenset({"numeric", "decimal"})
    _SIZED_CHAR_TYPES = frozenset(
        {
            "character varying",
            "varchar",
            "character",
            "char",
            "bpchar",
            "nchar",
            "nvarchar",
        }
    )

    @classmethod
    def _type_string(cls, data_type, char_length, precision, scale) -> str:
        """Reassemble a type string from one ``svv_all_columns`` row."""
        base = (data_type or "").strip()
        lowered = base.lower()
        if lowered in cls._DECIMAL_TYPES and precision is not None:
            return f"{base}({precision},{scale or 0})"
        if lowered in cls._SIZED_CHAR_TYPES and char_length is not None:
            return f"{base}({char_length})"
        return base

    @staticmethod
    def _is_nullable(flag) -> bool:
        """``is_nullable`` is a ``varchar(3)``, not a boolean.

        The reference documents the values as ``yes``/``no`` and prints them as
        ``YES``/``NO`` in its own worked example, so neither case is worth
        betting on. Passing the string straight through as ``nullable=`` would
        mark every column nullable, because both spellings are truthy.
        """
        if isinstance(flag, bool):
            return flag
        return str(flag).strip().lower() == "yes"

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

        Unlike the inherited version this does *not* fold in the session temp
        schema. That would mean calling ``_session_temp_db``, which asks for
        ``pg_my_temp_schema()``, and ``svv_all_columns`` is documented as a
        union of ``SVV_REDSHIFT_COLUMNS`` and external columns -- neither is
        documented to include temporary tables. Both points need a live
        warehouse to settle, so the narrower query is the honest one; the
        consequence is that binding a *temporary* table by name is not
        supported here.
        """
        predicates = ["schema_name = %(schema)s", "table_name = %(table)s"]
        params: dict[str, Any] = {
            "schema": database or self.current_database,
            "table": name,
        }
        if catalog is not None:
            # svv_all_columns spans databases, so an unscoped lookup could
            # match a same-named table in another one.
            predicates.append("database_name = %(catalog)s")
            params["catalog"] = catalog

        query = (
            self._SVV_ALL_COLUMNS_SELECT
            + "\n  AND ".join(predicates)
            + "\nORDER BY ordinal_position ASC"
        )

        con = self.con
        with con.cursor() as cursor, con.transaction():
            rows = cursor.execute(query, params).fetchall()

        if not rows:
            raise exc.TableNotFound(name)

        type_mapper = self.compiler.type_mapper
        return sch.Schema(
            {
                column_name: type_mapper.from_string(
                    self._type_string(data_type, char_length, precision, scale),
                    nullable=self._is_nullable(is_nullable),
                )
                for (
                    column_name,
                    data_type,
                    is_nullable,
                    char_length,
                    precision,
                    scale,
                ) in rows
            }
        )

    @classmethod
    def _type_string_from_column(cls, column) -> str:
        """The type name for one ``psycopg.Column`` of a result description.

        ``type_code`` is a PostgreSQL type OID. Redshift is a PostgreSQL 8.0
        derivative and reports the standard OIDs, which psycopg's builtin
        registry resolves without a round trip.

        An OID the registry does not know -- Redshift's own ``SUPER``,
        ``VARBYTE`` and ``GEOMETRY`` are the expected cases -- raises rather
        than degrading to ``unknown``. A schema that is quietly wrong is the
        failure mode this backend's tests exist to prevent, and the OID goes in
        the message so a live session can map it.

        ``psycopg`` is imported here rather than at module level because it is
        an optional extra: this module is imported when the backend entry point
        is resolved, and a module-level import would make that fail wherever
        the postgres extra is not installed. ``test_core_module_imports_are_
        declared`` enforces exactly this.
        """
        import psycopg  # noqa: PLC0415

        info = psycopg.postgres.types.get(column.type_code)
        if info is None:
            raise exc.UnsupportedBackendType(
                f"{cls.name} returned column {column.name!r} with type OID "
                f"{column.type_code}, which psycopg cannot name; it is most "
                f"likely a Redshift-specific type (SUPER, VARBYTE, GEOMETRY)"
            )
        name = info.name
        if name in cls._DECIMAL_TYPES and column.precision is not None:
            return f"{name}({column.precision},{column.scale or 0})"
        return name

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

        The query is *wrapped* rather than suffixed with ``LIMIT 0``: a query
        that already ends in a ``LIMIT`` would become a syntax error, and one
        ending in a ``UNION`` branch would bind the limit to that branch alone.
        The bound is not cosmetic -- psycopg buffers the whole result client
        side, so an unbounded probe would read the table it selects from.

        Every column comes back nullable, because a result description carries
        no nullability. Whether that differs from what the inherited path
        reported for the same query is not measured here -- it would depend on
        what Redshift records for a temporary view's columns, which needs a
        live warehouse.
        """
        probe = (
            sg.select(sge.Star())
            .from_(sg.parse_one(query, read=self.dialect).subquery(PROBE_ALIAS))
            .limit(0)
            .sql(self.dialect)
        )

        con = self.con
        with con.cursor() as cursor, con.transaction():
            description = list(cursor.execute(probe).description)

        type_mapper = self.compiler.type_mapper
        return sch.Schema(
            {
                column.name: type_mapper.from_string(
                    self._type_string_from_column(column), nullable=True
                )
                for column in description
            }
        )

    def _adbc_unavailable_reason(self) -> str | None:
        """Why the ADBC accelerator cannot be used, or ``None`` if it can.

        The single point of dispatch for both Arrow paths, and the reason
        neither of them needs a catch-all. Availability is decided from local
        facts *before* anything is dialled, so every exception raised by the
        subsequent connect is a real failure and propagates -- which is what
        separates "no driver installed" from "these credentials were rejected".
        Under a rotating IAM credential that distinction is the difference
        between a quiet fallback and silence about an expired password.

        Both checks mirror what ``PgADBC`` would actually do, rather than
        approximating it: it imports ``adbc_driver_postgresql`` and it reads
        ``_con_kwargs["password"]`` to interpolate into a URI. A ``password``
        that is absent *or* ``None`` is disqualifying, and the ``None`` case is
        the one worth stating: it does not raise, it formats into the URI as
        the literal string ``"None"`` and fails later as an auth error against
        a password nobody set.

        This method is also the seam the accelerator work extends. Adding the
        Columnar driver, or ruling ``adbc_driver_postgresql`` in or out against
        a live endpoint, changes a clause here and touches neither Arrow path.

        Note what is *not* settled: whether ``adbc_driver_postgresql`` works
        against Redshift at all is untested -- it is recorded as an alternative
        in ADR-redshift-psycopg-baseline-adbc-optional, needs a live endpoint,
        and may fail on ``pg_catalog`` introspection the way ``CURRENT_SCHEMA``
        did. So a "no reason" answer here means the accelerator is *installed
        and credentialed*, not that it is known to work.
        """
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
        """
        from xorq.common.utils.postgres_utils import PgADBC  # noqa: PLC0415

        if (reason := self._adbc_unavailable_reason()) is not None:
            logger.debug(
                "ADBC accelerator unavailable; using the psycopg baseline",
                backend=self.name,
                reason=reason,
            )
            return None
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
        """Ingest Arrow record batches, over ADBC if it is there and psycopg if
        it is not.

        The postgres implementation is unconditional ADBC. Inheriting it made
        this backend claim a psycopg baseline it did not have, and under that
        baseline an absent driver is the *common* case rather than the edge:
        no Columnar driver is installable from PyPI, and none is built for
        Intel macOS at all. So the fallback is the path that has to work.

        Dispatching here rather than rescuing a failed ADBC attempt is what
        keeps the accelerator an addition: when the driver question is settled
        the ADBC branch gains a clause in ``_adbc_unavailable_reason``, and
        this method does not change shape.

        ``kwargs`` reach ``adbc_ingest`` on the ADBC branch and are dropped on
        the psycopg one, which has nothing to spend them on. That asymmetry is
        inherited rather than chosen: ``read_csv`` and ``read_parquet`` forward
        their *own* reader kwargs down this call, so rejecting unknown ones
        would break both callers on the branch that is meant to be the
        baseline.
        """
        if table_name is None:
            raise ValueError("table_name is required")
        if mode not in INGEST_MODES:
            raise ValueError(f"mode must be one of {INGEST_MODES}, got {mode!r}")

        if (reason := self._adbc_unavailable_reason()) is None:
            return super().read_record_batches(
                record_batches,
                table_name=table_name,
                password=password,
                temporary=temporary,
                mode=mode,
                **kwargs,
            )

        logger.debug(
            "ingesting over psycopg",
            backend=self.name,
            reason=reason,
            table_name=table_name,
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
        ``num_rows`` while working perfectly on the ADBC branch, which accepts
        tables. Which branch runs has to stay an implementation detail.
        """
        if isinstance(record_batches, pa.Table):
            record_batches = record_batches.to_reader()

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
                    cursor.executemany(insert, list(zip(*batch.to_pydict().values())))
        return self.table(table_name)


def connect(**kwargs):
    con = Backend()
    Backend.connect(**kwargs)
    return con
