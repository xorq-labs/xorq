from __future__ import annotations

import importlib.util
from types import MappingProxyType
from typing import Any

import psycopg
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


# Settings ``do_connect`` injects below the caller's kwargs, so they reach the
# driver but never ``_con_kwargs``, the profile or the build hash. The live DSN
# reports each back, so ``clone`` drops exactly these from it: one dict, so the
# two cannot drift apart.
CONNECT_DEFAULTS = MappingProxyType({"client_encoding": "utf8"})


class Backend(PostgresBackend):
    """Redshift Serverless, over the PostgreSQL wire protocol.

    Subclasses the *xorq* postgres backend rather than the vendored ibis one:
    the ADBC/psycopg ``to_pyarrow_batches`` and ``read_record_batches`` live on
    the xorq subclass, and the vendored base has no ``read_record_batches`` at
    all.
    """

    name = "redshift"
    compiler = compiler

    _clone_drop_dsn_params = tuple(CONNECT_DEFAULTS)

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
        for key, value in CONNECT_DEFAULTS.items():
            kwargs.setdefault(key, value)
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
        scope; it needs no seam held open here.

        ``password`` is unused and kept because it is the inherited signature:
        ``read_csv`` and ``read_parquet`` both forward it down this call.
        ``kwargs`` are likewise accepted and dropped -- those two callers
        forward their *own* reader kwargs here, so rejecting unknown ones would
        break them.

        The return value is ``self.table(table_name)``, and on a live Redshift
        that raises until the backend has its own table introspection: the
        inherited one reads ``pg_catalog`` objects Redshift lacks. The ingest
        commits before that call, so the table exists when it raises.
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
