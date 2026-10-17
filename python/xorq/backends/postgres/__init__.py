from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import sqlglot as sg
import sqlglot.expressions as sge
import toolz
from adbc_driver_manager import ProgrammingError as ADBCProgrammingError

from xorq.backends.postgres.compiler import compiler
from xorq.common.utils.defer_utils import (
    read_csv_rbr,
)
from xorq.common.utils.env_utils import maybe_substitute_env_var
from xorq.common.utils.logging_utils import get_logger
from xorq.config import default_backend
from xorq.vendor.ibis import util
from xorq.vendor.ibis.backends.postgres import Backend as IbisPostgresBackend
from xorq.vendor.ibis.expr import types as ir
from xorq.vendor.ibis.util import (
    gen_name,
)


logger = get_logger(__name__)

# Where to connect: the DSN keys a clone may fill in when the caller left them
# out (see ``clone``). ``dbname`` is carried separately, as ``database``.
_CLONE_ADDRESS_KEYS = frozenset(("host", "port", "user", "dbname"))


__all__ = [
    "Backend",
    "connect",
]


def _default_ingest_name(table_name, *, temporary, prefix):
    if table_name is not None:
        return table_name
    if not temporary:
        raise ValueError("If `table_name` is not provided, `temporary` must be True")
    return gen_name(prefix)


class Backend(IbisPostgresBackend):
    _top_level_methods = ("connect_examples", "connect_env")
    _secret_keys = (
        "password",
        "sslcert",
        "sslkey",
        "sslrootcert",
        "sslcrl",
        "options",
        "passfile",
    )
    compiler = compiler

    @classmethod
    def connect_env(cls, **kwargs):
        from xorq.common.utils.postgres_utils import make_connection  # noqa: PLC0415

        return make_connection(**kwargs)

    @classmethod
    def connect_examples(cls, **kwargs):
        examples_kwargs = {
            "host": "examples.letsql.com",
            "user": "letsql",
            "password": "letsql",
            "database": "letsql",
        }
        return cls().connect(**(examples_kwargs | kwargs))

    def _open_adbc_conn_or_none(self):
        """Open an ADBC connection for the Arrow paths, or ``None`` if there is
        not one to be had.

        A seam, with one narrow behaviour change: this import used to run when
        ``to_pyarrow_batches`` was called, and now runs on the first batch
        read, because the only caller is inside the generator. A missing
        ``adbc_driver_postgresql`` still raises -- the import is above the
        ``try``, not inside it -- just later, and from inside iteration.

        Postgres keeps the catch-all on purpose: an absent ``password`` in
        ``_con_kwargs`` is an ordinary way for the ADBC URI to be unbuildable
        while psycopg is perfectly connected -- a ``.pgpass``, a service file,
        ``PGPASSWORD`` -- and for a static credential, quietly using the
        psycopg path is the right answer.

        A subclass whose credentials rotate cannot afford that catch-all,
        because driver-absent and auth-failed arrive here as the same
        exception. Overriding this one method is how it tells them apart.
        """
        from xorq.common.utils.postgres_utils import PgADBC  # noqa: PLC0415

        try:
            return PgADBC(self).get_conn()
        except Exception:
            # A genuine config/auth error is indistinguishable here from a
            # missing ADBC URI; both fall through to the psycopg path, so log
            # at debug to keep the real cause diagnosable.
            logger.debug(
                "ADBC connection unavailable; falling back to psycopg",
                backend=self.name,
                exc_info=True,
            )
            return None

    @util.experimental
    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
        **_: Any,
    ) -> pa.ipc.RecordBatchReader:
        def _batches(self, *, pyarrow_schema, struct_type, query):
            # Primary path: ADBC opens its own independent postgres connection
            # per call, so concurrent generators never share psycopg connection
            # state (eliminates OutOfOrderTransactionNesting).
            adbc_con = self._open_adbc_conn_or_none()

            if adbc_con is not None:
                cur = adbc_con.cursor()
                # ``fall_through`` exists for the one recoverable failure (temp
                # table invisible on a fresh ADBC connection), but it catches
                # every ``ADBCProgrammingError``, and ADBC reports syntax and
                # permission errors that way too; those re-run on psycopg and
                # raise there. The ``finally`` always closes both the cursor
                # and connection, so any other error raised by ``execute``
                # (network, for one) propagates without leaking the ADBC
                # resources.
                fall_through = False
                try:
                    try:
                        cur.execute(query)
                    except ADBCProgrammingError:
                        fall_through = True
                    if not fall_through:
                        for batch in cur.fetch_record_batch():
                            yield batch.cast(pyarrow_schema)
                finally:
                    cur.close()
                    adbc_con.close()
                if not fall_through:
                    return

            # Psycopg fallback: session-local temp tables or no ADBC URI.
            con = self.con
            with (
                con.cursor(name=util.gen_name("postgres_cursor")) as cursor,
                con.transaction(),
            ):
                cur = cursor.execute(query)
                while batch := cur.fetchmany(chunk_size):
                    yield pa.RecordBatch.from_struct_array(
                        pa.array(batch, type=struct_type)
                    )

        self._run_pre_execute_hooks(expr)

        raw_schema = expr.as_table().schema()
        query = self.compile(expr, limit=limit, params=params)
        pyarrow_schema = raw_schema.to_pyarrow()
        return pa.RecordBatchReader.from_batches(
            pyarrow_schema,
            _batches(
                self,
                pyarrow_schema=pyarrow_schema,
                struct_type=raw_schema.as_struct().to_pyarrow(),
                query=query,
            ),
        )

    def read_record_batches(
        self,
        record_batches: pa.RecordBatchReader,
        table_name: str | None = None,
        temporary: bool = False,
        mode: str = "create",
        **kwargs: Any,
    ) -> ir.Table:
        """Ingest ``record_batches`` into ``table_name`` over ADBC.

        Parameters
        ----------
        record_batches
            The rows to write; consumed once.
        table_name
            Target table. Created, appended to or replaced per ``mode``.
        temporary
            Copy the ingested table into a session temporary table of the
            same name afterwards.
        mode
            One of ``create``, ``append``, ``replace``, ``create_append``.
        **kwargs
            ADBC ingest options (``db_schema_name``, ``catalog_name``, ...).

        Returns
        -------
        ir.Table
            The ingested table, read from ``db_schema_name`` when given.

        Contract
        --------
        Preconditions
            - ``table_name`` must be a str; ``None`` reaches the ADBC
              target-table option and fails there (implicit).
            - ``temporary`` and ``db_schema_name`` cannot be combined: the
              temporary copy goes through the search path, so it would take
              a same-named table of the current schema (explicit check).
            - ``mode`` must be one ADBC accepts; any other value is rejected
              by the cursor after the connection is open (implicit).

        Raises
        ------
        ValueError
            ``temporary`` with ``db_schema_name``, before any ingest.
        RuntimeError
            The ADBC driver build has no bulk ingest (from ``adbc_ingest``).
        adbc_driver_manager.Error
            Connection, permission or schema errors from the ADBC connection
            it opens from this backend's credentials; not caught.
        """
        from xorq.common.utils.postgres_utils import (  # noqa: PLC0415
            PgADBC,
            make_table_temporary,
        )

        schema = kwargs.get("db_schema_name")
        if temporary and schema is not None:
            # the temporary copy is made through the search path, so it would
            # take a same-named table in the current schema, not the ingest
            raise ValueError("temporary is not supported with db_schema_name")
        pgadbc = PgADBC(self)
        pgadbc.adbc_ingest(table_name, record_batches, mode=mode, **kwargs)
        if temporary:
            make_table_temporary(self, table_name)
        # a one-element tuple keeps the schema a literal name (a str is parsed
        # as catalog.schema)
        return self.table(table_name, database=(schema,) if schema else None)

    def read_parquet(
        self,
        path: str | Path,
        table_name: str | None = None,
        temporary: bool = False,
        mode: str = "create",
        ingest_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Read a parquet file through the default backend and ingest it.

        Parameters
        ----------
        path
            A path or URL the default backend's ``read_parquet`` accepts.
        table_name
            Target table; a generated ``ls-read-parquet-*`` name when
            ``None``, allowed only with ``temporary=True``.
        temporary, mode
            As on ``read_record_batches``.
        ingest_kwargs
            ADBC ingest options (``db_schema_name``, ...), spelled the same
            way on ``read_csv``.
        **kwargs
            Reader options for the default backend's ``read_parquet``.

        Returns
        -------
        ir.Table
            The ingested table.

        Contract
        --------
        Preconditions
            - ``table_name=None`` requires ``temporary=True`` (explicit
              check in ``_default_ingest_name``).
            - On a deferred read the recorded kwargs must hash and build;
              ``ingest_kwargs`` is one of them.

        Raises
        ------
        ValueError
            ``table_name=None`` without ``temporary``, or ``temporary`` with
            ``db_schema_name`` (from ``read_record_batches``).
        TypeError
            A reader option the default backend's ``read_parquet`` does not
            accept (implicit, from the datafusion call); the old route of
            ADBC options through ``**kwargs`` fails this way now.
        FileNotFoundError
            From the default backend when ``path`` does not resolve.
        """
        # reader options stay with the reader; the ADBC ingest options travel
        # under ingest_kwargs, spelled the same way on read_csv
        table_name = _default_ingest_name(
            table_name, temporary=temporary, prefix="ls-read-parquet"
        )
        record_batches = (
            default_backend().read_parquet(path, **kwargs).to_pyarrow_batches()
        )
        return self.read_record_batches(
            record_batches=record_batches,
            table_name=table_name,
            temporary=temporary,
            mode=mode,
            **(ingest_kwargs or {}),
        )

    def read_csv(
        self,
        path,
        table_name=None,
        chunksize=10_000,
        temporary=False,
        mode="create",
        schema=None,
        ingest_kwargs=None,
        **kwargs,
    ):
        """Read one or more csv files with pandas, in chunks, and ingest them.

        Parameters
        ----------
        path
            A path, a glob, or a list of them, read in order.
        table_name
            Target table; a generated ``ls-read-csv-*`` name when ``None``,
            allowed only with ``temporary=True``.
        chunksize
            Rows per pandas chunk and per ingested batch; must be a positive
            int.
        temporary, mode
            As on ``read_record_batches``.
        schema
            The ibis schema to read with; inferred by pandas when ``None``.
        ingest_kwargs
            ADBC ingest options (``db_schema_name``, ...), spelled the same
            way on ``read_parquet``.
        **kwargs
            Reader options for ``pandas.read_csv``.

        Returns
        -------
        ir.Table
            The ingested table.

        Contract
        --------
        Preconditions
            - ``chunksize`` must not be ``None`` (explicit check); a value
              below one is rejected by pandas (implicit).
            - ``table_name=None`` requires ``temporary=True`` (explicit
              check in ``_default_ingest_name``).
            - ``schema`` and a ``dtype`` reader option cannot be combined
              (explicit check in ``read_csv_rbr``).
            - Every file must share the schema of the first; a later file
              that disagrees fails inside the ingest, after earlier chunks
              were written.

        Raises
        ------
        ValueError
            ``chunksize=None``; ``table_name=None`` without ``temporary``;
            ``temporary`` with ``db_schema_name``.
        TypeError
            A reader option ``pandas.read_csv`` does not accept (implicit);
            ``password``, once accepted and ignored, fails this way now.
        FileNotFoundError
            From pandas when ``path`` does not resolve.
        """
        if chunksize is None:
            raise ValueError("chunksize must not be None")
        table_name = _default_ingest_name(
            table_name, temporary=temporary, prefix="ls-read-csv"
        )
        # reader options stay with the reader: the ingest cursor rejects them
        record_batches = read_csv_rbr(
            path, schema=schema, chunksize=chunksize, **kwargs
        )
        return self.read_record_batches(
            record_batches=record_batches,
            table_name=table_name,
            temporary=temporary,
            mode=mode,
            **(ingest_kwargs or {}),
        )

    def create_catalog(self, name: str, force: bool = False) -> None:
        # https://stackoverflow.com/a/43634941
        if force:
            raise ValueError("postgres does not support force=True for create_catalog")
        quoted = self.compiler.quoted
        create_stmt = sge.Create(
            this=sg.to_identifier(name, quoted=quoted), kind="DATABASE", exists=force
        )
        (prev_autocommit, self.con.autocommit) = (self.con.autocommit, True)
        with self._safe_raw_sql(create_stmt):
            pass
        self.con.autocommit = prev_autocommit

    def _clone_credential_default_password(self) -> str | None:
        """The env reference ``clone`` falls back to when no password is known.

        A hook rather than a direct call because the fallback is
        service-specific and a wrong one is worse than none: a subclass that
        inherited this method unchanged would dial ITS warehouse with whatever
        ``$POSTGRES_PASSWORD`` happens to hold locally, or refuse with a
        message naming a service the caller never mentioned. Returning ``None``
        means "this backend has no environment default", which ``clone`` turns
        into an error that names the backend.
        """
        from xorq.common.utils.postgres_utils import (  # noqa: PLC0415
            make_credential_defaults,  # noqa: PLC0415
        )

        return make_credential_defaults()["password"]

    def clone(self, password: str | None = None, **kwargs: Any) -> Backend:
        """necessary because "UnsupportedOperationError: postgres does not support creating a database in a different catalog" """
        dsn_parameters = self.con.info.get_parameters()
        # Which DSN settings the clone carries is decided by provenance.
        # ``get_parameters`` reports every non-default libpq setting, including
        # ones nobody asked for (libpq 17+ reports ``sslcertmode``), and reports
        # resolved values where the caller passed an env reference such as
        # ``$POSTGRES_USER``. A clone that took either hashed differently from
        # its source. So when the source was opened with kwargs, those kwargs
        # are the caller's values and win, and the DSN fills in only address
        # keys the caller left out; a setting libpq derived from the
        # environment is re-derived from the same environment. A
        # ``from_connection`` source has no kwargs, so the DSN is all there is:
        # it is carried less the settings libpq derived on its own, which the
        # clone's libpq derives again, and which would otherwise enter
        # ``_con_kwargs`` as caller settings and the ADBC URI with them.
        if not self._con_kwargs:
            from xorq.common.utils.postgres_utils import (  # noqa: PLC0415
                libpq_derived_settings,
            )

            derived = libpq_derived_settings()
            dsn_parameters = {
                **{
                    key: value
                    for key, value in dsn_parameters.items()
                    if key in _CLONE_ADDRESS_KEYS or derived.get(key) != value
                },
                # libpq omits a default port from ``get_parameters``, so
                # without this a subclass with a different default port dials
                # that one.
                "port": self.con.info.port,
            }
        else:
            dsn_parameters = toolz.keyfilter(
                lambda key: (
                    key in _CLONE_ADDRESS_KEYS
                    and key not in self._con_kwargs
                    and not (key == "dbname" and "database" in self._con_kwargs)
                ),
                dsn_parameters,
            )
        dct = {
            # ``get_parameters`` reports libpq conninfo keywords only, so
            # settings that never reach libpq cannot come back out of it:
            # ``schema`` is applied by ``_post_connect``. Take those from the
            # kwargs this connection was opened with.
            **self._con_kwargs,
            # ``autocommit`` is a psycopg ``Connection`` setting, so it is not
            # in the DSN either -- and it is absent from ``_con_kwargs`` too
            # whenever the source was built by ``from_connection`` or
            # positionally. The live connection knows it on every path, and
            # ``create_catalog`` above already treats it as the truth.
            "autocommit": self.con.autocommit,
            # For a ``from_connection`` source, ``options`` is kept: it carries
            # libpq runtime settings, most importantly ``search_path``, and
            # dropping it silently changed which schema unqualified names in
            # the clone resolved against. A source with kwargs keeps it only as
            # the caller passed it, which is where it came from.
            #
            # ``hostaddr`` is dropped: psycopg resolves ``host`` when it opens
            # a connection and records the IP it picked as ``hostaddr``
            # (``psycopg._conninfo_attempts._resolve_hostnames``), so what the
            # DSN reports there is resolver output, not something the caller
            # asked for. Carrying it over bakes that IP into the clone's
            # profile, ``dehydrate_cons`` persists the profile into a build,
            # and a later hydration then targets a stale address after DNS
            # failover. A ``hostaddr`` the caller did pass is in
            # ``_con_kwargs`` and survives.
            #
            **toolz.dissoc(dsn_parameters, "dbname", "hostaddr"),
            # ...but ``_con_kwargs`` and the DSN hold resolved values, and
            # ``Profile.from_con`` bakes whatever is in here into the clone's
            # profile, which ``xo.build`` writes to disk without a secret
            # check. The source's profile kept the form the caller actually
            # passed (an env reference such as ``$POSTGRES_USER``), so for every
            # key the caller passed, and for every declared secret, that form
            # wins: a secret stays out of the build, and the clone hashes as
            # its source does.
            **{
                key: value
                for key, value in (
                    self._profile.kwargs_dict if self._profile is not None else {}
                ).items()
                if (key in self._secret_keys or key in self._con_kwargs)
                and value is not None
            },
            **(
                {"database": dsn_parameters["dbname"]}
                if "dbname" in dsn_parameters
                else {}
            ),
            **kwargs,
        }
        # Password precedence: explicit > the source's own > the env default.
        # The DSN never reports the password, so "the source's own" is whatever
        # ``_con_kwargs`` and the secret merge left in ``dct``: the env
        # reference the caller passed, where they passed one.
        password = (
            password or dct.get("password") or self._clone_credential_default_password()
        )
        if password is None:
            raise ValueError(
                f"password is required: {self.name} has no environment default "
                f"to fall back on, and neither this call nor the source "
                f"connection supplied one"
            )
        # ``BaseBackend.__init__`` resolves env references with a bare
        # ``ctx[name]``, so an unpopulated one would surface as a ``KeyError``
        # from inside ``connect``. Name what is actually missing instead.
        try:
            maybe_substitute_env_var(password)
        except KeyError as exc:
            raise ValueError(
                f"password is required if {exc.args[0]} env var is not populated"
            ) from exc
        # ``self.connect``, not the module-level ``connect``: the inherited
        # ``BaseBackend.connect`` instantiates ``self.__class__``, so a
        # subclass clones into its own class rather than into this one.
        return self.connect(**{**dct, "password": password})


def connect(**kwargs: Any) -> Backend:
    """Connect to postgres.

    Retained as public API; the loader builds ``xo.postgres.connect`` from the
    bound ``Backend.connect`` and does not come through here.
    """
    # Was ``Backend.connect(**kwargs)``: an unbound call that raised
    # ``TypeError`` and discarded its result. ``clone`` was the only caller.
    return Backend().connect(**kwargs)
