from __future__ import annotations

from typing import Any

import sqlglot as sg
import sqlglot.expressions as sge

from xorq.backends.postgres import Backend as PostgresBackend
from xorq.backends.redshift.compiler import compiler


__all__ = [
    "Backend",
    "connect",
]

# Redshift listens on 5439; the postgres backend defaults to 5432.
DEFAULT_PORT = 5439


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


def connect(**kwargs):
    con = Backend()
    Backend.connect(**kwargs)
    return con
