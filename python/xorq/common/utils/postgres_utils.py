import re
import urllib.parse

import adbc_driver_postgresql.dbapi
import psycopg
import sqlglot as sg
import sqlglot.expressions as sge
from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
)

from xorq.backends.postgres import (
    Backend as PGBackend,
)
from xorq.common.utils.adbc_utils import ADBCBase
from xorq.common.utils.env_utils import (
    EnvConfigable,
    env_templates_dir,
)
from xorq.vendor import ibis
from xorq.vendor.ibis.backends.sql.compilers.base import STAR, AlterTable, RenameTable


# libpq keywords a caller may pass through ``connect`` beyond the ones the URI's
# authority carries. Taken from psycopg's libpq, and forwarded only when the
# caller passed them: the live connection's ``get_parameters()`` also reports
# settings nobody asked for (libpq 17+ reports ``sslcertmode``), and libpq
# rejects a URI parameter it does not know, so a key from psycopg's newer libpq
# could fail against the driver's. Measured with an unknown keyword; whether any
# driver release this project allows predates ``sslcertmode`` is not.
LIBPQ_SETTING_KEYWORDS = frozenset(
    option.keyword.decode() for option in psycopg.pq.Conninfo.get_defaults()
) - {"user", "password", "host", "port", "dbname"}


def libpq_derived_settings() -> dict[str, str]:
    """What psycopg's libpq sets on a connection nobody configured.

    ``get_parameters`` hides a value equal to its compiled default, but libpq
    fills some settings in at connect time instead (``sslcertmode`` from libpq
    17), and those it reports as though they were chosen. Measured rather than
    listed, so it keeps up with the libpq psycopg bundles: ``connect_start``
    processes the options before dialling, and a socket directory that does
    not exist fails without touching the network. The environment is read
    here as at any connect, so a ``PGSSLMODE`` counts as derived too.
    """
    probe = psycopg.pq.PGconn.connect_start(b"host=/nonexistent-xorq-libpq-probe")
    try:
        return {
            option.keyword.decode(): option.val.decode()
            for option in probe.info
            if option.val is not None
        }
    finally:
        probe.finish()


def search_path_option(schema: str) -> str:
    r"""A libpq ``options`` argument setting ``search_path`` to ``schema``.

    libpq splits ``options`` on whitespace unless it is backslash-escaped, and
    reads ``\\`` as one backslash, so both are escaped. The value is otherwise
    what ``_post_connect`` passes to ``set_config``: a ``search_path`` string,
    so a comma list keeps its meaning.
    """
    escaped = re.sub(r"([\\\s])", r"\\\1", schema)
    return f"-csearch_path={escaped}"


@frozen
class PgADBC(ADBCBase):
    con = field(validator=instance_of(PGBackend))

    @property
    def password(self):
        return self.con._con_kwargs["password"]

    @property
    def params(self):
        con_info = self.con.con.info
        dct = {key: getattr(con_info, key) for key in ("user", "host", "port")} | {
            "database": con_info.dbname,
            "password": self.password,
        }
        return dct

    @property
    def settings(self):
        """The query part of the URI: what psycopg's connection was configured
        with beyond its address, so the two connections agree.

        The caller's libpq settings (``sslmode``, ``sslrootcert``,
        ``options``, ...) as passed to ``connect``, plus ``schema`` as a
        ``search_path`` in ``options``. psycopg gets the schema from
        ``_post_connect``'s ``set_config``, which the ADBC connection never
        runs; without it the ADBC connection resolves unqualified names
        against the server default ``'$user, public'``.
        """
        con_kwargs = self.con._con_kwargs  # xorq-style: disable=protected-access
        settings = {
            key: str(value)
            for key, value in con_kwargs.items()
            if key in LIBPQ_SETTING_KEYWORDS and value is not None
        }
        if schema := con_kwargs.get("schema"):
            settings["options"] = " ".join(
                filter(None, (settings.get("options"), search_path_option(schema)))
            )
        return settings

    @property
    def uri(self):
        return self.get_uri()

    @property
    def conn(self):
        return self.get_conn()

    def get_uri(self, **kwargs):
        params = {**self.params, **kwargs}
        # Userinfo is percent-encoded: libpq ends it at the first ``@`` and
        # splits user from password at the first ``:``, so a raw Redshift IAM
        # user (``IAMR:<role>``) or a password containing ``@ / # %`` parses
        # into different credentials.
        user, password = (
            urllib.parse.quote(str(params[key]), safe="")
            for key in ("user", "password")
        )
        uri = f"postgresql://{user}:{password}@{params['host']}:{params['port']}/{params['database']}"
        if query := urllib.parse.urlencode(self.settings, quote_via=urllib.parse.quote):
            uri = f"{uri}?{query}"
        return uri

    def get_conn(self, **kwargs):
        return adbc_driver_postgresql.dbapi.connect(self.get_uri(**kwargs))


PostgresConfig = EnvConfigable.subclass_from_env_file(
    env_templates_dir.joinpath(".env.postgres.template")
)
postgres_config = PostgresConfig.from_env()


def make_credential_defaults():
    return {
        "user": "$POSTGRES_USER",
        "password": "$POSTGRES_PASSWORD",
    }


def make_connection_defaults():
    return {
        "host": postgres_config["POSTGRES_HOST"],
        "port": postgres_config["POSTGRES_PORT"],
        "database": postgres_config["POSTGRES_DATABASE"],
    }


def make_connection(**kwargs):
    con = PGBackend()
    con = con.connect(
        **{
            **make_credential_defaults(),
            **make_connection_defaults(),
            **kwargs,
        }
    )
    return con


def do_checkpoint(con):
    con.raw_sql("CHECKPOINT")


def do_analyze(con, name):
    con.raw_sql(f'ANALYZE "{name}"')


def get_postgres_n_changes(dt):
    (con, name, schemaname) = (dt.source, dt.name, dt.namespace.catalog or "public")
    sql = f"""
        SELECT n_tup_upd + n_tup_ins + n_tup_del AS n_changes FROM pg_stat_user_tables
        WHERE relname = '{name}' AND schemaname = '{schemaname}';
    """
    do_checkpoint(con)
    do_analyze(con, name)
    ((n_changes,),) = con.sql(sql).execute().values
    return n_changes


def get_postgres_n_reltuples(dt):
    # FIXME: determine how to track "temporary" tables
    (con, name) = (dt.source, dt.name)
    which = "reltuples"
    sql = f"""
        SELECT {which}
        FROM pg_class
        WHERE relname = '{name}'
        AND relpersistence = 'p'
    """
    do_checkpoint(con)
    do_analyze(con, name)
    (n_reltuples, *rest) = con.sql(sql).execute()[which]
    if rest:
        raise ValueError(str((n_reltuples, rest)))
    return n_reltuples


def get_postgres_n_scans(dt):
    (con, name, schemaname) = (dt.source, dt.name, dt.namespace.catalog or "public")
    sql = f"""
        SELECT seq_scan FROM pg_stat_user_tables
        WHERE relname = '{name}' AND schemaname = '{schemaname}';
    """
    ((n_scans,),) = con.sql(sql).execute().values
    return n_scans


def make_table_temporary(con, name):
    def rename_table_pg(con, old_name, new_name):
        # rename_stmt = f"ALTER TABLE {old_name} RENAME TO {new_name}"
        # sg.parse_one(rename_stmt)
        sql = AlterTable(
            this=sg.table(old_name, quoted=True),
            actions=[
                RenameTable(
                    this=sg.table(new_name, quoted=True),
                ),
            ],
        )
        rename_stmt = sql.sql(dialect="postgres")
        with con._safe_raw_sql(rename_stmt):
            pass

    def copy_table_pg(con, from_name, to_name, temporary=False):
        # copy_stmt = f"CREATE {'TEMP ' if temporary else ''}TABLE {to_name} AS SELECT * FROM {from_name}"
        # sg.parse_one(copy_stmt)
        sql = sge.Create(
            kind="TABLE",
            this=sg.table(to_name, quoted=True),
            expression=sg.select(STAR).from_(sg.table(from_name, quoted=True)),
            properties=sge.Properties(
                expressions=[sge.TemporaryProperty()] if temporary else []
            ),
        )
        copy_stmt = sql.sql(dialect="postgres")
        with con._safe_raw_sql(copy_stmt):
            pass

    tmp_name = ibis.util.gen_name(f"tmp-{name}")
    rename_table_pg(con, name, tmp_name)
    copy_table_pg(con, tmp_name, name, temporary=True)
    con.drop_table(tmp_name)
