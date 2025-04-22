import adbc_driver_postgresql.dbapi
import sqlglot as sg
import sqlglot.expressions as sge
import toolz
from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
    optional,
)

from xorq.backends.postgres import (
    Backend as PGBackend,
)
from xorq.common.utils.env_utils import (
    EnvConfigable,
    env_templates_dir,
)
from xorq.vendor import ibis


try:
    from sqlglot.expressions import Alter
except ImportError:
    from sqlglot.expressions import AlterTable
else:

    def AlterTable(*args, kind="TABLE", **kwargs):
        return Alter(*args, kind=kind, **kwargs)


@frozen
class PgADBC:
    con = field(validator=instance_of(PGBackend))
    password = field(validator=optional(instance_of(str)), default=None, repr=False)

    def __attrs_post_init__(self):
        if self.password is None:
            object.__setattr__(self, "password", make_credential_defaults()["password"])

    @property
    def params(self):
        dsn_parameters = self.con.con.get_dsn_parameters()
        dct = {
            **toolz.dissoc(
                dsn_parameters,
                "dbname",
                "options",
            ),
            **{
                "database": dsn_parameters["dbname"],
                "password": self.password,
            },
        }
        return dct

    def get_uri(self, **kwargs):
        params = {**self.params, **kwargs}
        uri = f"postgresql://{params['user']}:{params['password']}@{params['host']}:{params['port']}/{params['database']}"
        return uri

    @property
    def uri(self):
        return self.get_uri()

    def get_conn(self, **kwargs):
        return adbc_driver_postgresql.dbapi.connect(self.get_uri(**kwargs))

    @property
    def conn(self):
        return self.get_conn()

    def adbc_ingest(
        self, table_name, record_batch_reader, mode="create", temporary=False, **kwargs
    ):
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.adbc_ingest(
                    table_name,
                    record_batch_reader,
                    mode=mode,
                    temporary=temporary,
                    **kwargs,
                )
            # must commit!
            conn.commit()


PostgresConfig = EnvConfigable.subclass_from_env_file(
    env_templates_dir.joinpath(".env.postgres.template")
)
postgres_config = PostgresConfig.from_env()


def make_credential_defaults():
    return {
        "user": postgres_config["POSTGRES_USER"],
        "password": postgres_config["POSTGRES_PASSWORD"],
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
                sge.RenameTable(
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
            this=sg.table(to_name, quoted=True),
            kind="TABLE",
            expression=sge.Select(
                **{
                    "expressions": [sge.Star()],
                    "from": sge.From(
                        this=sg.table(from_name, quoted=True),
                    ),
                }
            ),
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
