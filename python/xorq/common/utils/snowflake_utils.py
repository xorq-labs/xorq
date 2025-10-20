import pandas as pd
import snowflake.connector
from adbc_driver_snowflake import (
    DatabaseOptions,
    dbapi,
)
from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
)

from xorq.backends.snowflake import Backend as SnowflakeBackend
from xorq.backends.snowflake import SnowflakeAuthenticator, connect
from xorq.common.utils.env_utils import (
    EnvConfigable,
    env_templates_dir,
    filter_existing_env_vars,
    maybe_substitute_env_var,
)


SnowflakeConfig = EnvConfigable.subclass_from_env_file(
    env_templates_dir.joinpath(".env.snowflake.template")
)
snowflake_config = SnowflakeConfig.from_env()


def make_auth_defaults(authenticator=None):
    match authenticator := str(authenticator).lower():
        case SnowflakeAuthenticator.password:
            return {
                "password": "${SNOWFLAKE_PASSWORD}",
            }
        case SnowflakeAuthenticator.mfa:
            return {
                "password": "${SNOWFLAKE_PASSWORD}",
                "authenticator": authenticator,
            }
        case SnowflakeAuthenticator.keypair:
            return {
                "private_key": "${SNOWFLAKE_PRIVATE_KEY}",
                "private_key_pwd": "${SNOWFLAKE_PRIVATE_KEY_PWD}",
                "authenticator": authenticator,
            }
        case SnowflakeAuthenticator.sso:
            return {
                "authenticator": authenticator,
            }
        case _:
            raise ValueError


def make_credential_defaults(authenticator=None):
    dct = {
        "user": "${SNOWFLAKE_USER}",
    } | make_auth_defaults(authenticator)
    return filter_existing_env_vars(dct, snowflake_config)


def make_connection_defaults():
    return {
        "account": snowflake_config["SNOWFLAKE_ACCOUNT"],
        "role": snowflake_config["SNOWFLAKE_ROLE"],
        "warehouse": snowflake_config["SNOWFLAKE_WAREHOUSE"],
    }


def execute_statement(con, statement, do_assert=True):
    fetched = con.raw_sql(statement).fetchall()
    if do_assert:
        assert fetched == [("Statement executed successfully.",)]
    return fetched


def make_connection(
    database,
    schema,
    authenticator=None,
    **kwargs,
):
    con = connect(
        database=f"{database}/{schema}",
        **{
            **make_credential_defaults(authenticator=authenticator),
            **make_connection_defaults(),
            **kwargs,
        },
    )
    return con


def make_raw_connection(database, schema, **kwargs):
    return snowflake.connector.connect(
        database=database,
        schema=schema,
        **{
            **make_credential_defaults(),
            **make_connection_defaults(),
            **kwargs,
        },
    )


def grant_create_database(con, role="public"):
    current_role = con.con.role
    statement = f"""
        USE ROLE accountadmin;
        GRANT CREATE DATABASE ON account TO ROLE {role};
        USE ROLE {current_role}
    """
    execute_statement(con, statement)


def grant_create_schema(con, role="public"):
    current_role = con.con.role
    statement = f"""
        USE ROLE accountadmin;
        GRANT CREATE SCHEMA ON account TO ROLE {role};
        USE ROLE {current_role}
    """
    execute_statement(con, statement)


def get_snowflake_last_modification_time(dt):
    (con, table, database, schema) = (
        dt.source,
        dt.name,
        dt.namespace.catalog,
        dt.namespace.database,
    )
    values = (
        con.table("TABLES", database=(database, "INFORMATION_SCHEMA"))[
            lambda t: t.TABLE_NAME == table
        ][lambda t: t.TABLE_SCHEMA == schema]
        .LAST_ALTERED.execute()
        .values
    )
    if not len(values):
        raise ValueError
    (value, *rest) = values
    if rest:
        raise ValueError
    return value


def get_grants(con, role="public"):
    data = con.raw_sql(f"SHOW GRANTS TO ROLE {role}").fetchall()
    df = pd.DataFrame(
        data,
        columns=(
            "created_on",
            "privilege",
            "granted_on",
            "name",
            "granted_to",
            "grant_option",
            "granted_by_role_type",
            "granted_by",
        ),
    )
    return df


def get_session_query_df(con):
    stmt = """
    SELECT *
    FROM table(information_schema.query_history_by_session())
    ORDER BY start_time;
    """
    session_query_df = con.raw_sql(stmt).fetch_pandas_all()
    return session_query_df


@frozen
class SnowflakeADBC:
    con = field(validator=instance_of(SnowflakeBackend))

    @property
    def password(self):
        return maybe_substitute_env_var(self.con._profile.kwargs_dict.get("password"))

    @property
    def is_keypair_auth(self):
        return self.con._profile.kwargs_dict.get("authenticator") == "snowflake_jwt"

    @property
    def db_kwargs(self, N=20):
        from xorq.common.utils.snowflake_keypair_utils import SnowflakeKeypair

        if self.is_keypair_auth:
            keypair = SnowflakeKeypair.from_bytes_der(self.con.con._private_key)
            return {
                DatabaseOptions.AUTH_TYPE.value: "auth_jwt",
                DatabaseOptions.JWT_PRIVATE_KEY_VALUE.value: keypair.private_bytes,
                DatabaseOptions.JWT_PRIVATE_KEY_PASSWORD.value: keypair.private_key_pwd,
            }
        else:
            return {}

    @property
    def params(self):
        dct = {
            attr: getattr(self.con.con, attr)
            for attr in ("user", "host", "database", "schema", "warehouse", "role")
        } | {
            # ADBC connection always requires a password, even if using private key
            "password": self.password or "nopassword",
        }
        return dct

    def get_uri(self, **kwargs):
        params = {**self.params, **kwargs}
        uri = f"{params['user']}:{params['password']}@{params['host']}/{params['database']}/{params['schema']}?warehouse={params['warehouse']}&role={params['role']}"
        return uri

    @property
    def uri(self):
        return self.get_uri()

    def get_conn(self, **kwargs):
        return dbapi.connect(
            self.get_uri(**kwargs),
            db_kwargs=self.db_kwargs,
        )

    @property
    def conn(self):
        return self.get_conn()

    def adbc_ingest(
        self,
        table_name,
        record_batch_reader,
        mode="create",
        temporary=False,
        database=None,
        conn_kwargs=(),
        **kwargs,
    ):
        def make_use_stmt(con, catalog, db):
            import sqlglot as sg
            import sqlglot.expressions as sge

            use_stmt = sge.Use(
                kind="SCHEMA",
                this=sg.table(db, catalog=catalog, quoted=con.compiler.quoted),
            ).sql(dialect=con.name)
            return use_stmt

        catalog, db = (
            (self.con.current_catalog, self.con.current_database)
            if database is None
            else database
        )
        # create adbc con pointing to a "known safe" catalog/db
        d = {"database": "SNOWFLAKE_SAMPLE_DATA", "schema": "TPCH_SF1"}
        with self.get_conn(**d | dict(conn_kwargs)) as conn:
            with conn.cursor() as cur:
                cur.execute(make_use_stmt(self.con, catalog, db))
                cur.adbc_ingest(
                    table_name,
                    record_batch_reader,
                    mode=mode,
                    temporary=temporary,
                    **kwargs,
                )
            # must commit!
            conn.commit()
