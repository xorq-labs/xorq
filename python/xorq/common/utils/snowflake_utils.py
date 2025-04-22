import pandas as pd
import snowflake.connector
from adbc_driver_snowflake import dbapi
from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
    optional,
)

import xorq as xo
from xorq.backends.snowflake import (
    Backend as SnowflakeBackend,
)
from xorq.common.utils.env_utils import (
    EnvConfigable,
    env_templates_dir,
)


SnowflakeConfig = EnvConfigable.subclass_from_env_file(
    env_templates_dir.joinpath(".env.snowflake.template")
)
snowflake_config = SnowflakeConfig.from_env()


def make_credential_defaults():
    return {
        "user": snowflake_config["SNOWFLAKE_USER"],
        "password": snowflake_config["SNOWFLAKE_PASSWORD"],
    }


def make_connection_defaults():
    return {
        "account": snowflake_config["SNOWFLAKE_ACCOUNT"],
        "role": snowflake_config["SNOWFLAKE_ROLE"],
        "warehouse": snowflake_config["SNOWFLAKE_WAREHOUSE"],
    }


def execute_statement(con, statement):
    (((resp,), *rest0), *rest1) = con.con.execute_string(statement)
    if rest0 or (resp != "Statement executed successfully."):
        raise ValueError


def make_connection(
    database,
    schema,
    **kwargs,
):
    con = xo.snowflake.connect(
        database=f"{database}/{schema}",
        **{
            **make_credential_defaults(),
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
    if not values:
        raise ValueError
    (value,) = values
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
    password = field(validator=optional(instance_of(str)), default=None, repr=False)

    def __attrs_post_init__(self):
        if self.password is None:
            object.__setattr__(self, "password", make_credential_defaults()["password"])

    @property
    def params(self):
        con = self.con.con

        dct = {
            "user": con.user,
            "password": self.password,
            "host": con.host,
            "database": con.database,
            "schema": con.schema,
            "warehouse": con.warehouse,
            "role": con.role,
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
        return dbapi.connect(self.get_uri(**kwargs))

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
