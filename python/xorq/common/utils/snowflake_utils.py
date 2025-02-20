import os

import pandas as pd
import snowflake.connector

import xorq as xo


def make_credential_defaults():
    return {
        "user": os.environ.get("SNOWFLAKE_USER"),
        "password": os.environ.get("SNOWFLAKE_PASSWORD"),
    }


def make_connection_defaults():
    return {
        "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
        "role": os.environ.get("SNOWFLAKE_ROLE"),
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
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
