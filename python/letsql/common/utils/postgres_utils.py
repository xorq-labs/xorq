import os

import ibis

import letsql.backends.postgres.hotfix  # noqa: F401


def make_credential_defaults():
    return {
        "user": os.environ.get("POSTGRES_USER"),
        "password": os.environ.get("POSTGRES_PASSWORD"),
    }


def make_connection_defaults():
    return {
        "host": os.environ.get("POSTGRES_HOST"),
        "port": os.environ.get("POSTGRES_PORT"),
        "database": os.environ.get("POSTGRES_DATABASE"),
    }


def make_ibis_connection(**kwargs):
    con = ibis.postgres.connect(
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
    (con, name) = (dt.source, dt.name)
    sql = f"""
        SELECT reltuples
        FROM pg_class
        WHERE relname = '{name}'
    """
    do_checkpoint(con)
    do_analyze(con, name)
    ((n_reltuples,),) = con.sql(sql).execute().values
    return n_reltuples


def get_postgres_n_scans(dt):
    (con, name, schemaname) = (dt.source, dt.name, dt.namespace.catalog or "public")
    sql = f"""
        SELECT seq_scan FROM pg_stat_user_tables
        WHERE relname = '{name}' AND schemaname = '{schemaname}';
    """
    ((n_scans,),) = con.sql(sql).execute().values
    return n_scans
