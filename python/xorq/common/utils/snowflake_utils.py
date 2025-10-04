from pathlib import Path
from tempfile import NamedTemporaryFile

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
from snowflake.connector.connection import _get_private_bytes_from_file

from xorq.backends.snowflake import Backend as SnowflakeBackend
from xorq.backends.snowflake import connect
from xorq.common.utils.env_utils import (
    EnvConfigable,
    env_templates_dir,
)
from xorq.common.utils.process_utils import (
    Popened,
)
from xorq.vendor.ibis.backends.profiles import (
    filter_existing_env_vars,
)


SnowflakeConfig = EnvConfigable.subclass_from_env_file(
    env_templates_dir.joinpath(".env.snowflake.template")
)
snowflake_config = SnowflakeConfig.from_env()


def make_credential_defaults():
    dct = {
        # the issue here is that we will try to evaluate an env var and raise if it does not exist
        # so we want to filter out any kwargs not necessary
        "user": "${SNOWFLAKE_USER}",
        "private_key": "${SNOWFLAKE_KEYPAIR_PRIVATE_STR}",
        "private_key_pwd": "${SNOWFLAKE_KEYPAIR_PASSWORD}",
    }
    return filter_existing_env_vars(dct, snowflake_config)


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
    con = connect(
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
class SnowflakeKeypair:
    private_str = field(validator=instance_of(str))
    public_str = field(validator=instance_of(str))
    password = field(validator=optional(instance_of(str)), default=None)
    prefix = "SNOWFLAKE_"
    infix = "KEYPAIR"

    def assign_public_key(self, con, user, do_assert=True):
        return assign_public_key(con, user, self.public_str, do_assert=do_assert)

    def deassign_public_key(self, con, user, do_assert=True):
        return deassign_public_key(con, user, do_assert=do_assert)

    def get_con(self, **kwargs):
        import xorq.api as xo

        return xo.snowflake.connect_env_private_key(
            self.private_str, self.password, **kwargs
        )

    def write_envrc(
        self, path=Path(".envrc.secrets.snowflake.keypair"), prefix=prefix, infix=infix
    ):
        text = "\n".join(
            f"export {name}='{value}'"
            for name, value in (
                (f"{prefix}{infix}_{name.upper()}", getattr(self, name))
                for name in ("private_str", "public_str", "password")
            )
        )
        path.write_text(text)
        return path

    @classmethod
    def generate(cls, password=None):
        (private_str, public_str) = make_snowflake_keypair_strs(password)
        return cls(private_str, public_str, password)

    @classmethod
    def from_environment(cls, prefix=prefix, infix=infix):
        import os

        kwargs = {
            name: os.environ[f"{prefix}{infix}_{name.upper()}"]
            for name in (attr.name for attr in cls.__attrs_attrs__)
        }
        return cls(**kwargs)


def make_snowflake_keypair_strs(password=None):
    # https://docs.snowflake.com/en/user-guide/key-pair-auth#generate-the-private-keys
    (maybe_passout_arg, maybe_passin_arg) = (
        (
            f"-passout 'pass:{password}'",
            f"-passin 'pass:{password}'",
        )
        if password
        else (
            "-nocrypt",
            "",
        )
    )
    with (
        NamedTemporaryFile() as private_key_ntf,
        NamedTemporaryFile() as public_key_ntf,
    ):
        # can we do this with raw lib and not shell out?
        Popened.check_output(
            f"openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -out {private_key_ntf.name} {maybe_passout_arg}"
        )
        Popened.check_output(
            f"openssl rsa -in {private_key_ntf.name} -pubout -out {public_key_ntf.name} {maybe_passin_arg}"
        )
        (private_key_str, public_key_str) = (
            Path(path).read_text()
            for path in (private_key_ntf.name, public_key_ntf.name)
        )
    return private_key_str, public_key_str


def run_statement(con, stmt, do_assert=True):
    fetched = con.raw_sql(stmt).fetchall()
    if do_assert:
        assert fetched == [("Statement executed successfully.",)]
    return fetched


def assign_public_key(con, user, public_key_str, do_assert=True):
    def massage_public_key_str(public_key_str):
        # https://docs.snowflake.com/en/user-guide/key-pair-auth#assign-the-public-key-to-a-snowflake-user
        # # Note: Exclude the public key delimiters in the SQL statement.
        sep = "\n"
        (preamble, *lines, postamble, end) = public_key_str.split(sep)
        assert (preamble, postamble, end) == (
            "-----BEGIN PUBLIC KEY-----",
            "-----END PUBLIC KEY-----",
            "",
        )
        massaged = sep.join(lines)
        return massaged

    massaged_text = massage_public_key_str(public_key_str)
    stmt = f"ALTER USER {user} SET RSA_PUBLIC_KEY='{massaged_text}';"
    fetched = run_statement(con, stmt, do_assert=do_assert)
    return fetched


def deassign_public_key(con, user, do_assert=True):
    stmt = f"ALTER USER {user} UNSET RSA_PUBLIC_KEY;"
    fetched = run_statement(con, stmt, do_assert=do_assert)
    return fetched


def decrypt_private_key_bytes_snowflake(private_key_bytes, password_str):
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        load_pem_private_key,
    )

    private_key = load_pem_private_key(private_key_bytes, password_str.encode("utf-8"))
    return private_key.private_bytes(
        Encoding.DER,
        PrivateFormat.PKCS8,
        NoEncryption(),
    )


def encrypt_private_key_bytes_snowflake_adbc(private_key_bytes, password_str):
    from cryptography.hazmat.primitives.serialization import (
        BestAvailableEncryption,
        Encoding,
        PrivateFormat,
        load_der_private_key,
    )

    private_key = load_der_private_key(private_key_bytes, None)
    return private_key.private_bytes(
        Encoding.PEM,
        PrivateFormat.PKCS8,
        BestAvailableEncryption(password_str.encode("ascii")),
    )


def ensure_private_key_bytes(private_key, private_key_pwd=None):
    if isinstance(private_key, str):
        if (path := Path(private_key)).exists():
            private_key = path
        else:
            private_key = private_key.encode()
    match private_key:
        case Path():
            private_key = _get_private_bytes_from_file(private_key, private_key_pwd)
        case bytes():
            if private_key_pwd is not None:
                private_key = decrypt_private_key_bytes_snowflake(
                    private_key, private_key_pwd
                )
        case _:
            raise NotImplementedError(f"Can't handle type {type(private_key)}")
    return private_key


def maybe_decrypt_private_key(kwargs):
    match kwargs:
        case {"private_key": private_key, "private_key_pwd": private_key_pwd, **rest}:
            if isinstance(private_key, bytes):
                raise ValueError
            if isinstance(private_key, str):
                private_key = private_key.encode("utf-8")
            kwargs = rest | {
                "private_key": decrypt_private_key_bytes_snowflake(
                    private_key, private_key_pwd
                )
            }
        case {"private_key_pwd": private_key_pwd, **rest}:
            raise ValueError("private_key_pwd passed without private_key")
        case _:
            pass
    return kwargs


@frozen
class SnowflakeADBC:
    con = field(validator=instance_of(SnowflakeBackend))

    @property
    def password(self):
        from xorq.vendor.ibis.backends.profiles import maybe_process_env_var

        return maybe_process_env_var(self.con._profile.kwargs_dict.get("password"))

    @property
    def is_keypair_auth(self):
        return self.con._profile.kwargs_dict.get("authenticator") == "snowflake_jwt"

    @property
    def db_kwargs(self, N=20):
        import random
        import string

        from adbc_driver_snowflake import DatabaseOptions

        if self.is_keypair_auth:
            match self.con._profile.kwargs_dict:
                case {
                    "private_key": private_key,
                    "private_key_pwd": private_key_pwd,
                    **rest,  # noqa: F841
                }:
                    private_key_encrypted = private_key
                case {
                    "private_key": private_key,
                    **rest,  # noqa: F841
                }:
                    # ADBC connection requires an encrypted private key, so encrypt on the fly
                    private_key_pwd = "".join(random.choices(string.printable, k=N))
                    private_key_encrypted = encrypt_private_key_bytes_snowflake_adbc(
                        # how do we know that the private key is decrypted in kwargs?
                        self.con._profile.kwargs_dict["private_key"].encode("utf-8"),
                        private_key_pwd,
                    )
                case _:
                    raise ValueError("must have private_key")

            return {
                DatabaseOptions.AUTH_TYPE.value: "auth_jwt",
                DatabaseOptions.JWT_PRIVATE_KEY_VALUE.value: private_key_encrypted,
                DatabaseOptions.JWT_PRIVATE_KEY_PASSWORD.value: private_key_pwd,
            }
        else:
            return {}

    @property
    def params(self):
        con = self.con.con

        dct = {
            "user": con.user,
            # ADBC connection always requires a password, even if using private key
            "password": self.password or "nopassword",
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
        return dbapi.connect(
            self.get_uri(**kwargs),
            db_kwargs=self.db_kwargs,
        )

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
