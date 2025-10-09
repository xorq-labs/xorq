import os
import random
import string
from pathlib import Path
from typing import Optional

from attr import (
    evolve,
    field,
    frozen,
)
from attr.validators import (
    instance_of,
)
from cryptography.hazmat.primitives.asymmetric import (
    rsa,
)
from cryptography.hazmat.primitives.serialization import (
    BestAvailableEncryption,
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
    load_der_private_key,
    load_pem_private_key,
)

from xorq.common.utils.env_utils import (
    parse_env_file,
)


snowflake_env_var_prefix = "SNOWFLAKE_"
default_env_path = Path(".envrc.secrets.snowflake.keypair")


def make_env_name(name, prefix=snowflake_env_var_prefix):
    return f"{prefix}{name.upper()}"


def make_private_key_pwd(k=20, choices=string.ascii_letters + string.digits):
    return "".join(random.choices(choices, k=k))


def encode_utf8(string):
    return string.encode("utf-8")


def filter_none_one(el):
    # for use with *args
    return filter(None, (el,))


@frozen
class SnowflakeKeypair:
    private_key = field(validator=instance_of(rsa.RSAPrivateKey))
    private_key_pwd = field(validator=instance_of(str), factory=make_private_key_pwd)

    prefix = snowflake_env_var_prefix
    default_path = default_env_path

    def get_private_bytes(
        self, encoding=Encoding.PEM, format=PrivateFormat.PKCS8, encrypted=True
    ):
        return self.private_key.private_bytes(
            encoding=encoding,
            format=format,
            encryption_algorithm=BestAvailableEncryption(
                encode_utf8(self.private_key_pwd)
            )
            if encrypted
            else NoEncryption(),
        )

    @property
    def private_bytes(self):
        return self.get_private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.PKCS8,
            encrypted=True,
        )

    @property
    def private_str(self):
        return self.private_bytes.decode("ascii")

    @property
    def private_str_unencrypted(self):
        return self.get_private_bytes(encrypted=False).decode("ascii")

    @property
    def public_key(self):
        return self.private_key.public_key()

    @property
    def public_bytes(self):
        return self.public_key.public_bytes(
            encoding=Encoding.PEM,
            format=PublicFormat.SubjectPublicKeyInfo,
        )

    @property
    def public_str(self):
        return self.public_bytes.decode("ascii")

    def with_password(self, private_key_pwd):
        return evolve(self, private_key_pwd=private_key_pwd)

    def to_envrc(
        self,
        path=default_path,
        prefix=prefix,
        encrypted=True,
    ):
        names_fields = (
            (
                ("private_key", "private_str"),
                ("public_key", "public_str"),
                ("private_key_pwd", "private_key_pwd"),
            )
            if encrypted
            else (
                ("private_key", "private_str_unencrypted"),
                ("public_key", "public_str"),
            )
        )
        text = "\n".join(
            f"export {name}='{value}'"
            for name, value in (
                (
                    make_env_name(name, prefix=prefix),
                    getattr(self, field),
                )
                for (name, field) in names_fields
            )
        )
        (path := Path(path)).write_text(text)
        return path

    @classmethod
    def generate(cls, password=None):
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        return cls(private_key, *filter_none_one(password))

    @classmethod
    def from_bytes_pem(
        cls, private_bytes: bytes, private_key_pwd: Optional[str] = None
    ):
        private_key = load_pem_private_key(
            private_bytes,
            encode_utf8(private_key_pwd) if private_key_pwd else None,
        )
        return cls(private_key, *filter_none_one(private_key_pwd))

    @classmethod
    def from_bytes_der(
        cls, private_bytes: bytes, private_key_pwd: Optional[str] = None
    ):
        private_key = load_der_private_key(
            private_bytes,
            encode_utf8(private_key_pwd) if private_key_pwd else None,
        )
        return cls(private_key, *filter_none_one(private_key_pwd))

    from_bytes = from_bytes_pem

    @classmethod
    def from_str_pem(cls, private_str: str, private_key_pwd: Optional[str] = None):
        return cls.from_bytes(encode_utf8(private_str), private_key_pwd)

    from_str = from_str_pem

    @classmethod
    def from_environment(cls, ctx=os.environ, prefix=prefix):
        kwargs = {
            field: ctx.get(make_env_name(name, prefix=prefix))
            for field, name in (
                ("private_str", "private_key"),
                ("private_key_pwd", "private_key_pwd"),
            )
        }
        return cls.from_str_pem(**kwargs)

    @classmethod
    def from_envrc(cls, path=default_path, prefix=prefix):
        ctx = parse_env_file(path)
        return cls.from_environment(ctx=ctx, prefix=prefix)


def assign_public_key(con, user, public_key_str, do_assert=True):
    from xorq.common.utils.snowflake_utils import execute_statement

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
    fetched = execute_statement(con, stmt, do_assert=do_assert)
    return fetched


def deassign_public_key(con, user, do_assert=True):
    from xorq.common.utils.snowflake_utils import execute_statement

    stmt = f"ALTER USER {user} UNSET RSA_PUBLIC_KEY;"
    fetched = execute_statement(con, stmt, do_assert=do_assert)
    return fetched


def decrypt_private_key_bytes_snowflake(private_key_bytes, password_str):
    # encrypted PEM to unencrypted DER
    private_key = load_pem_private_key(private_key_bytes, password_str.encode("utf-8"))
    return private_key.private_bytes(
        Encoding.DER,
        PrivateFormat.PKCS8,
        NoEncryption(),
    )


def encrypt_private_key_bytes_snowflake_adbc(private_key_bytes, password_str):
    # unencrypted DER to encrypted PEM
    private_key = load_der_private_key(private_key_bytes, None)
    return private_key.private_bytes(
        Encoding.PEM,
        PrivateFormat.PKCS8,
        BestAvailableEncryption(password_str.encode("ascii")),
    )


def maybe_decrypt_private_key(kwargs):
    # snowflake requires DER format for connection
    # ProgrammingError: 251008: Failed to decode private key: Incorrect padding
    # Please provide a valid unencrypted rsa private key in base64-encoded DER format as a str object
    match kwargs:
        case {"private_key": private_key, "private_key_pwd": private_key_pwd, **rest}:
            match private_key:
                case str():
                    private_key = private_key.encode("utf-8")
                case _:
                    raise ValueError("expecting str and only str")
            kwargs = rest | {
                "private_key": decrypt_private_key_bytes_snowflake(
                    private_key, private_key_pwd
                )
            }
        case {"private_key": private_key, **rest}:
            match private_key:
                case str():
                    pass
                case _:
                    raise ValueError("expecting str and only str")
        case _:
            raise ValueError("private_key must be passed")
    return kwargs


def maybe_encrypt_private_key_snowflake_adbc(private_key, private_key_pwd, N=20):
    match (private_key, private_key_pwd):
        case [str(private_key_encrypted), str()]:
            pass
        case [str(), None]:
            private_key_pwd = "".join(random.choices(string.printable, k=N))
            private_key_encrypted = encrypt_private_key_bytes_snowflake_adbc(
                private_key.encode("utf-8"),
                private_key_pwd,
            )
        case _:
            raise ValueError(
                f"invalid types: {(type(private_key), type(private_key_pwd))}"
            )
    return (private_key_encrypted, private_key_pwd)
