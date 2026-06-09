from xorq.common.compat import StrEnum


class SnowflakeAuthenticator(StrEnum):
    # https://docs.snowflake.com/en/developer-guide/node-js/nodejs-driver-options#label-nodejs-auth-options
    password = "none"
    mfa = "username_password_mfa"
    keypair = "snowflake_jwt"
    sso = "externalbrowser"
