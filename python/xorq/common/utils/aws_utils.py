from xorq.common.utils.env_utils import (
    EnvConfigable,
    env_templates_dir,
)


AWSConfig = EnvConfigable.from_env_file(env_templates_dir.joinpath(".env.aws.template"))
aws_config = AWSConfig.from_env()


def make_s3_credentials_defaults():
    return {
        "aws.access_key_id": aws_config.get("AWS_ACCESS_KEY_ID"),
        "aws.secret_access_key": aws_config.get("AWS_SECRET_ACCESS_KEY"),
    }


def make_s3_connection():
    connection = {
        **make_s3_credentials_defaults(),
        "aws.session_token": aws_config.get("AWS_SESSION_TOKEN", ""),
        "aws.allow_http": aws_config.get("AWS_ALLOW_HTTP", "false"),
    }

    if region := aws_config.get("AWS_REGION"):
        connection["aws.region"] = region

    return connection, connection_is_set(connection)


def connection_is_set(connection: dict[str, str]):
    keys = ("aws.access_key_id", "aws.secret_access_key")
    return all(connection[value] for value in keys)
