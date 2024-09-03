import os


def make_s3_credentials_defaults():
    return {
        "aws.access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
        "aws.secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
    }


def make_s3_connection():
    connection = {
        **make_s3_credentials_defaults(),
        "aws.session_token": os.environ.get("AWS_SESSION_TOKEN", ""),
        "aws.allow_http": os.environ.get("AWS_ALLOW_HTTP", "false"),
    }

    if region := os.environ.get("AWS_REGION"):
        connection["aws.region"] = region

    return connection, connection_is_set(connection)


def connection_is_set(connection: dict[str, str]):
    keys = ("aws.access_key_id", "aws.secret_access_key")
    return all(connection[value] for value in keys)
