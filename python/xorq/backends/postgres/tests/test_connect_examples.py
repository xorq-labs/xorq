import xorq.api as xo
from xorq.common.utils.postgres_utils import postgres_config


def test_connect_examples_follows_env_overrides(monkeypatch):
    for field in ("host", "user", "password", "database"):
        monkeypatch.setenv(
            f"XORQ_EXAMPLES_POSTGRES_{field.upper()}",
            postgres_config[f"POSTGRES_{field.upper()}"],
        )
    con = xo.postgres.connect_examples(port=postgres_config["POSTGRES_PORT"])
    with con.raw_sql("SELECT current_database()") as cursor:
        assert cursor.fetchone() == (postgres_config["POSTGRES_DATABASE"],)
