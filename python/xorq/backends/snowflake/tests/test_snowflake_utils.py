from xorq.common.utils.snowflake_utils import make_connection_defaults


def test_connection_defaults_follow_the_environment(monkeypatch):
    def defaults_for(account):
        monkeypatch.setenv("SNOWFLAKE_ACCOUNT", account)
        monkeypatch.setenv("SNOWFLAKE_ROLE", f"{account}_role")
        monkeypatch.setenv("SNOWFLAKE_WAREHOUSE", f"{account}_wh")
        return make_connection_defaults()

    accounts = ("first", "second")
    assert [defaults_for(account) for account in accounts] == [
        {"account": account, "role": f"{account}_role", "warehouse": f"{account}_wh"}
        for account in accounts
    ]
