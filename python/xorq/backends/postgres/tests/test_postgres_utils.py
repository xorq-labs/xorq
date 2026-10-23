from xorq.common.utils.postgres_utils import make_connection_defaults


def test_connection_defaults_follow_the_environment(monkeypatch):
    def defaults_for(host):
        monkeypatch.setenv("POSTGRES_HOST", host)
        monkeypatch.setenv("POSTGRES_PORT", "5432")
        monkeypatch.setenv("POSTGRES_DATABASE", f"{host}_db")
        return make_connection_defaults()

    hosts = ("first", "second")
    assert [defaults_for(host) for host in hosts] == [
        {"host": host, "port": "5432", "database": f"{host}_db"} for host in hosts
    ]
