import letsql as ls
from letsql.backends.let.datafusion import Backend


def test_version():
    assert ls.__version__ == Backend().version


def test_context_name():
    con = ls.connect()
    assert "let.SessionContext" in str(type(con.con))
