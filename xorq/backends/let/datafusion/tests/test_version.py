import xorq as xo
from xorq.backends.let.datafusion import Backend


def test_version():
    assert xo.__version__ == Backend().version


def test_context_name():
    con = xo.connect()
    assert "let.SessionContext" in str(type(con.con))
