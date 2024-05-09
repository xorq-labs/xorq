import letsql as ls
from letsql.backends.datafusion import Backend


def test_version():
    assert ls.__version__ == Backend().version
