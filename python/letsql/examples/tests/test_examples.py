import pytest

import letsql as ls
from letsql.examples import Example
from letsql.examples.core import whitelist


def test_whitelist():
    assert [name in dir(ls.examples) for name in whitelist]


def test_attributes():
    for name in whitelist:
        example = getattr(ls.examples, name)
        assert isinstance(example, Example)

    with pytest.raises(AttributeError):
        ls.examples.missing
