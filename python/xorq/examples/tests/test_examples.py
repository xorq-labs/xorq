import pytest

import xorq as xo
from xorq.examples import Example
from xorq.examples.core import whitelist


def test_whitelist():
    assert [name in dir(xo.examples) for name in whitelist]


def test_attributes():
    for name in whitelist:
        example = getattr(xo.examples, name)
        assert isinstance(example, Example)

    with pytest.raises(AttributeError):
        xo.examples.missing
