import pytest

import xorq as xo


@pytest.mark.parametrize(
    "fun",
    [
        pytest.param(
            lambda e: e.filter(xo._.symbol.isin(("GOOGL", "MSFT"))), id="isin"
        ),
        pytest.param(lambda e: e.filter(xo._.symbol.isnull()), id="isnull"),
        pytest.param(lambda e: e.filter(xo._.bid.isnan()), id="isnan"),
    ],
)
def test_execute(quotes_table, fun):
    expr = fun(quotes_table)
    actual = expr.execute()
    assert actual is not None
