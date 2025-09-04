import pytest

import xorq.api as xo


@pytest.mark.parametrize(
    "metrics",
    [
        (xo._.int_col.first(),),
        (xo._.bool_col.any(),),
        (xo._.int_col.first(), xo._.bool_col.any()),
    ],
)
def test_first(compiler, alltypes, metrics):
    expr = alltypes.group_by(xo._.tinyint_col).aggregate(
        *metrics,
    )

    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    assert roundtrip_expr.equals(expr)
