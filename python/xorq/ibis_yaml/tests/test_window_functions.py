from __future__ import annotations

import pytest

import xorq.api as xo
import xorq.vendor.ibis as ibis
import xorq.vendor.ibis.expr.types as ir
from xorq.ibis_yaml.compiler import YamlExpressionTranslator
from xorq.ibis_yaml.enums import RefEnum, RegistryEnum


def test_window_function_roundtrip(
    compiler: YamlExpressionTranslator, t: ir.Table
) -> None:
    expr = t.select(
        [
            t.c.mean()
            .over(ibis.window(preceding=5, following=0, group_by=t.a))
            .name("mean_c")
        ]
    )

    yaml_dict = compiler.to_yaml(expr)

    reconstructed_expr = compiler.from_yaml(yaml_dict)

    assert expr.equals(reconstructed_expr)


def test_aggregation_window(compiler, t):
    cases = [
        (None, None),
        (5, 0),
        (0, 5),
        (5, 5),
    ]

    for preceding, following in cases:
        expr = t.select(
            [
                t.c.mean()
                .over(
                    ibis.window(preceding=preceding, following=following, group_by=t.a)
                )
                .name("mean_c")
            ]
        )

        yaml_dict = compiler.to_yaml(expr)
        node_ref = yaml_dict["expression"][RefEnum.node_ref]

        expression = yaml_dict["definitions"][RegistryEnum.nodes][node_ref]
        assert expression["op"] == "Project"
        window_func = expression["values"]["mean_c"]
        assert window_func["op"] == "WindowFunction"
        assert window_func["args"][0]["op"] == "Mean"

        if preceding is None:
            assert "start" not in window_func
        else:
            assert window_func["start"] == preceding

        if following is None:
            assert "end" not in window_func
        else:
            assert window_func["end"] == following


def test_row_number_simple_roundtrip(compiler, t):
    expr = t.select([ibis.row_number().name("row_num")])
    yaml_dict = compiler.to_yaml(expr)
    reconstructed_expr = compiler.from_yaml(yaml_dict)
    assert expr.equals(reconstructed_expr)


def test_row_number_window_roundtrip(compiler, t):
    expr = t.select(
        [
            ibis.row_number()
            .over(
                ibis.window(
                    group_by=[t.a, t.b],
                    order_by=[t.c.desc(), t.d],
                    preceding=5,
                    following=0,
                )
            )
            .name("row_num")
        ]
    )
    yaml_dict = compiler.to_yaml(expr)
    reconstructed_expr = compiler.from_yaml(yaml_dict)
    assert expr.equals(reconstructed_expr)


def test_multiple_rank_expressions_roundtrip(compiler, t):
    expr = t.select(
        [
            ibis.row_number().over(ibis.window(group_by=t.a)).name("simple_row_num"),
            ibis.row_number()
            .over(ibis.window(group_by=[t.a, t.b], order_by=t.c.desc()))
            .name("ordered_row_num"),
            t.c.mean()
            .over(ibis.window(preceding=3, following=0, group_by=t.a))
            .name("mean_c"),
        ]
    )
    yaml_dict = compiler.to_yaml(expr)
    reconstructed_expr = compiler.from_yaml(yaml_dict)
    assert expr.equals(reconstructed_expr)


@pytest.mark.parametrize(
    "make_expr",
    [
        pytest.param(lambda t: t.mutate(p=xo.row_number()), id="row_number()"),
        pytest.param(lambda t: t.mutate(p=xo.row_number() + 1), id="row_number() + 1"),
        pytest.param(
            lambda t: t.order_by("a").mutate(p=xo.rank() + 1), id="rank() + 1"
        ),
        pytest.param(
            lambda t: t.mutate(p=xo.row_number().over(order_by="a") + 1),
            id="row_number().over(order_by='a') + 1",
        ),
        pytest.param(
            lambda t: t.mutate(p=xo.row_number().over(group_by="b") + 1),
            id="row_number().over(group_by='b') + 1",
        ),
    ],
)
def test_unordered_window_inside_op_roundtrip(compiler, t, make_expr):
    # gh-2395: an unordered window function nested in another op used to
    # come back from yaml as an unbound Deferred and fail with InputTypeError
    expr = make_expr(t)
    assert expr.equals(compiler.from_yaml(compiler.to_yaml(expr)))
