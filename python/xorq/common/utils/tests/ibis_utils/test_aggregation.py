import pytest
from pytest import param

from xorq.common.utils.ibis_utils import from_ibis


ibis = pytest.importorskip("ibis")


@pytest.mark.parametrize(
    "metrics",
    [
        (ibis._.int_col.first(),),
        (ibis._.bool_col.any(),),
        (ibis._.int_col.first(), ibis._.bool_col.any()),
    ],
)
def test_first(ibis_alltypes, metrics):
    expr = ibis_alltypes.group_by(ibis._.tinyint_col).aggregate(
        *metrics,
    )

    xo_expr = from_ibis(expr)
    assert xo_expr is not None
    assert not xo_expr.execute().empty


@pytest.mark.parametrize(
    "result_fn",
    [
        param(
            lambda t, where: t.bool_col.count(where=where),
            id="count",
        ),
        param(
            lambda t, where: t.bool_col.nunique(where=where),
            id="nunique",
        ),
        param(
            lambda t, where: t.bool_col.any(where=where),
            id="any",
        ),
        param(
            lambda t, where: t.bool_col.notany(where=where),
            id="notany",
        ),
        param(
            lambda t, where: -t.bool_col.any(where=where),
            id="any_negate",
        ),
        param(
            lambda t, where: t.bool_col.all(where=where),
            id="all",
        ),
        param(
            lambda t, where: t.bool_col.notall(where=where),
            id="notall",
        ),
        param(
            lambda t, where: -t.bool_col.all(where=where),
            id="all_negate",
        ),
        param(
            lambda t, where: t.double_col.sum(where=where),
            id="sum",
        ),
        param(
            lambda t, where: (t.int_col > 0).sum(where=where),
            id="bool_sum",
        ),
        param(
            lambda t, where: t.double_col.mean(where=where),
            id="mean",
        ),
        param(
            lambda t, where: t.double_col.min(where=where),
            id="min",
        ),
        param(
            lambda t, where: t.double_col.max(where=where),
            id="max",
        ),
        param(
            lambda t, where: t.double_col.std(how="sample", where=where),
            id="std",
        ),
        param(
            lambda t, where: t.double_col.var(how="sample", where=where),
            id="var",
        ),
        param(
            lambda t, where: t.double_col.std(how="pop", where=where),
            id="std_pop",
        ),
        param(
            lambda t, where: t.double_col.var(how="pop", where=where),
            id="var_pop",
        ),
        param(
            lambda t, where: t.string_col.nunique(where=where),
            id="string_nunique",
        ),
        param(
            lambda t, where: t.double_col.first(where=where),
            id="first",
        ),
        param(
            lambda t, where: t.double_col.last(where=where),
            id="last",
        ),
        param(
            lambda t, where: t.bigint_col.bit_and(where=where),
            id="bit_and",
        ),
        param(
            lambda t, where: t.bigint_col.bit_or(where=where),
            id="bit_or",
        ),
        param(
            lambda t, where: t.bigint_col.bit_xor(where=where),
            id="bit_xor",
        ),
        param(
            lambda t, where: t.count(where=where),
            id="count_star",
        ),
    ],
)
@pytest.mark.parametrize(
    "ibis_cond",
    [
        param(lambda _: None, id="no_cond"),
        param(
            lambda t: t.string_col.isin(["1", "7"]),
            id="is_in",
        ),
        param(
            lambda _: ibis._.string_col.isin(["1", "7"]),
            id="is_in_deferred",
        ),
    ],
)
def test_reduction_ops(
    ibis_alltypes,
    result_fn,
    ibis_cond,
):
    expr = ibis_alltypes.agg(tmp=result_fn(ibis_alltypes, ibis_cond(ibis_alltypes))).tmp

    xorq_expr = from_ibis(expr)
    assert not xorq_expr.execute().empty


def test_approx_median(ibis_alltypes):
    expr = ibis_alltypes.double_col.approx_median()

    xorq_expr = from_ibis(expr)
    assert xorq_expr.execute()


def test_median(ibis_alltypes):
    expr = ibis_alltypes.double_col.median()

    xorq_expr = from_ibis(expr)
    assert xorq_expr.execute()
