import pandas as pd
import pytest

import xorq.api as xo
from xorq.api import _
from xorq.ibis_yaml.compiler import YamlExpressionTranslator, build_expr


@pytest.mark.benchmark
def test_baseball_stats_compilation(builds_dir):
    pg = xo.postgres.connect_env()

    batting_old = (
        pg.table("batting")
        .filter(_.yearID < 2000)
        .filter(_.yearID >= 1990)
        .mutate(batting_avg=_.H / _.AB, era=_.yearID.cast("int32"))
    )

    batting_recent = (
        pg.table("batting")
        .filter(_.yearID >= 2000)
        .mutate(batting_avg=_.H / _.AB, era=_.yearID.cast("int32"))
    )

    all_batting = batting_old.union(batting_recent)

    league_averages = all_batting.group_by(["yearID", "lgID"]).aggregate(
        league_avg=_.batting_avg.mean(), league_HR_avg=_.HR.mean()
    )

    normalized_stats = all_batting.join(league_averages, ["yearID", "lgID"]).mutate(
        avg_vs_league=_.batting_avg - _.league_avg, HR_vs_league=_.HR - _.league_HR_avg
    )

    prev_year_stats = (
        normalized_stats.order_by(["playerID", "yearID"])
        .group_by("playerID")
        .mutate(
            prev_year_avg=_.batting_avg.lag(1).over(order_by=_.yearID),
            prev_year_HR=_.HR.lag(1).over(order_by=_.yearID),
            three_year_avg=_.batting_avg.mean().over(
                order_by=_.yearID, window=xo.window(preceding=2, following=0)
            ),
        )
    )

    build_path = build_expr(prev_year_stats, builds_dir=builds_dir)
    expr_hash = build_path.name
    assert isinstance(expr_hash, str)
    assert len(expr_hash) > 0


@pytest.mark.benchmark
def test_to_yaml_long_chain(benchmark):
    t = xo.table({"a": "int64"}, name="t")
    for i in range(40):
        t = t.mutate(**{f"m{i}": t.a + i}) if i % 2 == 0 else t.filter(t.a > -i)
    benchmark(YamlExpressionTranslator.to_yaml, t)


@pytest.mark.benchmark
def test_to_yaml_shared_remote_tables(benchmark):
    # n levels of unioning two RemoteTables over one expression share their
    # subtrees; translation must stay linear in the distinct nodes, not 2**n
    con, left, right = xo.connect(), xo.connect(), xo.connect()
    expr = con.create_table("t", pd.DataFrame({"a": [1, 2]}))
    for level in range(12):  # noqa: B007 - the level number is not used
        expr = expr.into_backend(left).union(expr.into_backend(right))
    benchmark(YamlExpressionTranslator.to_yaml, expr)
