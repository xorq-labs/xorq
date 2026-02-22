import operator
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest
import toolz

import xorq.api as xo
from xorq.flight.exchanger import make_udxf
from xorq.flight.tests.conftest import (
    do_agg,
    field_name,
    my_udf,
    my_udf_on_expr,
    return_type,
)


def test_flight_expr(con, diamonds, baseline):
    unbound_expr = (
        xo.table(diamonds.schema()).pipe(do_agg).mutate(my_udf_on_expr).order_by("cut")
    )
    expr = xo.expr.relations.flight_expr(
        diamonds,
        unbound_expr,
        inner_name="flight-expr",
        name="remote-expr",
        con=con,
    )
    df = expr.execute()
    pd.testing.assert_frame_equal(
        baseline.sort_values("cut", ignore_index=True),
        df.sort_values("cut", ignore_index=True),
        check_exact=False,
    )


def test_flight_udxf(con, diamonds, baseline):
    input_expr = diamonds.pipe(do_agg)
    process_df = operator.methodcaller("assign", **{field_name: my_udf.fn})
    maybe_schema_in = input_expr.schema()
    maybe_schema_out = xo.schema(input_expr.schema() | {field_name: return_type})
    expr = xo.expr.relations.flight_udxf(
        input_expr,
        process_df=process_df,
        maybe_schema_in=maybe_schema_in,
        maybe_schema_out=maybe_schema_out,
        con=con,
        # operator.methodcaller doesn't have name, so must explicitly pass
        make_udxf_kwargs={"name": my_udf.__name__},
    ).order_by("cut")
    df = expr.execute()
    actual = df.sort_values("cut", ignore_index=True)
    expected = baseline.sort_values("cut", ignore_index=True)
    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_exact=False,
    )


def test_make_udxf_fails():
    def dummy(df: pd.DataFrame):
        return pd.DataFrame({"row_count": [42]})

    with pytest.raises(ValueError):
        make_udxf(
            dummy,
            xo.schema({"dummy": "int64"}),
            pa.schema(
                [
                    ("row_count", pa.int64()),
                ]
            ),
        )

    with pytest.raises(ValueError):
        make_udxf(
            dummy,
            pa.schema(
                [
                    ("dummy", pa.int64()),
                ]
            ),
            xo.schema({"row_count": "int64"}),
        )


def test_flight_serve_unbound_finds_con(parquet_dir):
    batting = xo.deferred_read_parquet(
        parquet_dir.joinpath("batting.parquet"), xo.connect()
    )
    awards_players = xo.deferred_read_parquet(
        parquet_dir.joinpath("awards_players.parquet"),
        xo.connect(),
    )

    awards_players_unbound = xo.table(
        name="awards_players", schema=awards_players.schema()
    )
    predicates = tuple(
        set(batting.columns).intersection(awards_players_unbound.columns)
    )
    joined = batting.select(predicates).join(
        awards_players_unbound.select(predicates), predicates=predicates
    )
    _, do_exchange = xo.expr.relations.flight_serve_unbound(joined)
    actual = do_exchange(awards_players).read_pandas()
    expected = batting.execute()[list(predicates)].merge(
        awards_players.execute()[list(predicates)], on=predicates
    )
    assert not actual.empty
    assert actual.sort_values(list(actual.columns), ignore_index=True).equals(
        expected.sort_values(list(expected.columns), ignore_index=True)
    )


@pytest.mark.parametrize(
    "i,j",
    (
        (0, 2),
        (1, 2),
        (2, 2),
    ),
)
def test_flight_serve_unbound_finds_con_complex(i, j, parquet_dir, tmpdir):
    def do_join(left, right, predicates):
        match con := toolz.excepts(xo.api.XorqError, right._find_backend)():
            case None:
                return left.join(right, predicates=predicates)
            case xo.Backend():
                return left.into_backend(con).join(right, predicates=predicates)
            case _:
                raise ValueError(f"unexpected backend type: {type(con)}")

    name = "batting"
    path = Path(tmpdir).joinpath(f"{name}.parquet")
    predicates = ("playerID", "yearID", "teamID")
    xo.deferred_read_parquet(parquet_dir.joinpath(f"{name}.parquet")).select(
        predicates
    ).distinct().to_parquet(path)

    unbound_batting = xo.table(
        schema=xo.deferred_read_parquet(path).schema(), name=name
    )
    (*battings, to_exchange) = tuple(
        xo.deferred_read_parquet(path, xo.connect()) for _ in range(j + 1)
    )
    (batting0, batting1, *rest) = (*battings[:i], unbound_batting, *battings[i:])
    joined = do_join(batting0, batting1, predicates)
    for other in rest:
        joined = do_join(joined, other, predicates)

    _, do_exchange = xo.expr.relations.flight_serve_unbound(joined)
    actual = do_exchange(to_exchange).read_pandas()
    expected = xo.deferred_read_parquet(path).execute()
    assert not actual.empty
    assert actual.sort_values(list(actual.columns), ignore_index=True).equals(
        expected.sort_values(list(expected.columns), ignore_index=True)
    )
