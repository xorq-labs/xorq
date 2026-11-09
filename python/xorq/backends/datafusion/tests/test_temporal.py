import datetime

import pytest

import xorq.api as xo


@pytest.fixture
def t():
    con = xo.datafusion.connect()
    return con.create_table(
        "t",
        {
            "d": [datetime.date(2024, 1, 1)],
            "ts": [datetime.datetime(2024, 1, 1, 10, 40)],
            "s": ["2024-03-05"],
            "i": [3],
        },
    )


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        pytest.param(lambda t: t.ts.strftime("%Y-%m"), "2024-01", id="strftime"),
        pytest.param(
            lambda t: t.s.as_date("%Y-%m-%d"),
            datetime.date(2024, 3, 5),
            id="string_to_date",
        ),
        pytest.param(
            lambda t: t.s.as_timestamp("%Y-%m-%d"),
            datetime.datetime(2024, 3, 5, tzinfo=datetime.timezone.utc),
            id="string_to_timestamp",
        ),
        pytest.param(
            lambda t: t.d + t.i.as_interval("D"),
            datetime.date(2024, 1, 4),
            id="interval_from_integer",
        ),
        pytest.param(
            lambda t: t.ts.bucket(hours=1),
            datetime.datetime(2024, 1, 1, 10),
            id="bucket",
        ),
        pytest.param(
            lambda t: t.ts.bucket(hours=1, offset=xo.interval(minutes=15)),
            datetime.datetime(2024, 1, 1, 10, 15),
            id="bucket_offset",
        ),
    ],
)
def test_temporal_op_executes(t, make_expr, expected):
    assert make_expr(t).execute().tolist() == [expected]
