import operator
from datetime import datetime, timedelta

import pytest

import xorq.api as xo
from xorq.vendor import ibis
from xorq.vendor.ibis.util import gen_name


try:
    from duckdb import InvalidInputException as DuckDBInvalidInputException
except ImportError:
    DuckDBInvalidInputException = None

pd = pytest.importorskip("pandas")
tm = pytest.importorskip("pandas.testing")


@pytest.fixture(scope="module")
def time_keyed_df1():
    return pd.DataFrame(
        {
            "time": pd.Series(
                pd.date_range(start="2017-01-02 01:02:03.234", periods=6)
            ),
            "key": [1, 2, 3, 1, 2, 3],
            "value": [1.2, 1.4, 2.0, 4.0, 8.0, 16.0],
        }
    )


@pytest.fixture(scope="module")
def time_keyed_df2():
    return pd.DataFrame(
        {
            "time": pd.Series(
                pd.date_range(start="2017-01-02 01:02:03.234", freq="3D", periods=3)
            ),
            "key": [1, 2, 3],
            "other_value": [1.1, 1.2, 2.2],
        }
    )


@pytest.fixture(scope="module")
def time_keyed_left(time_keyed_df1):
    return ibis.memtable(time_keyed_df1)


@pytest.fixture(scope="module")
def time_keyed_right(time_keyed_df2):
    return ibis.memtable(time_keyed_df2)


@pytest.mark.parametrize(
    ("direction", "op"), [("backward", operator.ge), ("forward", operator.le)]
)
@pytest.mark.parametrize("right_column_key", [None, "on_right_time"])
@pytest.mark.xfail_version(
    duckdb=["duckdb>=0.10.2,<1.1.1"], raises=DuckDBInvalidInputException
)
def test_keyed_asof_join_with_tolerance(
    duckdb_con,
    time_keyed_left,
    time_keyed_right,
    time_keyed_df1,
    time_keyed_df2,
    direction,
    op,
    right_column_key,
):
    left_key = right_key = "time"
    if right_column_key:
        right_key = right_column_key
        time_keyed_right = time_keyed_right.rename({right_column_key: "time"})
        assert left_key != right_key

    on = op(time_keyed_left[left_key], time_keyed_right[right_key])
    expr = time_keyed_left.asof_join(
        time_keyed_right, on, "key", tolerance=ibis.interval(days=2)
    )

    result = duckdb_con.execute(expr)
    expected = pd.merge_asof(
        time_keyed_df1,
        time_keyed_df2,
        on="time",
        by="key",
        tolerance=pd.Timedelta("2D"),
        direction=direction,
    )

    result = result.sort_values(["key", "time"]).reset_index(drop=True)
    expected = expected.sort_values(["key", "time"]).reset_index(drop=True)

    tm.assert_frame_equal(
        # drop `time` from comparison to avoid issues with different time resolution
        result[expected.columns].drop(["time"], axis=1),
        expected.drop(["time"], axis=1),
    )

    # check that time is equal in value, if not dtype
    tm.assert_series_equal(result["time"], expected["time"], check_dtype=False)


# https://github.com/xorq-labs/xorq/issues/983
def test_asof_join_tolerance_with_into_backend(duckdb_con):
    sensors_df = pd.DataFrame(
        {
            "site": ["a", "b", "a", "b", "a"],
            "humidity": [0.3, 0.4, 0.5, 0.6, 0.7],
            "event_time": [
                datetime(2024, 11, 16, 12, 0, 15, 500000),
                datetime(2024, 11, 16, 12, 0, 15, 700000),
                datetime(2024, 11, 17, 18, 12, 14, 950000),
                datetime(2024, 11, 17, 18, 12, 15, 120000),
                datetime(2024, 11, 18, 18, 12, 15, 100000),
            ],
        }
    )
    events_df = pd.DataFrame(
        {
            "site": ["a", "b", "a"],
            "event_type": ["cloud coverage", "rain start", "rain stop"],
            "event_time": [
                datetime(2024, 11, 16, 12, 0, 15, 400000),
                datetime(2024, 11, 17, 18, 12, 15, 100000),
                datetime(2024, 11, 18, 18, 12, 15, 100000),
            ],
        }
    )
    tolerance = timedelta(seconds=1)

    sensors = duckdb_con.create_table(gen_name("sensors"), sensors_df, overwrite=True)
    events = duckdb_con.create_table(gen_name("events"), events_df, overwrite=True)

    expected = (
        sensors.asof_join(
            events,
            on="event_time",
            predicates="site",
            tolerance=tolerance,
        )
        .drop("event_time_right")
        .order_by("event_time")
        .execute()
    )

    result = (
        xo.memtable(sensors_df)
        .into_backend(duckdb_con)
        .asof_join(
            xo.memtable(events_df).into_backend(duckdb_con),
            on="event_time",
            predicates="site",
            tolerance=tolerance,
        )
        .drop("event_time_right")
        .order_by("event_time")
        .execute()
    )

    assert not result.empty
    tm.assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )
