from __future__ import annotations

import operator
from datetime import datetime, timedelta

import pytest

import xorq.api as xo
from xorq.backends.duckdb import Backend
from xorq.vendor import ibis


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


# regression tests for xorq-labs/xorq#983: the tolerance lowering used to
# reference the left source twice (filter + re-join), which silently broke
# one-shot inputs and could fan out on duplicate left keys. The single-pass
# null-out lowering must keep every left row exactly once.


@pytest.fixture(scope="module")
def sensors_df() -> "pd.DataFrame":
    return pd.DataFrame(
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


@pytest.fixture(scope="module")
def events_df() -> "pd.DataFrame":
    return pd.DataFrame(
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


def test_asof_tolerance_preserves_unmatched_left_rows(
    duckdb_con: Backend,
    sensors_df: "pd.DataFrame",
    events_df: "pd.DataFrame",
) -> None:
    # every left row must survive; out-of-tolerance matches get NULL right cols
    left = duckdb_con.create_table("sensors_tol", sensors_df)
    right = duckdb_con.create_table("events_tol", events_df)
    expr = left.asof_join(
        right, on="event_time", predicates="site", tolerance=timedelta(seconds=1)
    )
    result = expr.execute().sort_values(["site", "humidity"]).reset_index(drop=True)

    assert len(result) == len(sensors_df)
    matched = result.dropna(subset=["event_type"]).set_index("humidity")["event_type"]
    assert matched.to_dict() == {
        0.3: "cloud coverage",
        0.6: "rain start",
        0.7: "rain stop",
    }
    # the two out-of-tolerance rows are present with NULL right columns
    assert set(result.loc[result["event_type"].isna(), "humidity"]) == {0.4, 0.5}


def test_asof_tolerance_one_shot_reader(
    sensors_df: "pd.DataFrame", events_df: "pd.DataFrame"
) -> None:
    # the actual issue #983: into_backend registers one-shot RecordBatchReaders;
    # a double-scan lowering reads 0 rows on the second pass.
    con = xo.duckdb.connect()
    sensors = ibis.memtable(sensors_df)
    events = ibis.memtable(events_df)
    expr = (
        sensors.into_backend(con, "sensors_rbr")
        .asof_join(
            events.into_backend(con, "events_rbr"),
            on="event_time",
            predicates="site",
            tolerance=timedelta(seconds=1),
        )
        .order_by(["site", "humidity"])
    )
    result = expr.execute()

    assert len(result) == len(sensors_df)
    matched = result.dropna(subset=["event_type"]).set_index("humidity")["event_type"]
    assert matched.to_dict() == {
        0.3: "cloud coverage",
        0.6: "rain start",
        0.7: "rain stop",
    }


def test_asof_tolerance_boundary_inclusive(duckdb_con: Backend) -> None:
    # a match exactly at the tolerance boundary must be kept (inclusive window)
    left = duckdb_con.create_table(
        "boundary_left",
        pd.DataFrame(
            {"key": [1], "time": [datetime(2024, 1, 1, 0, 0, 10)], "lv": [1.0]}
        ),
    )
    right = duckdb_con.create_table(
        "boundary_right",
        pd.DataFrame(
            {"key": [1], "time": [datetime(2024, 1, 1, 0, 0, 8)], "rv": [2.0]}
        ),
    )
    expr = left.asof_join(
        right, on="time", predicates="key", tolerance=timedelta(seconds=2)
    )
    result = expr.execute()
    assert result["rv"].notna().all()
    assert result["rv"].iloc[0] == 2.0


def test_asof_tolerance_duplicate_left_keys(duckdb_con: Backend) -> None:
    # duplicate left `on` values within a predicate group: the old re-join on
    # left_on == right_on could fan out / mismatch; single pass keeps 1 row each
    left = duckdb_con.create_table(
        "dup_left",
        pd.DataFrame(
            {
                "key": [1, 1, 1],
                "time": [datetime(2024, 1, 1, 0, 0, 5)] * 3,
                "lv": [10.0, 20.0, 30.0],
            }
        ),
    )
    right = duckdb_con.create_table(
        "dup_right",
        pd.DataFrame(
            {"key": [1], "time": [datetime(2024, 1, 1, 0, 0, 5)], "rv": [99.0]}
        ),
    )
    expr = left.asof_join(
        right, on="time", predicates="key", tolerance=timedelta(seconds=1)
    )
    result = expr.execute()
    # no fan-out: exactly the 3 left rows, each matched to the single right row
    assert len(result) == 3
    assert (result["rv"] == 99.0).all()


def test_asof_tolerance_multi_table_left_chain(duckdb_con: Backend) -> None:
    # the left side is itself a multi-link join chain: the null-out projection
    # discriminates columns by field origin, so columns from the earlier link
    # (bv) must stay untouched while only the asof right columns (cv) get nulled
    a = duckdb_con.create_table(
        "chain_a",
        pd.DataFrame(
            {
                "key": [1, 1],
                "time": [
                    datetime(2024, 1, 1, 0, 0, 10),
                    datetime(2024, 1, 1, 0, 1, 40),
                ],
                "av": [1.0, 2.0],
            }
        ),
    )
    b = duckdb_con.create_table(
        "chain_b",
        pd.DataFrame({"key": [1], "bv": [7.0]}),
    )
    c = duckdb_con.create_table(
        "chain_c",
        pd.DataFrame(
            {"key": [1], "time": [datetime(2024, 1, 1, 0, 0, 9)], "cv": [99.0]}
        ),
    )
    left = a.join(b, "key")
    expr = left.asof_join(
        c, on="time", predicates="key", tolerance=timedelta(seconds=2)
    ).order_by("av")
    result = expr.execute()

    # both left-chain rows preserved; earlier-link column bv untouched
    assert len(result) == 2
    assert (result["bv"] == 7.0).all()
    # in-tolerance row keeps cv, out-of-tolerance row is nulled
    assert result["cv"].tolist()[0] == 99.0
    assert pd.isna(result["cv"].tolist()[1])
