import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from feast.driver_test_data import create_driver_hourly_stats_df

import xorq.api as xo
from xorq.common.utils.feast_replication_utils import (
    Store,
)


def freshen_driver_stats(
    store, end_date=None, delta=None, driver_entities=(1001, 1002, 1003, 1004, 1005)
):
    end_date = end_date or datetime.now().replace(microsecond=0, second=0, minute=0)
    delta = delta or timedelta(days=15)
    driver_df = create_driver_hourly_stats_df(
        list(driver_entities),
        end_date - delta,
        end_date,
    )
    parquet_path = store.path.joinpath("data", "driver_stats.parquet")
    driver_df.to_parquet(parquet_path)


def make_store(fixture_dir, tmpdir):
    import tarfile

    tgz_path = fixture_dir.joinpath("feature_repo.tgz")
    with tarfile.TarFile.gzopen(tgz_path) as tfh:
        tfh.extractall(tmpdir)
    store = Store(tmpdir)
    freshen_driver_stats(store)
    return store


def make_store_applied(fixture_dir, tmpdir):
    store_applied = make_store(fixture_dir, tmpdir)
    # chdir doesn't matter inside pytest?
    # monkeypatch.syspath_prepend is only function scope
    old_sys_path, sys.path = sys.path, [str(tmpdir)] + sys.path
    store_applied.apply()
    sys.path = old_sys_path
    return store_applied


@pytest.fixture(scope="function")
def fresh_store(fixture_dir, tmpdir):
    fresh_store = make_store(fixture_dir, tmpdir)
    return fresh_store


@pytest.fixture(scope="session")
def store_applied(fixture_dir, tmpdir_factory):
    tmpdir = Path(tmpdir_factory.mktemp("tmp"))
    store_applied = make_store_applied(fixture_dir, tmpdir)
    return store_applied


@pytest.fixture(scope="session")
def store_applied_materialized(fixture_dir, tmpdir_factory):
    tmpdir = Path(tmpdir_factory.mktemp("tmp"))
    store_applied_materialized = make_store_applied(fixture_dir, tmpdir)
    store_applied_materialized.store.materialize_incremental(end_date=datetime.now())
    return store_applied_materialized


@pytest.fixture(scope="session")
def entity_df():
    entity_df = pd.DataFrame.from_dict(
        {
            # entity's join key -> entity values
            "driver_id": [1001, 1002, 1003],
            # "event_timestamp" (reserved key) -> timestamps
            "event_timestamp": [
                datetime(2021, 4, 12, 10, 59, 42),
                datetime(2021, 4, 12, 8, 12, 10),
                datetime(2021, 4, 12, 16, 40, 26),
            ],
            # (optional) label name -> label values. Feast does not process these
            "label_driver_reported_satisfaction": [1, 5, 3],
            # values we're using for an on-demand transformation
            "val_to_add": [1, 2, 3],
            "val_to_add_2": [10, 20, 30],
        }
    )
    return entity_df


@pytest.fixture(scope="session")
def historical_features():
    features = (
        "driver_hourly_stats:conv_rate",
        "driver_hourly_stats:acc_rate",
        "driver_hourly_stats:avg_daily_trips",
        "transformed_conv_rate:conv_rate_plus_val1",
        "transformed_conv_rate:conv_rate_plus_val2",
    )
    return features


@pytest.fixture(scope="session")
def entity_rows():
    entity_rows = [
        # {join_key: entity_value}
        {
            "driver_id": 1001,
            "val_to_add": 1000,
            "val_to_add_2": 2000,
        },
        {
            "driver_id": 1002,
            "val_to_add": 1001,
            "val_to_add_2": 2002,
        },
    ]
    return entity_rows


def get_online_features(store_applied, source):
    if source == "feature_service":
        features_to_fetch = store_applied.store.get_feature_service(
            "driver_activity_v1"
        )
    elif source == "push":
        features_to_fetch = store_applied.store.get_feature_service(
            "driver_activity_v3"
        )
    else:
        features_to_fetch = [
            "driver_hourly_stats:acc_rate",
            "transformed_conv_rate:conv_rate_plus_val1",
            "transformed_conv_rate:conv_rate_plus_val2",
        ]
    return features_to_fetch


def test_unapplied_failures0(fresh_store, entity_df, historical_features):
    with pytest.raises(
        Exception, match="Feature view driver_hourly_stats does not exist in project"
    ):
        fresh_store.get_historical_features(entity_df, historical_features).to_df()


@pytest.mark.parametrize("source", ("feature_service", "push"))
def test_unapplied_failures1(fresh_store, source):
    with pytest.raises(Exception, match="Feature service .* does not exist in project"):
        get_online_features(fresh_store, source)


@pytest.mark.parametrize("source", ("feature_service", "push"))
def test_unmaterialized_failure(store_applied, entity_rows, source):
    features_to_fetch = get_online_features(store_applied, source)
    actual = store_applied.get_online_features(features_to_fetch, entity_rows)
    assert all(v == [None, None] for k, v in actual.items() if k != "driver_id")


def df_copy(df):
    return pd.DataFrame(df.values, index=df.index, columns=df.columns)


def test_get_historical_features(store_applied, entity_df, historical_features):
    entity_expr = xo.pandas.connect({"t": entity_df}).table("t")
    actual = store_applied.get_historical_features(
        entity_expr, historical_features
    ).execute()
    expected = store_applied.store.get_historical_features(
        entity_df, list(historical_features)
    ).to_df()
    # FIXME: make reindex_like unnecessary
    assert not actual.empty and df_copy(actual.reindex_like(expected)).equals(
        df_copy(expected)
    )


@pytest.mark.parametrize("source", ("feature_service", "push", None))
def test_get_online_features(store_applied_materialized, entity_rows, source):
    features_to_fetch = get_online_features(store_applied_materialized, source)
    actual = store_applied_materialized.get_online_features(
        features_to_fetch, entity_rows
    )
    expected = store_applied_materialized.store.get_online_features(
        features=features_to_fetch,
        entity_rows=entity_rows,
    ).to_dict()
    assert actual == expected
    assert not any(v == [None, None] for k, v in actual.items() if k != "driver_id")
