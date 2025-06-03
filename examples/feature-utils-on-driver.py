from datetime import timedelta

import pandas as pd

import xorq as xo
import xorq.expr.datatypes as dt
from xorq.common.utils.feature_utils import (
    EVENT_TIMESTAMP,
    Entity,
    Feature,
    FeatureStore,
    FeatureView,
)


driver_path = xo.options.pins.get_path("driver_stats")
key_column = "driver_id"
feature_names = [
    "conv_rate",
    "acc_rate",
    "avg_daily_trips",
]
by = [EVENT_TIMESTAMP, key_column]


def make_split(path=driver_path):
    df = pd.read_parquet(path).sort_values(by)
    dct = {
        feature: (
            df.groupby(key_column)
            .apply(lambda t: t[i :: len(feature_names)], include_groups=False)
            .reset_index(level=key_column)[[key_column, EVENT_TIMESTAMP, feature]]
            .sort_values(by, ignore_index=True)
        )
        for i, feature in enumerate(feature_names, 1)
    }
    return dct


def make_merged(path=driver_path):
    (df, *others) = make_split(path=path).values()
    for other in others:
        df = pd.merge_asof(
            df,
            other,
            on=EVENT_TIMESTAMP,
            by=key_column,
        )
    return df


def make_entity_df():
    driver_ids = [1001, 1003, 1005]
    dts = pd.to_datetime(
        ["2025-04-27 12:13:14", "2025-05-02 12:13:14", "2025-05-04 12:13:14"]
    ).tolist()
    entity_df = pd.DataFrame(
        {
            key_column: sum(
                [[driver_id] * len(dts) for driver_id in driver_ids], start=[]
            ),
            EVENT_TIMESTAMP: dts * len(driver_ids),
        }
    ).assign(**{EVENT_TIMESTAMP: lambda t: t[EVENT_TIMESTAMP].dt.tz_localize("utc")})
    return entity_df


dct = make_split()
entity_df = make_entity_df()
con = xo.connect()


entity = Entity(
    name="driver",
    key_column="driver_id",
    description="the id of the driver",
)
features = [
    Feature(
        name=feature_name,
        dtype=dt.float,
    )
    for feature_name in feature_names
]
feature_views = [
    FeatureView(
        name=feature.name,
        features=(feature,),
        offline_expr=xo.memtable(dct[feature.name]).into_backend(con),
        entities=(entity,),
        timestamp_column=EVENT_TIMESTAMP,
        ttl=timedelta(hours=2, minutes=14),
    )
    for feature in features
]
store = FeatureStore(views={view.name: view for view in feature_views})
references = [f"{view.name}:{view.features[0].name}" for view in feature_views]
expr = store.get_historical_features(entity_df, references)
df = expr.execute().sort_values(by, ignore_index=True)
comparison = pd.merge_asof(df, make_merged(), on=EVENT_TIMESTAMP, by=key_column)
print(df)
print(comparison)
