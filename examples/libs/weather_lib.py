# examples/libs/weather_lib.py
from datetime import (
    timedelta,
)
from pathlib import Path

import pandas as pd
import pyarrow as pa
import requests
import toolz
from hash_cache.hash_cache import (
    Serder,
    hash_cache,
)

import xorq as xo
from xorq.common.utils.env_utils import (
    EnvConfigable,
)
from xorq.flight.exchanger import make_udxf


env_config = EnvConfigable.subclass_from_kwargs(
    "OPENWEATHER_API_KEY",
    "WEATHER_FEATURES_PORT",
    "WEATHER_API_URL",
).from_env()
OPENWEATHER_KEY = env_config.OPENWEATHER_API_KEY
assert OPENWEATHER_KEY
WEATHER_FEATURES_PORT = int(env_config.WEATHER_FEATURES_PORT)
assert WEATHER_FEATURES_PORT
WEATHER_API_URL = env_config.WEATHER_API_URL
assert WEATHER_API_URL


def extract_dct(data):
    extraction_pairs = (
        ("longitude", ("coord", "lon")),
        ("latitude", ("coord", "lat")),
        ("country", ("sys", "country")),
        ("timezone_offset", ("timezone",)),
        #
        ("weather_main", ("weather", 0, "main")),
        ("weather_description", ("weather", 0, "description")),
        ("weather_icon", ("weather", 0, "icon")),
        ("weather_id", ("weather", 0, "id")),
        #
        ("temp_c", ("main", "temp")),
        ("feels_like_c", ("main", "feels_like")),
        ("temp_min_c", ("main", "temp_min")),
        ("temp_max_c", ("main", "temp_max")),
        ("pressure_hpa", ("main", "pressure")),
        ("humidity_percent", ("main", "humidity")),
        ("sea_level_pressure_hpa", ("main", "sea_level")),
        ("ground_level_pressure_hpa", ("main", "grnd_level")),
        #
        ("wind_direction_deg", ("wind", "deg")),
        ("wind_gust_ms", ("wind", "gust")),
        ("clouds_percent", ("clouds", "all")),
        ("visibility_m", ("visibility",)),
        ("data_timestamp", ("dt",)),
        ("sunset_timestamp", ("sys", "sunset")),
        ("sunrise_timestamp", ("sys", "sunrise")),
        ("city_id", ("id",)),
        ("response_code", ("cod",)),
    )
    return {k: toolz.get_in(v, data) for k, v in extraction_pairs}


@hash_cache(
    Path("./weather-cache"),
    serder=Serder.json_serder(),
    args_kwargs_serder=Serder.args_kwargs_json_serder(),
    ttl=timedelta(seconds=3),
)
def fetch_one_city(*, city: str):
    resp = requests.get(
        WEATHER_API_URL, params={"q": city, "appid": OPENWEATHER_KEY, "units": "metric"}
    )
    resp.raise_for_status()
    data = resp.json()
    return extract_dct(data) | {
        "city": city,
        "timestamp": pd.Timestamp.utcnow().isoformat(),
    }


def get_current_weather_batch(df: pd.DataFrame) -> pd.DataFrame:
    records = [fetch_one_city(city=city) for city in df["city"].values]
    # build DataFrame and ensure nullable Int64 dtypes for wind columns so Arrow always emits a values buffer
    tbl = pd.DataFrame(records).reindex(schema_out.to_pyarrow().names, axis=1)

    # convert any object-type columns that are numeric in schema_out to proper numeric dtypes
    # why do we need this? Arrows Int64 is supposed to be nullable?
    object_cols = tbl.select_dtypes(include=["object"]).columns
    for col in object_cols:
        arrow_type = schema_out.to_pyarrow().field_by_name(col).type
        if pa.types.is_integer(arrow_type):
            tbl[col] = pd.to_numeric(tbl[col], errors="coerce").astype("Int64")
        elif pa.types.is_floating(arrow_type):
            tbl[col] = pd.to_numeric(tbl[col], errors="coerce")

    return tbl


schema_in = xo.schema({"city": "string"})
schema_out = xo.schema(
    {
        "city": "string",
        "timestamp": "string",
        "longitude": "double",
        "latitude": "double",
        "country": "string",
        "timezone_offset": "int64",
        "weather_main": "string",
        "weather_description": "string",
        "weather_icon": "string",
        "weather_id": "int64",
        "temp_c": "double",
        "feels_like_c": "double",
        "temp_min_c": "double",
        "temp_max_c": "double",
        "pressure_hpa": "int64",
        "humidity_percent": "int64",
        "sea_level_pressure_hpa": "int64",
        "ground_level_pressure_hpa": "int64",
        # wind fields: speed and gust as floats, direction as nullable integer
        "wind_direction_deg": "int64",
        "wind_gust_ms": "int64",
        "clouds_percent": "int64",
        "visibility_m": "int64",
        "data_timestamp": "int64",
        "sunrise_timestamp": "int64",
        "sunset_timestamp": "int64",
        "city_id": "int64",
        "response_code": "int64",
    }
)

do_fetch_current_weather_flight_udxf = xo.expr.relations.flight_udxf(
    process_df=get_current_weather_batch,
    maybe_schema_in=schema_in,
    maybe_schema_out=schema_out,
    name="FetchCurrentWeather",
)

do_fetch_current_weather_udxf = make_udxf(
    get_current_weather_batch,
    schema_in,
    schema_out,
)
