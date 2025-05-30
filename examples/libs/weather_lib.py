# examples/libs/weather_lib.py
from datetime import (
    timedelta,
)
from pathlib import Path

import pandas as pd
import requests
import toolz
from hash_cache.hash_cache import (
    Serder,
    hash_cache,
)

import xorq as xo
import xorq.expr.datatypes as dt
from xorq.common.utils.env_utils import (
    EnvConfigable,
)
from xorq.flight.exchanger import make_udxf


env_config = EnvConfigable.subclass_from_kwargs("OPENWEATHER_API_KEY").from_env()
OPENWEATHER_KEY = env_config.OPENWEATHER_API_KEY
assert OPENWEATHER_KEY
API_URL = "https://api.openweathermap.org/data/2.5/weather"


def extract_dct(data):
    pairs = (
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
        #
        ("pressure_hpa", ("main", "pressure")),
        ("humidity_percent", ("main", "humidity")),
        ("sea_level_pressure_hpa", ("main", "sea_level")),
        ("ground_level_pressure_hpa", ("main", "grnd_level")),
        #
        ("wind_speed_ms", ("wind", "speed")),
        ("wind_direction_deg", ("wind", "deg")),
        ("clouds_percent", ("clouds", "all")),
        ("visibility_m", ("visibility",)),
        ("data_timestamp", ("dt",)),
        ("sunset_timestamp", ("sys", "sunset")),
        ("sunrise_timestamp", ("sys", "sunrise")),
        ("city_id", ("id",)),
        ("response_code", ("cod",)),
    )
    return {k: toolz.get_in(v, data) for k, v in pairs} | {
        "wind_gust_ms": float(toolz.get_in(("wind", "gust"), data, default=0)),
        # "wind_gust_ms": 0.0,
    }


@hash_cache(
    Path("./weather-cache"),
    serder=Serder.json_serder(),
    args_kwargs_serder=Serder.args_kwargs_json_serder(),
    ttl=timedelta(seconds=3),
)
def fetch_one_city(*, city: str):
    resp = requests.get(
        API_URL, params={"q": city, "appid": OPENWEATHER_KEY, "units": "metric"}
    )
    resp.raise_for_status()
    data = resp.json()
    return extract_dct(data) | {
        "city": city,
        "timestamp": pd.Timestamp.utcnow().isoformat(),
    }


def get_current_weather_batch(df: pd.DataFrame) -> pd.DataFrame:
    records = [fetch_one_city(city=city) for city in df["city"].values]
    return pd.DataFrame(records).reindex(schema_out.names, axis=1)


schema_in = xo.schema({"city": "string"}).to_pyarrow()
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
        "wind_speed_ms": "double",
        "wind_direction_deg": "int64",
        "wind_gust_ms": dt.float64(nullable=True),
        "clouds_percent": "int64",
        "visibility_m": "int64",
        "data_timestamp": "int64",
        "sunrise_timestamp": "int64",
        "sunset_timestamp": "int64",
        "city_id": "int64",
        "response_code": "int64",
    }
).to_pyarrow()
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
