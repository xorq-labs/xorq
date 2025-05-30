# examples/libs/weather_lib.py
import os
from datetime import (
    timedelta,
)
from pathlib import Path

import pandas as pd
import requests
from hash_cache.hash_cache import (
    Serder,
    hash_cache,
)

import xorq as xo
import xorq.expr.datatypes as dt
from xorq.flight.exchanger import make_udxf


OPENWEATHER_KEY = os.environ["OPENWEATHER_API_KEY"]
API_URL = "https://api.openweathermap.org/data/2.5/weather"


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

    weather_info = data["weather"][0] if data["weather"] else {}
    main_data = data["main"]
    wind_data = data.get("wind", {})
    clouds_data = data.get("clouds", {})
    sys_data = data["sys"]
    coord_data = data["coord"]

    # Ensure wind_gust_ms is always a float, never None
    wind_gust = wind_data.get("gust")
    if wind_gust is None:
        wind_gust = 0.0
    else:
        wind_gust = float(wind_gust)

    return {
        "city": city,
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "longitude": coord_data["lon"],
        "latitude": coord_data["lat"],
        "country": sys_data.get("country"),
        "timezone_offset": data.get("timezone"),
        "weather_main": weather_info.get("main"),
        "weather_description": weather_info.get("description"),
        "weather_icon": weather_info.get("icon"),
        "weather_id": weather_info.get("id"),
        "temp_c": main_data["temp"],
        "feels_like_c": main_data["feels_like"],
        "temp_min_c": main_data["temp_min"],
        "temp_max_c": main_data["temp_max"],
        "pressure_hpa": main_data["pressure"],
        "humidity_percent": main_data["humidity"],
        "sea_level_pressure_hpa": main_data.get("sea_level"),
        "ground_level_pressure_hpa": main_data.get("grnd_level"),
        "wind_speed_ms": wind_data.get("speed"),
        "wind_direction_deg": wind_data.get("deg"),
        "wind_gust_ms": wind_gust,  # Use the processed value
        "clouds_percent": clouds_data.get("all"),
        "visibility_m": data.get("visibility"),
        "data_timestamp": data["dt"],
        "sunrise_timestamp": sys_data.get("sunrise"),
        "sunset_timestamp": sys_data.get("sunset"),
        "city_id": data["id"],
        "response_code": data["cod"],
    }


def get_current_weather_batch(df: pd.DataFrame) -> pd.DataFrame:
    records = [fetch_one_city(city=row["city"]) for _, row in df.iterrows()]
    return pd.DataFrame(records)


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
