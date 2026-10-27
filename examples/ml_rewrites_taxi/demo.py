"""Native ML rewrites on NYC taxi trips: score, aggregate and fit per segment.

Run: python examples/ml_rewrites_taxi/demo.py [--months N] [--data-dir DIR]
Needs the ml-spike dependency group (orbital) for inlining.
"""

import argparse
import urllib.request
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xorq.api as xo
from xorq.expr.ml.native.api import explain, fit, fit_by, predict


URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-{:02d}.parquet"
DATA_DIR = Path("~/.cache/xorq-taxi").expanduser()

NUMERIC = ("trip_distance", "passenger_count", "hour", "dow")
CATEGORICAL = ("payment_type", "RatecodeID")
FEATURES = NUMERIC + CATEGORICAL
TARGET = "fare_amount"
JFK = 132


def fetch(months, data_dir=DATA_DIR):
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = tuple(data_dir / Path(URL.format(month)).name for month in months)
    for month, path in zip(months, paths):
        if not path.exists():
            # download beside the target, so an interrupted run leaves no
            # partial file at the cached path
            partial = path.with_suffix(".part")
            try:
                urllib.request.urlretrieve(URL.format(month), partial)
                partial.rename(path)
            finally:
                partial.unlink(missing_ok=True)
    return paths


def trips(con, months, data_dir=DATA_DIR):
    paths = fetch(months, data_dir)
    raw = con.read_parquet([str(path) for path in paths], table_name="trips")
    return raw.filter(
        raw.fare_amount.between(0, 500),
        raw.trip_distance.between(0, 100),
        raw.passenger_count.notnull(),
        raw.RatecodeID.notnull(),
    ).mutate(
        hour=raw.tpep_pickup_datetime.hour().cast("float64"),
        dow=raw.tpep_pickup_datetime.day_of_week.index().cast("float64"),
        payment_type=raw.payment_type.cast("string"),
        RatecodeID=raw.RatecodeID.cast("int64").cast("string"),
    )


def make_pipeline(model):
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), list(NUMERIC)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def training(taxi):
    """The first week of January, the rows every model is fitted on."""
    return taxi.filter(taxi.tpep_pickup_datetime < "2024-01-08")


def scored(taxi, model):
    fitted = fit(training(taxi), FEATURES, TARGET, make_pipeline(model))
    return taxi.mutate(predicted=predict(fitted, taxi))


def q_score(taxi, model):
    """Predicted fares for JFK pickups in one week."""
    expr = scored(taxi, model)
    return expr.filter(
        expr.PULocationID == JFK,
        expr.tpep_pickup_datetime.between("2024-01-15", "2024-01-22"),
    ).select("tpep_pickup_datetime", "trip_distance", TARGET, "predicted")


def q_agg(taxi, model):
    """Mean predicted fare per pickup zone and hour."""
    expr = scored(taxi, model)
    return expr.group_by("PULocationID", "hour").agg(
        trips=expr.count(), mean_predicted=expr.predicted.mean()
    )


def q_fit_segment(taxi, model):
    """One model per payment type."""
    return fit_by(
        training(taxi), "payment_type", FEATURES, TARGET, make_pipeline(model)
    )


QUERIES = {"score": q_score, "agg": q_agg, "fit-segment": q_fit_segment}


def show(name, expr):
    with_rewrites = explain(expr)
    xo.options.ml.rewrite = False
    try:
        without = explain(expr)
    finally:
        xo.options.ml.rewrite = True
    print(f"== {name}: without rewrites ==\n{without}\n")
    print(f"== {name}: with rewrites ==\n{with_rewrites}\n")
    print(expr.execute().head(), "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    taxi = trips(xo.duckdb.connect(), range(1, args.months + 1), args.data_dir)
    for name, query in QUERIES.items():
        show(name, query(taxi, Ridge()))


if __name__ == "__main__":
    main()
