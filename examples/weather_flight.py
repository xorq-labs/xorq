import argparse
# import logging
from datetime import datetime, timedelta

import pandas as pd
import toolz

import xorq as xo
import xorq.expr.datatypes as dt
from xorq.common.utils.feature_utils import (
    Entity,
    Feature,
    FeatureStore,
    FeatureView,
)
from xorq.common.utils.import_utils import import_python
from xorq.common.utils.logging_utils import get_logger
from xorq.flight import Backend as FlightBackend
from xorq.flight import FlightServer, FlightUrl
from xorq.common.utils.feature_utils import (
    Entity,
    Feature,
    FeatureStore,
    FeatureView,
)


# from xorq.flight.client import FlightClient


logging = get_logger()

# from xorq.flight.client import FlightClient


logging = get_logger()

logging_format = "[%(asctime)s] %(levelname)s %(message)s"

weather_lib = import_python("examples/libs/weather_lib.py")
do_fetch_current_weather_udxf = weather_lib.do_fetch_current_weather_udxf
do_fetch_current_weather_flight_udxf = weather_lib.do_fetch_current_weather_flight_udxf

WEATHER_FEATURES_PORT = weather_lib.WEATHER_FEATURES_PORT
TIMESTAMP_COLUMN = "timestamp"

# Database files
TABLE_BATCH = "weather_history"
CITIES = ["London", "Tokyo", "New York", "Lahore"]


def setup_store() -> FeatureStore:
    logging.info("Setting up FeatureStore")

    # 1. Entity
    city = Entity("city", key_column="city", description="City identifier")

    # 2. Offline source (batch history)
    # this is not being used
    offline_con = xo.duckdb.connect()
    offline_con.raw_sql("""
        INSTALL ducklake;
        INSTALL sqlite;
        ATTACH 'ducklake:sqlite:metadata.sqlite' AS my_ducklake (DATA_PATH 'file_path/');
        USE my_ducklake;
        """)
    ensure_table(offline_con, TABLE_BATCH)

    # 3. Flight backend for online features
    fb = FlightBackend()
    fb.do_connect(host="localhost", port=WEATHER_FEATURES_PORT)

    # 4. Build offline expression for features
    # win6_online = xo.window(
    #     group_by=[city.key_column], order_by="timestamp", preceding=5, following=0
    # )
    offline_table = offline_con.table(TABLE_BATCH)
    win6_offline = xo.window(
        group_by=[city.key_column], order_by="timestamp", preceding=5, following=0
    )

    # Offline expression that computes the feature from historical data
    offline_expr = offline_table.select(
        [
            city.key_column,
            "timestamp",
            offline_table.temp_c.mean().over(win6_offline).name("temp_mean_6s"),
        ]
    )

    city = Entity(name="city", key_column="city", description="City identifier")
    features = [
        Feature(
            name=feature_name,
            dtype=dt.float,
        )
        for feature_name in ["temp_mean_6s"]
    ]

    feature_views = [
        FeatureView(
            name=feature.name,
            features=(feature,),
            entities=(city,),
            offline_expr=offline_expr,
            timestamp_column=TIMESTAMP_COLUMN,
            ttl=timedelta(seconds=3600),
        )
        for feature in features
    ]

    store = FeatureStore(
        views={fv.name: fv for fv in feature_views}, online_client=fb.con
    )
    references = [f"{fv.name}:{fv.features[0].name}" for fv in feature_views]
    return store, references


def run_feature_server() -> None:
    server = FlightServer(
        FlightUrl(port=WEATHER_FEATURES_PORT),
        connection=xo.duckdb.connect,
        exchangers=[do_fetch_current_weather_udxf],
    )
    logging.info(f"Serving feature store on grpc://localhost:{WEATHER_FEATURES_PORT}")

    def handle_keyboard_interrupt(_):
        logging.info("Keyboard Interrupt: Feature server shutting down")
        server.close()

    serve_excepting = toolz.excepts(
        KeyboardInterrupt, server.serve, handle_keyboard_interrupt
    )
    serve_excepting(block=True)


def run_materialize_online() -> None:
    store, references = setup_store()
    store.materialize_online(references)
    logging.info("Materialized features to online store")


def run_infer() -> None:
    store, references = setup_store()
    entity_df = pd.DataFrame(
        {
            "city": ["London", "Tokyo", "New York"],
            "event_timestamp": [
                datetime(2025, 7, 3, 12, 59, 42),
                datetime(2025, 7, 3, 12, 12, 10),
                datetime(2025, 7, 3, 12, 40, 26),
            ],
        }
    )

    df = store.get_online_features(references, entity_df).execute()
    logging.info("Retrieved online features")
    print(df)


def run_historical_features() -> None:
    store, references = setup_store()
    dts = pd.to_datetime(
        ["2025-06-03 18:03:14", "2025-06-03 18:03:14", "2025-06-03 18:03:14"]
    ).tolist()

    entity_df = pd.DataFrame(
        {
            "city": ["London", "Tokyo", "New York"],
            "event_timestamp": dts,
        }
    )

    training_df = store.get_historical_features(entity_df, references)

    logging.info("Retrieved historical features")
    print("Entity DataFrame:")
    print(entity_df)
    print("\nTraining DataFrame with Historical Features:")
    print(training_df.execute())

    return training_df


def run_push_to_view_source() -> None:
    store, references = setup_store()
    # client = FlightClient("localhost", WEATHER_FEATURES_PORT)
    table = (
        xo.memtable(
            [{"city": c} for c in CITIES],
            schema=do_fetch_current_weather_udxf.schema_in_required,
        )
        .pipe(do_fetch_current_weather_flight_udxf)
        .to_pandas()
    )
    print(f"table: {table}")
    fv_keys = store.views.keys()
    try:
        for view in fv_keys:
                backend = store.views[view].offline_expr._find_backend()
                for t in backend.tables:
                    if t == view:
                        backend.insert(t, table)
                        logging.info(f"Pushed live data to {view}")
                        logging.info(f"{table['timestamp']}")
        time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down")


def ensure_table(backend, name=TABLE_BATCH):
    if name not in backend.tables:
        table = (
            xo.memtable(
                [{"city": c} for c in CITIES],
                schema=do_fetch_current_weather_udxf.schema_in_required,
            )
            .pipe(do_fetch_current_weather_flight_udxf)
            .to_pandas()
        )
        backend.create_table(TABLE_BATCH, table)


def run_push_to_view_source() -> None:
    store, references = setup_store()
    # client = FlightClient("localhost", WEATHER_FEATURES_PORT)
    table = (
        xo.memtable(
            [{"city": c} for c in CITIES],
            schema=do_fetch_current_weather_udxf.schema_in_required,
        )
        .pipe(do_fetch_current_weather_flight_udxf)
        .to_pandas()
    )
    print(f"table: {table}")

    fv_keys = store.views.keys()
    # I need an easy way to access batch source to push
    for view in fv_keys:
        logging.info(f"View: {view}")
        backend = store.views[view].offline_expr._find_backend()
        if TABLE_BATCH not in backend.tables:
            print(f"Creating table {TABLE_BATCH} in backend")
            backend.create_table(TABLE_BATCH, table)
        else:
            print(f"Table {TABLE_BATCH} already exists in backend")
            backend.insert(TABLE_BATCH, table)


def main() -> None:
    parser = argparse.ArgumentParser("Weather Flight Store")
    parser.add_argument(
        "command",
        choices=(
            "serve_features",  # start feature lookup server
            "materialize_online",  # push latest to flight feature store
            "historical",
            "infer",
            "push",
        ),
        help="Action: 'serve_features', 'materialize_online', 'historical', 'push' or 'infer'",
    )
    args = parser.parse_args()

    if args.command == "serve_features":
        run_feature_server()
    elif args.command == "materialize_online":
        run_materialize_online()
    elif args.command == "infer":
        run_infer()
    elif args.command == "push":
        run_push_to_view_source()
    elif args.command == "historical":
        run_historical_features()
    else:
        logging.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
