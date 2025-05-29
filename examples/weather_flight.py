import argparse
import time
import logging
from pathlib import Path
from datetime import datetime

import xorq as xo
import pyarrow as pa
from xorq.flight import Backend as FlightBackend, FlightServer, FlightUrl
from xorq.flight.client import FlightClient
from xorq.common.utils.import_utils import import_python
from xorq.common.utils.feature_utils import (
    Entity, Feature, FeatureView, FeatureStore
)

logging_format = '[%(asctime)s] %(levelname)s %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format, datefmt='%Y-%m-%d %H:%M:%S')

weather_lib = import_python("examples/libs/weather_lib.py")
do_fetch_current_weather_udxf = weather_lib.do_fetch_current_weather_udxf
do_fetch_current_weather_flight_udxf = weather_lib.do_fetch_current_weather_flight_udxf

# Ports for two Flight servers
PORT_API = 8816          # for weather-API fetch UDXF ingestion
PORT_FEATURES = 8817     # for serving materialized features

# Database files
DB_BATCH     = "weather_history_batch.db"     # full history batch store
DB_ONLINE    = "weather_history.db"           # live UDXF ingestion store
TABLE_BATCH  = "weather_history"
TABLE_ONLINE = "weather_history"
FEATURE_VIEW = "city_weather"
CITIES       = ["London", "Tokyo", "New York", "Lahore"]


def setup_store() -> FeatureStore:
    logging.info("Setting up FeatureStore")

    # 1. Entity (removed timestamp_column)
    city = Entity("city", key_column="city", description="City identifier")

    # 2. Offline source (batch history)
    offline_con = xo.duckdb.connect()
    offline_con.raw_sql("""
        INSTALL ducklake;
        INSTALL sqlite;
        ATTACH 'ducklake:sqlite:metadata.sqlite' AS my_ducklake (DATA_PATH 'file_path/');
        USE my_ducklake;
        """)
    offline_schema = do_fetch_current_weather_udxf.calc_schema_out()

    # 3. Flight backend for online features
    fb = FlightBackend()
    fb.do_connect(host="localhost", port=PORT_FEATURES)

    # 4. Feature definition (6h rolling mean temp) - OFFLINE
    win6 = xo.window(group_by=[city.key_column], order_by="timestamp", preceding=5, following=0)
    offline_expr = offline_con.tables[TABLE_BATCH].select([
        city.key_column,  # entity
        "timestamp",  # timestamp
        offline_con.tables[TABLE_BATCH].temp_c.mean().over(win6).name("temp_mean_6h")
    ])
    # 5. Online expression for live features
    live_expr = xo.memtable([{"city": "London"}]).pipe(do_fetch_current_weather_flight_udxf)
    win6_online = xo.window(group_by=[city.key_column], order_by="timestamp", preceding=5, following=0)
    offline_expr = live_expr.select([
        city.key_column,
        "timestamp",
        live_expr.temp_c.mean().over(win6_online).name("temp_mean_6h")
    ])
    try:
        online_expr = fb.tables[FEATURE_VIEW].select([
            city.key_column,
            "timestamp",
            "temp_mean_6h"
        ])
    except:
        online_expr = offline_expr

    # 6. Create Feature with both offline and online expressions
    feature_temp = Feature(
        name="temp_mean_6h",
        entity=city,
        timestamp_column="timestamp",
        offline_expr=offline_expr,
        online_expr=online_expr,
        dtype="float",
        description="6h rolling mean temp"
    )

    # 7. FeatureView & Store
    view = FeatureView(FEATURE_VIEW, city, [feature_temp])
    store = FeatureStore(online_client=fb.con)
    store.registry.register_entity(city)
    store.register_view(view)
    return store


def run_api_server() -> None:
    pa_schema = do_fetch_current_weather_udxf.calc_schema_out()
    arrays = [pa.array([], type=pa_schema.field(i).type) for i in range(len(pa_schema))]
    names = [f.name for f in pa_schema]
    duck_con = xo.duckdb.connect()
    duck_con.raw_sql("""
        INSTALL ducklake;
        INSTALL sqlite;
        ATTACH 'ducklake:sqlite:metadata.sqlite' AS my_ducklake (DATA_PATH 'file_path');
        USE my_ducklake;
    """)
    duck_con.create_table(TABLE_ONLINE, pa.Table.from_arrays(arrays, names=names), overwrite=True)
    logging.info(f"Initialized UDXF-store at {DB_ONLINE}")

    server = FlightServer(
        FlightUrl(port=PORT_API),
        connection=lambda: duck_con,
        exchangers=[do_fetch_current_weather_udxf]
    )
    logging.info(f"Serving UDXF ingestion on grpc://localhost:{PORT_API}")
    try:
        server.serve()
        while server.server is not None:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("UDXF server shutting down")
        server.close()


def run_feature_server() -> None:
    duck_con = xo.duckdb.connect()

    server = FlightServer(
        FlightUrl(port=PORT_FEATURES),
        connection=lambda: duck_con,
    )
    logging.info(f"Serving feature store on grpc://localhost:{PORT_FEATURES}")
    try:
        server.serve()
        while server.server is not None:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Feature server shutting down")
        server.close()


def run_writer() -> None:
    client = FlightClient("localhost", PORT_API)
    pa_schema = do_fetch_current_weather_udxf.calc_schema_out()
    arrays = [pa.array([], type=f.type) for f in pa_schema]
    names = [f.name for f in pa_schema]
    client.upload_data(TABLE_ONLINE, pa.Table.from_arrays(arrays, names=names), overwrite=False)
    logging.info("Initialized UDXF-store via FlightClient")

    while True:
        batches = xo.memtable([{"city": c} for c in CITIES], schema=do_fetch_current_weather_udxf.schema_in_required).to_pyarrow_batches()
        _, reader = client.do_exchange(do_fetch_current_weather_udxf.command, batches)
        tbl = reader.read_all()
        client.upload_data(TABLE_ONLINE, tbl, overwrite=False)
        logging.info(f"Appended {tbl.num_rows} rows to offline store")
        time.sleep(1)


def run_materialize_offline() -> None:
    """Compute batch features from historical data"""
    store = setup_store()
    # Execute offline computation - this would typically save results somewhere
    offline_df = store.views[FEATURE_VIEW].offline_expr().execute()
    logging.info(f"Computed offline features: {offline_df.shape[0]} rows")
    print(offline_df.head())


def run_materialize_online() -> None:
    store = setup_store()
    store.materialize_online(FEATURE_VIEW)
    logging.info("Materialized features to online store")


def run_infer() -> None:
    store = setup_store()
    df = store.get_online_features(FEATURE_VIEW, rows=[{"city": "London"}])
    logging.info("Retrieved online features")
    print(df)


def main() -> None:
    parser = argparse.ArgumentParser("Weather Flight Σtore")
    parser.add_argument(
        "command",
        choices=(
            "write",                 # UDXF ingestion
            "serve_api",             # start UDXF server
            "serve_features",        # start feature lookup server
            "materialize_offline",   # compute batch features
            "materialize_online",    # push latest to flight feature store
            "infer"
        ),
        help="Action: 'write', 'serve_api', 'serve_features', 'materialize_offline', 'materialize_online', or 'infer'"
    )
    args = parser.parse_args()

    if args.command == "write":
        run_writer()
    elif args.command == "serve_api":
        run_api_server()
    elif args.command == "serve_features":
        run_feature_server()
    elif args.command == "materialize_offline":
        run_materialize_offline()
    elif args.command == "materialize_online":
        run_materialize_online()
    elif args.command == "infer":
        run_infer()
    else:
        logging.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
