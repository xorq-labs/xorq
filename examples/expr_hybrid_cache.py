import io
import logging

import gcsfs
import pyarrow.parquet as pq
from attr import field, frozen
from attr.validators import instance_of
from toolz import compose, curry, pipe

import xorq as xo
from xorq.caching import Cache, CacheStorage, ModificationTimeStrategy
from xorq.flight.client import FlightClient
from xorq.vendor.ibis.expr import types as ir


logger = logging.getLogger(__name__)


@curry
def construct_obj_store_path(bucket_name, key):
    return f"{bucket_name}/{key}.parquet"


@curry
def get_table_schema(client, key):
    result = client.do_action_one("get_schema_using_query", f"select * from '{key}';")
    return result if result else None


@curry
def create_table_expr(key, schema):
    if schema is None:
        return None
    return xo.table(schema=schema, name=key)


@curry
def execute_query(client, expr):
    if expr is None:
        return None
    return client.execute_query(expr)


@curry
def read_parquet_from_fs(fs, path):
    with fs.open(path, "rb") as f:
        return pq.read_table(f)


@curry
def to_memtable_op(table):
    return xo.memtable(table).op()


@curry
def write_to_fs(fs, path, table):
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    with fs.open(path, "wb") as f:
        f.write(buf.read())
    return table


@curry
def upload_to_flight(client, key, table):
    client.upload_data(key, table)
    return table


@frozen
class HybridStorage(CacheStorage):
    client: FlightClient = field(validator=instance_of(FlightClient))
    source: xo.vendor.ibis.backends.BaseBackend = field(
        validator=instance_of(xo.vendor.ibis.backends.BaseBackend),
        factory=xo.config._backend_init,
    )
    bucket_name: str = field(default="expr-cache", validator=instance_of(str))
    _fs: gcsfs.GCSFileSystem = field(init=False, default=None)

    @property
    def fs(self):
        if self._fs is None:
            object.__setattr__(self, "_fs", gcsfs.GCSFileSystem())
        return self._fs

    def key_exists(self, key):
        path = construct_obj_store_path(self.bucket_name, key)
        return self.fs.exists(path)

    def _get(self, key):
        try:
            # get_from_flight = compose(
            #     to_memtable_op,
            #     execute_query(self.client),
            #     create_table_expr(key),
            #     get_table_schema(self.client)
            # )

            result = pipe(
                key,
                get_table_schema(self.client),
                create_table_expr(key),
                execute_query(self.client),
                to_memtable_op,
            )

            if result is not None:
                return result

            return self._get_from_gcs(key)
        except Exception:
            print("Flight fetch error. Perhaps, the key does not exist?")
            return self._get_from_gcs(key)

    def _get_from_gcs(self, key):
        try:
            # process_gcs = compose(
            #     to_memtable_op,
            #     lambda table: upload_to_flight(self.client, key, table),
            #     read_parquet_from_fs(self.fs),
            #     construct_obj_store_path(self.bucket_name, key),
            # )
            #
            # return process_gcs(key)
            return pipe(
                key,
                construct_obj_store_path(self.bucket_name),
                read_parquet_from_fs(self.fs),
                lambda table: upload_to_flight(self.client, key, table),
                to_memtable_op,
            )

        except FileNotFoundError as e:
            raise KeyError(f"GCS object not found: {key}") from e

    def _put(self, key, value):
        arrow_table = xo.to_pyarrow(value.to_expr())
        path = construct_obj_store_path(self.bucket_name, key)
        # this is not optimal since it blocks and also send the same data twice
        # ideally this is managed server side to asynchronously run a method on
        # cache miss
        return pipe(
            arrow_table,
            lambda table: write_to_fs(self.fs, path, table),
            lambda table: upload_to_flight(self.client, key, table),
            to_memtable_op,
        )

    def _drop(self, key):
        raise NotImplementedError("Manual dropping not supported by HybridStorage.")


@frozen
class HybridStorageClient:
    client: FlightClient = field(validator=instance_of(FlightClient))
    source: xo.vendor.ibis.backends.BaseBackend = field(
        validator=instance_of(xo.vendor.ibis.backends.BaseBackend),
        factory=xo.config._backend_init,
    )
    bucket_name: str = field(default="expr-cache", validator=instance_of(str))
    cache: Cache = field(init=False)

    def __attrs_post_init__(self):
        hybrid_cache = pipe(
            HybridStorage(
                client=self.client, source=self.source, bucket_name=self.bucket_name
            ),
            lambda storage: Cache(strategy=ModificationTimeStrategy(), storage=storage),
        )

        object.__setattr__(self, "cache", hybrid_cache)

    def exists(self, expr: ir.Expr) -> bool:
        return self.cache.exists(expr)

    def __getattr__(self, attr):
        get_attr_from_components = compose(
            lambda obj_list: next(
                (getattr(obj, attr) for obj in obj_list if hasattr(obj, attr)), None
            ),
            lambda: [self.cache, self.cache.storage, self.cache.strategy],
        )

        result = get_attr_from_components()
        if result is not None:
            return result

        return object.__getattribute__(self, attr)


# Example usage
# TODO: wrap FlightClient in HybridStorageClient
# run python examples/duckdb_flight_example serve -p 500051
flight_client = FlightClient(host="localhost", port=50051)
storage = HybridStorageClient(
    client=flight_client,
    source=xo.config._backend_init(),
    bucket_name="expr-cache",
)
con = xo.connect()

expr = xo.deferred_read_csv(
    path=xo.options.pins.get_path("bank-marketing"),
    con=con,
).cache(storage=storage)
