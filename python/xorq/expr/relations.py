import functools
import itertools
import operator
from collections import defaultdict
from typing import Any, Callable

import pyarrow as pa
import toolz
from opentelemetry import trace

import xorq as xo
from xorq.common.utils.otel_utils import tracer
from xorq.common.utils.rbr_utils import (
    copy_rbr_batches,
    instrument_reader,
)
from xorq.vendor import ibis
from xorq.vendor.ibis import Expr, Schema
from xorq.vendor.ibis.common.collections import (
    FrozenDict,
    FrozenOrderedDict,
)
from xorq.vendor.ibis.common.graph import Graph
from xorq.vendor.ibis.expr import operations as ops
from xorq.vendor.ibis.expr.format import fmt, render_schema
from xorq.vendor.ibis.expr.operations import Node, Relation


def replace_cache_table(node, kwargs):
    if kwargs:
        node = node.__recreate__(kwargs)

    if isinstance(node, CachedNode):
        return node.parent.op().replace(replace_cache_table)
    elif isinstance(node, RemoteTable):
        return node.remote_expr.op().replace(replace_cache_table)
    else:
        return node


def legacy_replace_cache_table(node, _, **kwargs):
    return replace_cache_table(node, (kwargs or dict(zip(node.argnames, node.args))))


# https://stackoverflow.com/questions/6703594/is-the-result-of-itertools-tee-thread-safe-python
class SafeTee(object):
    """tee object wrapped to make it thread-safe"""

    def __init__(self, teeobj, lock):
        self.teeobj = teeobj
        self.lock = lock

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.teeobj)

    def __copy__(self):
        return SafeTee(self.teeobj.__copy__(), self.lock)

    @classmethod
    def tee(cls, iterable, n=2):
        """tuple of n independent thread-safe iterators"""
        from itertools import tee
        from threading import Lock

        lock = Lock()
        return tuple(cls(teeobj, lock) for teeobj in tee(iterable, n))


def recursive_update(obj, replacements):
    if isinstance(obj, Node):
        if obj in replacements:
            return replacements[obj]
        else:
            return obj.__recreate__(
                {
                    name: recursive_update(arg, replacements)
                    for name, arg in zip(obj.argnames, obj.args)
                }
            )
    elif isinstance(obj, (tuple, list)):
        return tuple(recursive_update(o, replacements) for o in obj)
    elif isinstance(obj, dict):
        return {
            recursive_update(k, replacements): recursive_update(v, replacements)
            for k, v in obj.items()
        }
    else:
        return obj


def replace_source_factory(source: Any):
    def replace_source(node, _, **kwargs):
        if "source" in kwargs:
            kwargs["source"] = source
        return node.__recreate__(kwargs)

    return replace_source


def make_native_op(node):
    # FIXME: how to reference let.Backend.name?
    if node.source.name != "let":
        raise ValueError
    sources = node.source._sources
    native_source = sources.get_backend(node)
    if native_source.name == "let":
        raise ValueError

    def replace_table(_node, _kwargs):
        return sources.get_table_or_op(
            _node, _node.__recreate__(_kwargs) if _kwargs else _node
        )

    return node.replace(replace_table).to_expr()


class Tag(ops.Relation):
    schema: Schema
    parent: Any
    metadata: FrozenOrderedDict = FrozenOrderedDict()
    values = FrozenDict()

    @property
    def tag(self):
        return self.metadata.get("tag")

    def __dask_tokenize__(self):
        from xorq.common.utils.dask_normalize.dask_normalize_expr import (
            normalize_seq_with_caller,
        )

        return normalize_seq_with_caller(
            self.schema,
            self.parent,
            self.metadata,
            caller="normalize_tag",
        )


class CachedNode(ops.Relation):
    schema: Schema
    parent: Any
    source: Any
    storage: Any
    values = FrozenDict()


gen_name_namespace = "rbr-placeholder"
gen_name = toolz.compose(
    # some engines simply truncate long names
    operator.itemgetter(slice(0, 35)),
    functools.partial(ibis.util.gen_name, gen_name_namespace),
)


class RemoteTable(ops.DatabaseTable):
    remote_expr: Expr = None

    @classmethod
    def from_expr(cls, con, expr, name=None):
        name = name or gen_name()
        return cls(
            name=name,
            schema=expr.schema(),
            source=con,
            remote_expr=expr,
        )


def into_backend(expr, con, name=None):
    return RemoteTable.from_expr(con=con, expr=expr, name=name).to_expr()


class FlightExpr(ops.DatabaseTable):
    input_expr: Expr = None
    unbound_expr: Expr = None
    make_server: Callable = None
    make_connection: Callable = None
    do_instrument_reader: bool = False

    @classmethod
    def validate_schema(cls, input_expr, unbound_expr):
        from xorq.common.utils.graph_utils import walk_nodes

        (dt, *rest) = walk_nodes(ops.UnboundTable, unbound_expr)
        if rest or not isinstance(dt, ops.UnboundTable):
            raise ValueError
        if dt.schema != input_expr.schema():
            raise ValueError

    @classmethod
    def from_exprs(
        cls,
        input_expr,
        unbound_expr,
        make_server=None,
        make_connection=None,
        name=None,
        **kwargs,
    ):
        from xorq.flight import FlightServer

        def roundtrip_cloudpickle(obj):
            import cloudpickle

            return cloudpickle.loads(cloudpickle.dumps(obj))

        cls.validate_schema(input_expr, unbound_expr)
        return cls(
            name=name or gen_name(),
            schema=unbound_expr.schema(),
            source=input_expr._find_backend(),
            input_expr=input_expr,
            unbound_expr=roundtrip_cloudpickle(unbound_expr),
            make_server=make_server or FlightServer,
            make_connection=make_connection or xo.connect,
            **kwargs,
        )

    def to_rbr(self, do_instrument_reader=None):
        from xorq.flight.action import AddExchangeAction
        from xorq.flight.exchanger import (
            UnboundExprExchanger,
        )

        if do_instrument_reader is None:
            do_instrument_reader = self.do_instrument_reader

        def inner(flight_exchange):
            rbr_in = flight_exchange.input_expr.to_pyarrow_batches()
            if do_instrument_reader:
                rbr_in = instrument_reader(rbr_in, "input: ")
            with flight_exchange.make_server() as server:
                client = server.client
                unbound_expr_exchanger = UnboundExprExchanger(
                    flight_exchange.unbound_expr
                )
                client.do_action(
                    AddExchangeAction.name,
                    unbound_expr_exchanger,
                    options=client._options,
                )
                (fut, rbr_out) = client.do_exchange_batches(
                    unbound_expr_exchanger.command, rbr_in
                )
                if do_instrument_reader:
                    rbr_out = instrument_reader(rbr_out, "output: ")

                # HAK: account for https://github.com/apache/arrow-rs/issues/6471
                rbr_out = copy_rbr_batches(rbr_out)
                yield from rbr_out

        gen = inner(self)
        schema = self.schema.to_pyarrow()
        return pa.RecordBatchReader.from_batches(schema, gen)

    def serve(self, make_server=None, **kwargs):
        return flight_serve_unbound(
            self.unbound_expr, make_server=make_server, **kwargs
        )


def flight_serve(
    expr,
    make_server=None,
    **kwargs,
):
    return flight_serve_unbound(expr.unbind(), make_server=make_server, **kwargs)


def flight_serve_unbound(
    unbound_expr,
    make_server=None,
    **kwargs,
):
    from xorq.flight import FlightServer
    from xorq.flight.exchanger import (
        UnboundExprExchanger,
    )

    @toolz.curry
    def do_exchange(server, command, expr):
        (_, rbr_out) = server.client.do_exchange(command, expr)
        return rbr_out

    unbound_expr_exchanger = UnboundExprExchanger(unbound_expr)
    server = (make_server or FlightServer)(**kwargs)
    server.exchangers += (unbound_expr_exchanger,)
    server.serve()
    return server, do_exchange(server, unbound_expr_exchanger.command)


@toolz.curry
def flight_expr(
    expr,
    unbound_expr,
    name=None,
    make_server=None,
    make_connection=None,
    inner_name=None,
    con=None,
    **kwargs,
):
    return (
        FlightExpr.from_exprs(
            expr,
            unbound_expr,
            make_server=make_server,
            make_connection=make_connection,
            name=inner_name,
            **kwargs,
        )
        .to_expr()
        .into_backend(con=con or expr._find_backend(), name=name)
    )


class FlightUDXF(ops.DatabaseTable):
    input_expr: Expr = None
    # FIXME: fix circular import issue so we can possibly pass an instance of AbstractExchanger
    udxf: type = None
    make_server: Callable = None
    make_connection: Callable = None
    do_instrument_reader: bool = False

    @classmethod
    def validate_schema(cls, input_expr, udxf):
        if not udxf.schema_in_condition(input_expr.schema()):
            schema_in_required = getattr(udxf, "schema_in_required", None)
            raise ValueError(
                "Schema validation failed"
                if schema_in_required is None
                else f"Schema validation failed, expected: {schema_in_required} found: {input_expr.schema()}"
            )
        schema_out = udxf.calc_schema_out(input_expr.schema())
        return schema_out

    @classmethod
    def from_expr(
        cls,
        input_expr,
        udxf,
        make_server=None,
        make_connection=None,
        name=None,
        **kwargs,
    ):
        from xorq.common.utils.tls_utils import TLSKwargs
        from xorq.flight import FlightServer

        def make_mtls_server():
            tls_kwargs = TLSKwargs.from_common_name(verify_client=True)
            return FlightServer(verify_client=True, **tls_kwargs.server_kwargs)

        # FIXME do we need make_connection

        schema = cls.validate_schema(input_expr, udxf)
        return cls(
            name=name or gen_name(),
            schema=schema,
            source=input_expr._find_backend(),
            input_expr=input_expr,
            udxf=udxf,
            make_server=make_server or make_mtls_server,
            make_connection=make_connection or xo.connect,
            **kwargs,
        )

    def to_rbr(self, do_instrument_reader=None):
        from xorq.flight.action import AddExchangeAction

        if do_instrument_reader is None:
            do_instrument_reader = self.do_instrument_reader

        def inner(flight_udxf):
            rbr_in = flight_udxf.input_expr.to_pyarrow_batches()
            if do_instrument_reader:
                rbr_in = instrument_reader(rbr_in, "input: ")
            with flight_udxf.make_server() as server:
                client = server.client
                client.do_action(
                    AddExchangeAction.name,
                    self.udxf,
                    options=client._options,
                )
                (fut, rbr_out) = client.do_exchange_batches(self.udxf.command, rbr_in)
                if do_instrument_reader:
                    rbr_out = instrument_reader(rbr_out, "output: ")
                # HAK: account for https://github.com/apache/arrow-rs/issues/6471
                rbr_out = copy_rbr_batches(rbr_out)
                yield from rbr_out

        gen = inner(self)
        schema = self.schema.to_pyarrow()
        return pa.RecordBatchReader.from_batches(schema, gen)


@toolz.curry
def flight_udxf(
    expr,
    process_df,
    maybe_schema_in,
    maybe_schema_out,
    name=None,
    make_server=None,
    make_connection=None,
    con=None,
    inner_name=None,
    make_udxf_kwargs=(),
    **kwargs,
):
    """
    Create a User-Defined Exchange Function (UDXF) that executes a pandas DataFrame
    transformation via Apache Arrow Flight protocol.

    This function wraps a pandas-based data processing function in an Arrow Flight
    ephemeral server, enabling distributed execution of custom user-defined functions.
    The function creates a FlightUDXF operation that can be integrated into xorq
    expression pipelines for scalable data processing.

    Parameters
    ----------
    expr : Expr
        The input Ibis expression that provides data to the UDXF. This expression's
        output will be streamed to the Flight server for processing.

    process_df : callable
        A function that takes a pandas DataFrame as input and returns a transformed
        pandas DataFrame. This function defines the core transformation logic that
        will be executed on the Flight server. The function signature should be:
        `process_df(df: pd.DataFrame) -> pd.DataFrame`

    maybe_schema_in : Schema or callable
        Input schema specification. Can be either:
        - A pyarrow Schema object defining the expected input schema
        - A callable that validates the input schema and returns True/False
        Used to validate that the input expression's schema matches expectations.

    maybe_schema_out : Schema or callable
        Output schema specification. Can be either:
        - A pyarrow Schema object defining the expected output schema
        - A callable that computes the output schema from the input schema
        Used to determine the schema of the transformed data.

    name : str, optional
        Name for the resulting table in the target backend. If not provided,
        a unique name will be generated automatically.

    make_server : callable, optional
        Factory function for creating the Arrow Flight server. Defaults to
        creating an mTLS-enabled FlightServer with client verification.
        The function should return a FlightServer instance.

    make_connection : callable, optional
        Factory function for creating connections to backends. Defaults to
        `xo.connect`. Used for establishing connections during Flight operations.

    con : Backend, optional
        Target backend connection where the result will be materialized.
        If not provided, uses the backend from the input expression.

    inner_name : str, optional
        Internal name for the FlightUDXF operation. If not provided,
        a unique name will be generated.

    make_udxf_kwargs : tuple, optional
        Additional keyword arguments to pass to the UDXF creation process.
        Should be a tuple of (key, value) pairs that will be converted to
        a dictionary and passed to `make_udxf`.

    **kwargs : dict
        Additional keyword arguments passed to the FlightUDXF constructor.

    Returns
    -------
    Expr
        A Xorq expression representing the transformed data. This expression
        can be further chained with other operations or executed to materialize
        the results.

    Examples
    --------
    Basic sentiment analysis:

    >>> import pandas as pd
    >>> import xorq as xo
    >>> from xorq.common.utils.toolz_utils import curry
    >>>
    >>> @curry
    >>> def add_sentiment(df: pd.DataFrame, input_col, output_col):
    ...     # Simplified sentiment analysis
    ...     sentiments = df[input_col].apply(lambda x: "POSITIVE" if "good" in x.lower() else "NEGATIVE")
    ...     return df.assign(**{output_col: sentiments})
    >>>
    >>> # Define schemas
    >>> schema_in = xo.schema({"text": "string"})
    >>> schema_out = xo.schema({"text": "string", "sentiment": "string"})
    >>>
    >>> # Create the UDXF
    >>> sentiment_udxf = xo.expr.relations.flight_udxf(
    ...     process_df=add_sentiment(input_col="text", output_col="sentiment"),
    ...     maybe_schema_in=schema_in,
    ...     maybe_schema_out=schema_out,
    ...     name="SentimentAnalyzer"
    ... )
    >>>
    >>> # Apply to data
    >>> data = xo.memtable({"text": ["This is good", "This is bad"]})
    >>> result = data.pipe(sentiment_udxf).execute()

    Data fetching and processing:

    >>> @curry
    >>> def fetch_external_data(df, api_endpoint):
    ...     # Fetch additional data for each row
    ...     results = []
    ...     for _, row in df.iterrows():
    ...         # Simulate API call
    ...         enriched_data = {"id": row["id"], "enriched": f"data_for_{row['id']}"}
    ...         results.append(enriched_data)
    ...     return pd.DataFrame(results)
    >>>
    >>> fetch_udxf = xo.expr.relations.flight_udxf(
    ...     process_df=fetch_external_data(api_endpoint="https://api.example.com"),
    ...     maybe_schema_in=xo.schema({"id": "int64"}).to_pyarrow(),
    ...     maybe_schema_out=xo.schema({"id": "int64", "enriched": "string"}).to_pyarrow(),
    ...     name="DataEnricher"
    ... ) # quartodoc: +SKIP

    Notes
    -----
    - The function uses Apache Arrow Flight for efficient data transfer between
      the client and server processes
    - By default, the Flight server uses mTLS (mutual TLS) for secure communication
    - The process_df function is executed in a separate process/server, enabling
      distributed processing and isolation
    - Schema validation ensures type safety and prevents runtime errors
    - The function is curried using toolz.curry, allowing partial application
    """

    from xorq.flight.exchanger import make_udxf

    udxf = make_udxf(
        process_df,
        maybe_schema_in,
        maybe_schema_out,
        **dict(make_udxf_kwargs),
    )
    return (
        FlightUDXF.from_expr(
            input_expr=expr,
            udxf=udxf,
            make_server=make_server,
            make_connection=make_connection,
            name=inner_name,
            **kwargs,
        )
        .to_expr()
        .into_backend(con=con or expr._find_backend(), name=name)
    )


class Read(ops.Relation):
    method_name: str
    name: str
    schema: Schema
    source: Any
    read_kwargs: Any
    normalize_method: Any
    values = FrozenDict()

    def make_dt(self):
        method = getattr(self.source, self.method_name)
        dt = method(**dict(self.read_kwargs)).op()
        return dt

    def make_unbound_dt(self):
        import dask

        name = f"{self.name}-{dask.base.tokenize(self)}"
        return ops.UnboundTable(
            name=name,
            schema=self.schema,
        )


_count = itertools.count()


@tracer.start_as_current_span("register_and_transform_remote_tables")
def register_and_transform_remote_tables(expr):
    created = {}

    op = expr.op()
    graph, _ = Graph.from_bfs(op).toposort()
    counts = defaultdict(int)
    for node in graph:
        if isinstance(node, RemoteTable):
            counts[node] += 1

        if isinstance(node, Relation):
            for arg in node.__args__:
                if isinstance(arg, RemoteTable):
                    counts[arg] += 1

    if counts:
        trace.get_current_span().add_event(
            "remote_table.replace", {"counts.values": tuple(counts.values())}
        )
    batches_table = {}
    for arg, count in counts.items():
        ex = arg.remote_expr
        if not ex.op().find((RemoteTable, CachedNode, Read)):
            batches = ex.to_pyarrow_batches()  # execute in native backend
        else:
            batches = xo.to_pyarrow_batches(ex)
        schema = ex.as_table().schema().to_pyarrow()
        replicas = SafeTee.tee(batches, count)
        batches_table[arg] = (schema, list(replicas))

    def mark_remote_table(node):
        schema, batchess = batches_table[node]
        name = f"{node.name}_cu{next(_count)}_t{len(batchess)}"
        reader = pa.RecordBatchReader.from_batches(schema, batchess.pop())
        result = node.source.read_record_batches(reader, table_name=name)
        created[name] = node.source
        return result.op()

    def replacer(node, kwargs):
        kwargs = kwargs or {}
        if isinstance(node, Relation):
            updated = {}
            for k, v in list(kwargs.items()):
                try:
                    if v in batches_table:
                        updated[v] = mark_remote_table(v)

                except TypeError:  # v may not be hashable
                    continue

            if len(updated) > 0:
                kwargs = {k: recursive_update(v, updated) for k, v in kwargs.items()}

        if kwargs:
            node = node.__recreate__(kwargs)
        if isinstance(node, RemoteTable):
            result = mark_remote_table(node)
            batches_table[result] = batches_table.pop(node)
            node = result

        return node

    expr = op.replace(replacer).to_expr()
    return expr, created


def render_backend(con):
    return f"{con.name}-{id(con)}"


def get_storage_params(storage):
    from xorq.caching import (
        ParquetSnapshotStorage,
        ParquetStorage,
        SourceSnapshotStorage,
        SourceStorage,
    )

    storage_repr = None, None
    match storage:
        case ParquetStorage():
            storage_repr = "modification_time", True
        case ParquetSnapshotStorage():
            storage_repr = "snapshot", True
        case SourceStorage():
            storage_repr = "modification_time", False
        case SourceSnapshotStorage():
            storage_repr = "snapshot", False
    return storage_repr + (render_backend(storage.source),)


@fmt.register(CachedNode)
def _fmt_cache_node(op, schema, parent, source, storage, **kwargs):
    strategy, parquet, backend = get_storage_params(storage)
    name = f"{op.__class__.__name__}[{parent}, strategy={strategy}, parquet={parquet}, source={backend}]\n"
    return name + render_schema(schema, 1)


@fmt.register(RemoteTable)
def _fmt_remote_table(op, name, remote_expr, **kwargs):
    name = f"{op.__class__.__name__}[{remote_expr}, name={name}]\n"
    return name + render_schema(op.schema, 1)


@fmt.register(Read)
def _fmt_read(op, name, method_name, source, **kwargs):
    backend = render_backend(source)
    name = f"{op.__class__.__name__}[name={name}, method_name={method_name}, source={backend}]\n"
    return name + render_schema(op.schema, 1)
