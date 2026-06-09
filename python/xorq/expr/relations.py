from __future__ import annotations

import functools
import operator
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Any, Callable

import pyarrow as pa
import toolz
from batchcorder import StreamCache
from opentelemetry import trace

from xorq.backends.xorq_datafusion import connect as xo_connect
from xorq.common.exceptions import IntegrityError
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
from xorq.vendor.ibis.expr import operations as ops
from xorq.vendor.ibis.expr.format import fmt, render_schema
from xorq.vendor.ibis.expr.operations import Node
from xorq.writes import DrainingIterator, WriteThrough


if TYPE_CHECKING:
    from xorq.vendor.ibis.backends import BaseBackend


def replace_cache_table(node: Node, kwargs: dict[str, Any] | None) -> Node:
    if kwargs:
        node = node.__recreate__(kwargs)
    while isinstance(node, CachedNode):
        node = node.parent.op()
    return node


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


class Tag(ops.Relation):
    schema: Schema
    parent: ops.Relation
    metadata: FrozenOrderedDict = FrozenOrderedDict()
    values = FrozenDict()

    @property
    def tag(self) -> str | None:
        return self.metadata.get("tag")


class HashingTag(Tag):
    """A Tag subclass whose metadata contributes to the content hash.

    Unlike Tag (which is stripped before hashing), HashingTag is preserved
    during hash computation so expressions with different HashingTag metadata
    produce distinct hashes.
    """

    def __dasher_tokenize__(self) -> tuple:
        return ("hashing-tag", self.schema, self.metadata)


class TeeNode(ops.Relation):
    """A transparent pass-through that hands its stream to a WriteThrough.

    Schema and rows equal the parent's. The cache hash
    (``expr.ls.tokenized``) strips the node so ``expr.tee(s)`` caches
    identically to ``expr``. The build hash (``get_expr_hash``) includes
    the writer identity so different writers produce different build artifacts.

    When ``drain`` is True (the default), early termination by downstream
    causes the remaining batches to be consumed through the writer in a
    background thread so the write completes. Pass ``drain=False`` to let a
    downstream early-stop (``LIMIT``/``head``) abort the write instead.
    """

    schema: Schema
    parent: ops.Relation
    writer: WriteThrough
    drain: bool = True
    values = FrozenDict()

    def __init__(
        self,
        schema: Schema,
        parent: ops.Relation,
        writer: WriteThrough,
        drain: bool = True,
    ) -> None:
        if schema != parent.schema:
            raise IntegrityError(
                f"TeeNode schema {schema} does not match parent schema "
                f"{parent.schema}; a TeeNode is a transparent pass-through."
            )
        super().__init__(schema=schema, parent=parent, writer=writer, drain=drain)

    def __dasher_tokenize__(self) -> tuple:
        return ("tee-node", self.schema, self.writer)


class DatabaseTableView(ops.DatabaseTable):
    pass


class CachedNode(DatabaseTableView):
    parent: Any = None
    cache: Any = None


gen_name_namespace = "rbr-placeholder"
gen_name = toolz.compose(
    # some engines simply truncate long names
    operator.itemgetter(slice(0, 35)),
    functools.partial(ibis.util.gen_name, gen_name_namespace),
)


class RemoteTable(DatabaseTableView):
    remote_expr: Expr

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


class FlightExpr(DatabaseTableView):
    input_expr: Expr = None
    unbound_expr: Expr = None
    make_server: Callable = None
    make_connection: Callable = None
    do_instrument_reader: bool = False

    @classmethod
    def validate_schema(cls, input_expr, unbound_expr):
        from xorq.common.utils.graph_utils import walk_nodes  # noqa: PLC0415

        (dt, *rest) = walk_nodes(ops.UnboundTable, unbound_expr)
        if rest or not isinstance(dt, ops.UnboundTable):
            raise ValueError("unbound_expr must contain exactly one UnboundTable")
        if dt.schema != input_expr.schema():
            raise ValueError(
                "Schema of unbound_expr does not match schema of input_expr"
            )

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
        from xorq.flight import FlightServer  # noqa: PLC0415

        def roundtrip_cloudpickle(obj):
            import cloudpickle  # noqa: PLC0415

            return cloudpickle.loads(cloudpickle.dumps(obj))

        cls.validate_schema(input_expr, unbound_expr)
        return cls(
            name=name or gen_name(),
            schema=unbound_expr.schema(),
            source=input_expr._find_backend(),
            input_expr=input_expr,
            unbound_expr=roundtrip_cloudpickle(unbound_expr),
            make_server=make_server or FlightServer,
            make_connection=make_connection or xo_connect,
            **kwargs,
        )

    def to_rbr(self, do_instrument_reader=None):
        from xorq.flight.action import AddExchangeAction  # noqa: PLC0415
        from xorq.flight.exchanger import (  # noqa: PLC0415
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
    from xorq.flight import FlightServer  # noqa: PLC0415
    from xorq.flight.exchanger import (  # noqa: PLC0415
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


class FlightUDXF(DatabaseTableView):
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
        from xorq.common.utils.tls_utils import TLSKwargs  # noqa: PLC0415
        from xorq.flight import FlightServer  # noqa: PLC0415

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
            make_connection=make_connection or xo_connect,
            **kwargs,
        )

    def to_rbr(self, do_instrument_reader=None):
        from xorq.flight.action import AddExchangeAction  # noqa: PLC0415

        if do_instrument_reader is None:
            do_instrument_reader = self.do_instrument_reader

        def inner(flight_udxf):
            rbr_in = flight_udxf.input_expr.to_pyarrow_batches()
            if do_instrument_reader:
                rbr_in = instrument_reader(rbr_in, "input: ")
            with flight_udxf.make_server() as server:
                client = server.client
                if not getattr(self.udxf, "_xorq_server_has_command", False):
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
    >>> import xorq.api as xo
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

    from xorq.flight.exchanger import make_udxf  # noqa: PLC0415

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


class Read(ops.DatabaseTable):
    method_name: str = None
    read_kwargs: Any = ()
    normalize_method: Callable = None

    def make_dt(self):
        from xorq.common.constants import READ_EXCLUDE_KEYS  # noqa: PLC0415

        method = getattr(self.source, self.method_name)
        args = tuple(v for k, v in self.read_kwargs if k == "hash_path")
        kwargs = {k: v for k, v in self.read_kwargs if k not in READ_EXCLUDE_KEYS}
        dt = method(*args, **kwargs).op()
        return dt

    def make_unbound_dt(self):
        return ops.UnboundTable(
            name=self.name,
            schema=self.schema,
        )


def count_remote_table_readers(expr):
    """Count how many times each ``RemoteTable`` is physically scanned.

    The scan count is not visible in the expression graph: a single graph
    reference can compile to several physical table scans. The asof-join with
    tolerance lowering, for one, scans an input twice (xorq #983). Each
    ``RemoteTable`` is rewritten to a placeholder table under a freshly
    generated, unique name, the placeholder expression is compiled to a sqlglot
    AST, and the ``Table`` nodes bearing that name are counted — giving the
    exact per-table scan count whatever lowering the backend compiler applies.
    Counting ``Table`` AST nodes (rather than substrings of the rendered SQL)
    ignores column references, aliases, and string literals, so even a short or
    shared user-supplied table name cannot inflate the count.

    The counts become each ``StreamCache``'s ``max_readers``, bounding memory:
    batches are evicted once all readers advance past them. Each count is
    floored at 1 — a bare ``RemoteTable`` is still scanned once, and
    ``max_readers=0`` would forbid the reader that is in fact created.

    Returns an empty mapping when no SQL AST can be produced (e.g. a non-SQL
    backend); the caller then builds an unbounded cache — safe, but without
    eviction.
    """
    import sqlglot as sg  # noqa: PLC0415

    op = expr.op()
    sentinels = {}

    def replacer(node, kwargs):
        if isinstance(node, RemoteTable):
            name = gen_name()
            sentinels[node] = name
            return ops.DatabaseTable(name=name, schema=node.schema, source=node.source)
        if kwargs:
            node = node.__recreate__(kwargs)
        return node

    placeholder = op.replace(replacer).to_expr()
    if not sentinels:
        return {}
    try:
        provider = placeholder._find_backend(use_default=True)
        compiler = getattr(provider, "compiler", None)
        if compiler is None:
            return {}
        out = compiler.to_sqlglot(placeholder.unbind())
    except Exception:
        return {}

    counts = Counter()
    for query in out if isinstance(out, list) else [out]:
        for table in query.find_all(sg.exp.Table):
            counts[table.name] += 1
    return {node: max(1, counts[name]) for node, name in sentinels.items()}


@tracer.start_as_current_span("register_and_transform_remote_tables")
def register_and_transform_remote_tables(
    expr: Expr, **kwargs: Any
) -> tuple[Expr, dict]:
    created = {}
    caches = []

    op = expr.op()
    reader_counts = count_remote_table_readers(expr)

    def replacer(node, kwargs):
        if isinstance(node, RemoteTable):
            remote_expr = node.remote_expr
            # Cache the reader's own (physical) schema, not the logical
            # ``as_table().schema()``. The two can diverge -- e.g. a dropped
            # ``row_number`` still rides along in the stream -- and StreamCache
            # exports the declared schema over the Arrow C stream FFI, where a
            # batch with extra children aborts the process
            # (``fields.len() == num_children``). The reader already carries the
            # true schema; column coercion is the consumer's job
            # (``read_record_batches`` -> ``_select_and_cast``).
            cache = StreamCache(
                remote_expr.to_pyarrow_batches(),
                max_readers=reader_counts.get(node),
            )
            caches.append(cache)
            table_name = gen_name()
            result = node.source.read_record_batches(cache, table_name=table_name)
            created[table_name] = node.source
            return result.op()

        if kwargs:
            node = node.__recreate__(kwargs)

        return node

    # Intentionally op.replace, not replace_nodes: replacer has side effects
    # that must not descend into opaque sub-exprs (e.g. ExprScalarUDF.computed_kwargs_expr)
    try:
        expr = op.replace(replacer).to_expr()
    except Exception:
        for cache in caches:
            cache.close()
        raise
    if caches:
        trace.get_current_span().add_event(
            "remote_table.replace", {"remote_table.count": len(caches)}
        )
    return expr, created, caches


# Backends whose single, non-re-entrant connection deadlocks the streaming tee
# transport (it pulls the parent reader while that connection serves the outer
# query). See ADR-0014.
_NON_REENTRANT_TEE_BACKENDS = frozenset({"duckdb"})


@tracer.start_as_current_span("register_and_transform_tee_nodes")
def register_and_transform_tee_nodes(
    expr: Expr,
) -> tuple[Expr, dict[str, BaseBackend], list[DrainingIterator]]:
    """Replace each surviving `TeeNode` with a backend table fed by the
    writer's ``write_through(batches)`` generator.

    The writer wraps the parent's batch stream: it pulls, writes as a side
    effect, and yields each batch onward. Runs after cache resolution, so a
    downstream cache hit prunes the `TeeNode` before this pass sees it and
    the write never fires.

    Returns ``(expr, created, drains)``: the transformed expression, a
    ``{table_name: con}`` map of the intermediate pass-through tables
    registered on each parent backend (these persist in the backend's
    catalog until dropped, so callers must drop them once downstream
    consumption is done), and a list of `DrainingIterator` instances whose
    ``close()`` must be called after downstream execution completes so that
    the writer can finish consuming the stream.
    """
    from xorq.common.utils.caching_utils import find_backend  # noqa: PLC0415

    drains: list[DrainingIterator] = []
    created: dict[str, BaseBackend] = {}

    def replacer(node, kwargs):
        if not isinstance(node, TeeNode):
            return node.__recreate__(kwargs) if kwargs else node
        if kwargs:
            node = node.__recreate__(kwargs)
        parent_expr = node.parent.to_expr()
        con, _ = find_backend(node.parent, use_default=True)
        if con.name in _NON_REENTRANT_TEE_BACKENDS:
            warnings.warn(
                f"tee() on a {con.name!r} backend is likely to deadlock: the "
                "streaming write pulls the parent reader on the same single "
                "connection that serves the outer query. Phase 1 targets "
                "engines that allow concurrent reader pulls (e.g. datafusion). "
                "See ADR-0014.",
                stacklevel=2,
            )
        reader = parent_expr.to_pyarrow_batches()
        write_iter = node.writer.write_through(reader)
        if node.drain:
            write_iter = DrainingIterator(write_iter)
            drains.append(write_iter)
        wrapped = pa.RecordBatchReader.from_batches(
            reader.schema,
            write_iter,
        )
        table_name = gen_name()
        table = con.read_record_batches(wrapped, table_name=table_name)
        created[table_name] = con
        return table.op()

    # op.replace, not replace_nodes: this replacer has side effects (registers
    # pass-through tables, starts writers), so it must fire only at *this*
    # execution boundary. Descending into opaque sub-exprs (RemoteTable,
    # CachedNode, Flight*, ExprScalarUDF) is deliberately avoided -- that is not
    # a coverage gap: each opaque interior re-enters this transform at its own
    # execution boundary (e.g. caching/storage.py resolves and re-transforms the
    # cached parent; into_backend/flight re-pull via to_pyarrow_batches), so its
    # TeeNodes still fire exactly once. replace_nodes would fire them twice.
    op = expr.op()
    return op.replace(replacer).to_expr(), created, drains


def render_backend(con):
    return f"{con.name}-{id(con)}"


def get_cache_params(cache):
    from xorq.caching import (  # noqa: PLC0415
        ParquetCache,
        ParquetSnapshotCache,
        SourceCache,
        SourceSnapshotCache,
    )

    cache_repr = None, None
    match cache:
        case ParquetCache():
            cache_repr = "modification_time", True
        case ParquetSnapshotCache():
            cache_repr = "snapshot", True
        case SourceCache():
            cache_repr = "modification_time", False
        case SourceSnapshotCache():
            cache_repr = "snapshot", False
    return cache_repr + (render_backend(cache.storage.source),)


@fmt.register(CachedNode)
def _fmt_cache_node(op, schema, parent, source, cache, **kwargs):
    strategy, parquet, backend = get_cache_params(cache)
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


def prepare_create_table_from_expr(con, expr, **kwargs):
    from xorq.expr.api import _transform_expr  # noqa: PLC0415

    if (expr_backend := expr._find_backend()) != con:
        raise ValueError(f"expr backend must be {con}, is {expr_backend}")
    (table, _created, _drains, _caches) = _transform_expr(expr, **kwargs)
    try:
        return table
    finally:
        for cache in _caches:
            cache.close()
