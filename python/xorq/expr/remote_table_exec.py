from __future__ import annotations

import functools
import weakref
from collections import Counter
from typing import Any, Callable, NamedTuple

import pyarrow as pa
from batchcorder import StreamCache
from opentelemetry import trace

from xorq.common.utils.logging_utils import get_logger
from xorq.common.utils.otel_utils import tracer
from xorq.expr.relations import RemoteTable, gen_name
from xorq.vendor.ibis import Expr
from xorq.vendor.ibis.backends import BaseBackend
from xorq.vendor.ibis.expr import operations as ops


logger = get_logger(__name__)


def count_remote_table_readers(expr: Expr) -> dict[RemoteTable, int]:
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
        provider = placeholder._find_backend(
            use_default=True
        )  # xorq-style: disable=protected-access
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


def drop_placeholder(con: BaseBackend, table_name: str) -> None:
    """Best-effort drop of a placeholder registered by ``read_record_batches``.

    duckdb registers the StreamCache as a VIEW, so ``drop_table`` always
    raises a kind-mismatch there (``force=True`` does not suppress it) and
    ``drop_view`` is the real path; pandas raises KeyError on a missing
    table even with ``force=True``, hence the inner suppression --
    ``force=True`` alone cannot make this idempotent across backends.
    """
    try:
        con.drop_table(table_name, force=True)
    except Exception:
        try:
            con.drop_view(table_name, force=True)
        except Exception:
            logger.debug(
                "drop_placeholder: neither drop_table nor drop_view succeeded",
                table_name=table_name,
                backend=con.name,
                exc_info=True,
            )


class _ScopeEntry(NamedTuple):
    kind: str
    label: str
    cleanup: Callable[[], None]

    def safe_cleanup(self) -> None:
        try:
            self.cleanup()
        except Exception:
            logger.warning(
                "scope cleanup failed",
                kind=self.kind,
                label=self.label,
                exc_info=True,
            )


class RemoteTableScope:
    """Owns resources materialized while replacing RemoteTable nodes.

    Each ``adopt_*`` method appends to a typed list.  ``close`` tears down
    in dependency order (tables -> caches -> readers), LIFO within each
    category, and is idempotent.

    Not thread-safe: all adopt/close calls must happen on a single thread.
    """

    def __init__(self) -> None:
        self._readers: list[_ScopeEntry] = []
        self._caches: list[_ScopeEntry] = []
        self._tables: list[_ScopeEntry] = []
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def table_count(self) -> int:
        return len(self._tables)

    @property
    def table_names(self) -> list[str]:
        return [e.label for e in self._tables]

    @property
    def reader_count(self) -> int:
        return len(self._readers)

    @property
    def cache_count(self) -> int:
        return len(self._caches)

    def adopt_reader(self, reader: pa.RecordBatchReader) -> pa.RecordBatchReader:
        self._readers.append(_ScopeEntry("reader", type(reader).__name__, reader.close))
        return reader

    def adopt_cache(self, cache: StreamCache) -> StreamCache:
        self._caches.append(_ScopeEntry("cache", type(cache).__name__, cache.close))
        return cache

    def adopt_table(self, con: BaseBackend, table_name: str) -> str:
        self._tables.append(
            _ScopeEntry(
                "table",
                table_name,
                functools.partial(drop_placeholder, con, table_name),
            )
        )
        return table_name

    def close(self) -> None:
        """Release everything the scope owns; idempotent, never raises."""
        if self._closed:
            return
        self._closed = True
        for entry in reversed(self._tables):
            entry.safe_cleanup()
        for entry in reversed(self._caches):
            entry.safe_cleanup()
        for entry in reversed(self._readers):
            entry.safe_cleanup()

    def __enter__(self) -> RemoteTableScope:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def bind_scope_to_reader(
    scope: RemoteTableScope,
    reader: pa.RecordBatchReader,
) -> pa.RecordBatchReader:
    """Tie scope cleanup to reader exhaustion; consumes the scope.

    Cleanup must stay deferred on the streaming path: duckdb/datafusion
    scan the placeholder when the result reader is drained, and dropping
    the duckdb view mid-drain silently truncates results.  The wrapping
    generator's ``finally`` fires on full drain; the ``weakref.finalize``
    backstop covers readers abandoned before the first read (a
    never-started generator never runs its ``finally``).  Note that on
    pyarrow 21 ``reader.close()`` alone does not finalize the wrapped
    generator -- cleanup then fires when the last reference drops.
    """

    def gen():
        try:
            yield from reader
        finally:
            scope.close()

    g = gen()
    out = pa.RecordBatchReader.from_batches(reader.schema, g)

    def _cleanup() -> None:
        g.close()
        scope.close()

    weakref.finalize(out, _cleanup)
    return out


@tracer.start_as_current_span("register_and_transform_remote_tables")
def register_and_transform_remote_tables(
    expr: Expr, **kwargs: Any
) -> tuple[Expr, RemoteTableScope]:
    scope = RemoteTableScope()
    # ``replacer``'s ``kwargs`` parameter (node-recreate args) shadows the
    # outer ``**kwargs``; capture the through-kwargs here so they reach
    # ``read_record_batches`` (e.g. Snowflake's ``database=(catalog, db)``).
    read_kwargs = kwargs

    op = expr.op()
    reader_counts = count_remote_table_readers(expr)

    def replacer(node, kwargs):
        if isinstance(node, RemoteTable):
            remote_expr = node.remote_expr
            # Cast batches to the logical schema before entering
            # StreamCache. The raw reader may carry extra physical columns
            # (e.g. row_number) or mismatched types (large_utf8 vs utf8);
            # StreamCache replays through the C Data Interface using the
            # declared schema, so uncasted data silently corrupts reads.
            raw_reader = scope.adopt_reader(remote_expr.to_pyarrow_batches())
            logical_schema = node.schema.to_pyarrow()
            casting_reader = pa.RecordBatchReader.from_batches(
                logical_schema,
                (
                    batch.select(logical_schema.names).cast(logical_schema)
                    for batch in raw_reader
                ),
            )
            cache = scope.adopt_cache(
                StreamCache(
                    casting_reader,
                    max_readers=reader_counts.get(node),
                )
            )
            # adopt before registering: a partial failure inside
            # read_record_batches (register, then raise) can't strand the
            # placeholder, and drop_placeholder is a no-op if it never
            # registered
            table_name = scope.adopt_table(node.source, gen_name())
            result = node.source.read_record_batches(
                cache, table_name=table_name, schema=logical_schema, **read_kwargs
            )
            return result.op()

        if kwargs:
            node = node.__recreate__(kwargs)

        return node

    # Intentionally op.replace, not replace_nodes: mark_remote_table has side effects
    # that must not descend into opaque sub-exprs (e.g. ExprScalarUDF.computed_kwargs_expr)
    try:
        expr = op.replace(replacer).to_expr()
    except Exception:
        scope.close()
        raise
    if scope.table_count:
        trace.get_current_span().add_event(
            "remote_table.replace", {"remote_table.count": scope.table_count}
        )
    return expr, scope


def prepare_create_table_from_expr(
    con: BaseBackend, expr: Expr, **kwargs: Any
) -> tuple[Expr, RemoteTableScope]:
    """Transform ``expr`` for eager-ingest backends and return the scope.

    The caller owns the returned scope and must close it after the
    transformed expr has been consumed (e.g. after CTAS execution).
    Only safe for backends whose ``read_record_batches`` eagerly ingests
    the stream synchronously (Snowflake, Postgres via ADBC).
    """
    from xorq.expr.api import _transform_expr  # noqa: PLC0415

    expr_backend = expr._find_backend()  # xorq-style: disable=protected-access
    if expr_backend != con:
        raise ValueError(f"expr backend must be {con}, is {expr_backend}")
    return _transform_expr(expr, **kwargs)
