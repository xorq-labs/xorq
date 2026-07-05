from __future__ import annotations

import functools
import weakref
from collections import Counter
from typing import Any, Callable, NamedTuple

import pyarrow as pa
from batchcorder import StreamCache

from xorq.common.compat import raise_collected_errors
from xorq.common.utils.logging_utils import get_logger
from xorq.expr.enums import Traversal
from xorq.expr.relations import RemoteTable, gen_name
from xorq.expr.transform import TransformCtx, TransformPass, apply_pass
from xorq.vendor.ibis import Expr
from xorq.vendor.ibis.backends import BaseBackend
from xorq.vendor.ibis.expr import operations as ops
from xorq.writes import DrainingIterator


logger = get_logger(__name__)


def project_and_cast_reader(
    reader: pa.RecordBatchReader, logical_schema: pa.Schema
) -> pa.RecordBatchReader:
    """Project to the logical columns and cast types, before any StreamCache.

    A raw remote reader may carry extra physical columns (e.g. row_number) or
    mismatched types (large_utf8 vs utf8). The cache must hold exactly the
    logical columns: ``StreamCache`` replays through the C Data Interface using
    the declared schema (so uncasted data silently corrupts reads), and the
    backend casting wrapper (``StreamCache.cast``) only retypes -- it cannot
    drop columns. Projecting here, before the cache, keeps that single
    invariant and lets every backend register the cache with a type-only cast.

    When the reader already declares the logical schema (the common case --
    ~95% of remote scans), the per-batch select+cast is a no-op, so the reader
    is returned unwrapped to avoid a redundant Python-level batch pass.
    """
    if reader.schema.equals(logical_schema):
        return reader
    return pa.RecordBatchReader.from_batches(
        logical_schema,
        (batch.select(logical_schema.names).cast(logical_schema) for batch in reader),
    )


def count_remote_table_readers(expr: Expr) -> dict[RemoteTable, int]:
    """Best-effort count of how many times each ``RemoteTable`` is scanned.

    The scan count is not visible in the expression graph: a single graph
    reference can compile to several physical table scans (e.g. a self-join over
    one ``RemoteTable``). Each ``RemoteTable`` is rewritten to a placeholder
    table under a freshly generated, unique name, the placeholder expression is
    compiled to a sqlglot AST, and the ``Table`` nodes bearing that name are
    counted. Counting ``Table`` AST nodes (rather than substrings of the
    rendered SQL) ignores column references, aliases, and string literals, so
    even a short or shared user-supplied table name cannot inflate the count.

    This is a *lower bound*, not the exact physical scan count: it sees only the
    scans the compiled SQL spells out, and cannot see re-scans the backend's
    optimiser introduces below the SQL layer. The known gap is a
    ``PARTITION BY``-only aggregate window on DuckDB, lowered to a ``GROUP BY``
    self-join that scans its input twice while the AST shows one ``Table`` node.

    The counts become each ``StreamCache``'s ``max_readers``, which both bounds
    memory (batches are evicted once all readers advance past them) and is a
    hard cap (a reader beyond it raises ``ValueError``). Because it is a hard
    cap, an undercount is a correctness bug for the shapes it misses, not just a
    missed optimisation. Each count is floored at 1 — a bare ``RemoteTable`` is
    still scanned once, and ``max_readers=0`` would forbid the reader that is in
    fact created.

    Returns an empty mapping when no SQL AST can be produced (e.g. a non-SQL
    backend); the caller then builds an unbounded cache — always safe, just
    without eviction. Omitting the count is safe; deriving one too low is not.
    See ADR-0013 ("Counting accuracy").
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
    """Owns resources materialized while replacing ``RemoteTable`` nodes:
    upstream readers, ``StreamCache``s, placeholder tables, and write-through
    drains (folded in from tee transforms).

    ``close()`` is idempotent and tears down in dependency order: drains first
    (close-all, then join-all -- they feed the tables), then tables, caches, and
    readers, LIFO within each. Table/cache/reader failures are logged (they only
    leak a resource); drain-join failures are raised only with
    ``raise_drain_errors=True`` (set once the result is consumed), else logged.

    ``adopt_*`` must run on the construction thread. ``close()`` may fire from a
    drain worker or a ``weakref.finalize`` GC thread on the streaming path; the
    ``_closed`` guard keeps that safe. Concurrent ``close()`` is not supported,
    but the paths are serialized so it does not arise.
    """

    def __init__(self) -> None:
        self._readers: list[_ScopeEntry] = []
        self._caches: list[_ScopeEntry] = []
        self._tables: list[_ScopeEntry] = []
        self._drains: list[DrainingIterator] = []
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

    @property
    def drain_count(self) -> int:
        return len(self._drains)

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

    def adopt_drain(self, drain: DrainingIterator) -> DrainingIterator:
        self._drains.append(drain)
        return drain

    def _drain_step(self) -> list[BaseException]:
        # Close-then-join in two passes (not LIFO _ScopeEntry cleanup): the
        # write-through drains feed the placeholder tables, so every writer
        # must finish consuming its stream before any table is dropped.
        # Close all first so the writers stop pulling, then join to surface
        # errors and ensure the side-effect writes have landed. Returns the
        # collected failures; the caller decides whether to raise (consumer
        # finished successfully) or log (teardown after failure/abandon).
        errors: list[BaseException] = []
        for drain in self._drains:
            try:
                drain.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        for drain in self._drains:
            try:
                drain.join()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        self._drains = []
        return errors

    def close(self, *, raise_drain_errors: bool = False) -> None:
        """Release everything the scope owns; idempotent.

        Drain (write-through) failures are correctness signals: a failed join
        means a tee/WAP write never landed. They are surfaced by raising only
        when ``raise_drain_errors`` is set -- the caller passes it once the
        result has been consumed successfully (eager execute, or the result
        reader fully drained). On the teardown-after-failure and finalizer/GC
        paths raising is unsafe (it masks the real error or fires during
        interpreter shutdown), so they are logged instead. Table/cache/reader
        teardown is best-effort either way: a failure there only leaks a
        resource, never corrupts a result, so it is always swallowed-and-logged.
        """
        if self._closed:
            return
        self._closed = True
        drain_errors = self._drain_step()
        for entry in reversed(self._tables):
            entry.safe_cleanup()
        for entry in reversed(self._caches):
            entry.safe_cleanup()
        for entry in reversed(self._readers):
            entry.safe_cleanup()
        if raise_drain_errors:
            raise_collected_errors("tee drain failures", drain_errors)
        else:
            for exc in drain_errors:
                logger.warning("scope drain cleanup failed", exc_info=exc)

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
            # Deferred path: close runs inside this generator's teardown (or
            # the weakref.finalize backstop), where raising is unsafe -- a
            # drain join here also races the write-through generator still in
            # the reader chain. Always swallow-and-log; the eager call sites
            # (remote_table_scope) surface drain failures instead.
            scope.close()

    g = gen()
    out = pa.RecordBatchReader.from_batches(reader.schema, g)

    def _cleanup() -> None:
        g.close()
        scope.close()

    weakref.finalize(out, _cleanup)
    return out


def _remote_replacer(
    expr: Expr, scope: RemoteTableScope, read_kwargs: dict
) -> Callable:
    """Build the per-node replacer that swaps each `RemoteTable` for a placeholder
    table fed by a cached, cast stream, adopting every resource into the
    caller-owned ``scope``.

    ``read_kwargs`` are the through-kwargs that must reach ``read_record_batches``
    (e.g. Snowflake's ``database=(catalog, db)``). The replacer never closes
    ``scope`` -- teardown stays with the caller that created it.
    """
    reader_counts = count_remote_table_readers(expr)

    def replacer(node, kwargs):
        if isinstance(node, RemoteTable):
            remote_expr = node.remote_expr
            raw_reader = scope.adopt_reader(remote_expr.to_pyarrow_batches())
            logical_schema = node.schema.to_pyarrow()
            casting_reader = project_and_cast_reader(raw_reader, logical_schema)
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
            # No ``schema=`` is passed: ``project_and_cast_reader`` already cast
            # the stream to ``logical_schema`` before the cache, so the cache
            # already declares it. Backends that genuinely use a schema
            # (xorq_datafusion, pyiceberg) default to the source's own schema.
            result = node.source.read_record_batches(
                cache, table_name=table_name, **read_kwargs
            )
            return result.op()

        # ``replacer``'s ``kwargs`` (node-recreate args) shadows the builder's
        # ``read_kwargs``; those are captured above so they reach the backend.
        if kwargs:
            node = node.__recreate__(kwargs)

        return node

    return replacer


# BOUNDARY (op.replace), not DESCEND: mark_remote_table has side effects that must
# not descend into opaque sub-exprs (e.g. ExprScalarUDF.computed_kwargs_expr).
# Runs after tee so a tee placeholder registered earlier is torn down by the same
# shared scope if this pass fails.
REMOTE_PASS = TransformPass(
    name="remote",
    traversal=Traversal.BOUNDARY,
    build=lambda expr, ctx: _remote_replacer(expr, ctx.scope, ctx.through_kwargs),
    produces_resources=True,
    after=("tee",),
)


def register_and_transform_remote_tables_into(
    expr: Expr, scope: RemoteTableScope, **kwargs: Any
) -> Expr:
    """Apply :data:`REMOTE_PASS` against the caller-owned ``scope``.

    Thin adapter over the shared driver so the standalone wrapper and the
    ``_transform_expr`` pipeline apply the remote pass identically. The scope is
    threaded explicitly by the caller; this never closes it -- ownership and
    teardown stay with the caller that created it.
    """
    return apply_pass(
        REMOTE_PASS, expr, TransformCtx(scope=scope, through_kwargs=kwargs)
    )


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
