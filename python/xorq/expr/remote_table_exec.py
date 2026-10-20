from __future__ import annotations

import functools
import weakref
from collections import Counter
from typing import Any, Callable, NamedTuple

import pyarrow as pa
from batchcorder import StreamCache

from xorq.common.compat import raise_collected_errors
from xorq.common.utils.logging_utils import get_logger
from xorq.common.utils.otel_utils import get_current_span
from xorq.expr.enums import Traversal
from xorq.expr.relations import RemoteTable, gen_name
from xorq.expr.transform import Replacer, TransformCtx, TransformPass
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


# dialects whose planner inlines a CTE at every reference, scanning its body
# once per reference; the others (duckdb) materialise it and scan it once
CTE_INLINING_DIALECTS = frozenset({"DataFusion"})
# dialects whose optimiser re-scans a PARTITION BY-only window's input below
# the SQL layer (a GROUP BY self-join), where counting Table nodes undercounts
RESCANNING_DIALECTS = frozenset({"DuckDB"})


def count_remote_table_readers(expr: Expr) -> dict[RemoteTable, int]:
    """Best-effort count of how many times each ``RemoteTable`` is scanned.

    The scan count is not visible in the expression graph: a single graph
    reference can compile to several physical table scans (e.g. a self-join over
    one ``RemoteTable``). Each ``RemoteTable`` is rewritten to a placeholder
    table under a freshly generated, unique name, the placeholder expression is
    compiled to a sqlglot AST, and the ``Table`` nodes bearing that name are
    counted. On a dialect that inlines a CTE at every reference (datafusion),
    each CTE's body is weighted by its reference count, since a window or join
    hoisted into a CTE and used twice scans its source twice; a dialect that
    materialises CTEs (duckdb) scans the body once, and an overcount there
    would keep every batch buffered waiting for a reader that never comes
    (eviction starts only once ``max_readers`` readers exist). Counting
    ``Table`` AST nodes (rather than substrings of the rendered SQL) ignores
    column references, aliases, and string literals, so even a short or shared
    user-supplied table name cannot inflate the count.

    This is a *lower bound*, not the exact physical scan count: it sees only the
    scans the compiled SQL spells out, and cannot see re-scans the backend's
    optimiser introduces below the SQL layer. The known such shape, a
    ``PARTITION BY``-only window on DuckDB (lowered to a ``GROUP BY`` self-join
    that scans its input twice while the AST shows one ``Table`` node), yields
    no count for the tables that window reads, so their caches are built
    unbounded (see ``tables_under_rescans`` and ``RESCANNING_DIALECTS``).

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
        provider = placeholder._find_backend(  # xorq-style: disable=protected-access
            use_default=True
        )
        compiler = getattr(provider, "compiler", None)
        if compiler is None:
            return {}
        out = compiler.to_sqlglot(placeholder.unbind())
    except Exception:
        return {}

    # xorq's datafusion compiler and the vendored one each carry their own
    # `DataFusion` dialect class, so the dialect is matched by name
    dialect = compiler.dialect.__name__
    inlines_ctes = dialect in CTE_INLINING_DIALECTS
    queries = out if isinstance(out, list) else [out]
    rescanned = (
        set().union(*map(tables_under_rescans, queries))
        if dialect in RESCANNING_DIALECTS
        else set()
    )
    counts = Counter()
    for query in queries:
        counts.update(_count_table_scans(query, inlines_ctes))
    return {
        node: max(1, counts[name])
        for node, name in sentinels.items()
        if name not in rescanned
    }


def tables_under_rescans(query) -> set[str]:
    """The names of the tables read by a shape duckdb re-scans below the SQL
    layer, so that counting their ``Table`` nodes undercounts: a window with
    a ``PARTITION BY`` and no ``ORDER BY``, which duckdb lowers to a
    ``GROUP BY`` self-join for an aggregate. A table is read by the window
    when it sits under the window's enclosing ``SELECT``, directly or through
    a CTE reference. The rule over-approximates: any such window, aggregate or
    ranking, over any input, counts. Which of those duckdb re-scans is its
    optimiser's rule, not the AST's; the cost of a false positive is eviction,
    not correctness."""
    import sqlglot as sg  # noqa: PLC0415

    def read_by(node) -> set[str]:
        # a CTE reference stands for the tables its body reads; each CTE is
        # resolved once, in definition order, since a CTE can only reference
        # the ones before it (a name not yet seen is a plain table)
        return set().union(
            *(
                sources.get(table.name, {table.name})
                for table in node.find_all(sg.exp.Table)
            )
        )

    sources: dict[str, set[str]] = {}
    for cte in query.find_all(sg.exp.CTE):
        sources[cte.alias_or_name] = read_by(cte.this)

    return set().union(
        *(
            read_by(select)
            for window in query.find_all(sg.exp.Window)
            if window.args.get("partition_by") and not window.args.get("order")
            if (select := window.find_ancestor(sg.exp.Select)) is not None
        )
    )


def _count_table_scans(query, inlines_ctes: bool) -> Counter:
    """Count ``Table`` nodes by name; with ``inlines_ctes``, expand each CTE by
    its reference count, else count each CTE body once.

    The compilers hoist windows and joins into CTEs, and a CTE referenced by
    both sides of a self cross join is inlined by datafusion (duckdb
    materialises it): its body is scanned once per reference. The walk only
    descends (the compiled AST shares nodes, so a node's parent pointer does
    not say where it was reached from) and stops at each scope: a CTE body, or
    a nested query with a WITH of its own (an embedded raw-SQL query, whose CTE
    names may shadow the generated ones). Nested queries are folded in as
    scanned once; CTE bodies last-to-first, so a later CTE's references to an
    earlier one are already in ``counts`` when the earlier one's body is
    weighted.
    """
    import sqlglot as sg  # noqa: PLC0415

    def is_scope(node):
        return isinstance(node, sg.exp.CTE) or (
            node is not query and isinstance(node, sg.exp.Query) and node.ctes
        )

    counts = Counter()
    ctes = []
    for node in query.walk(prune=is_scope):
        if isinstance(node, sg.exp.Table):
            counts[node.name] += 1
        elif isinstance(node, sg.exp.CTE):
            ctes.append(node)
        elif is_scope(node):
            counts.update(_count_table_scans(node, inlines_ctes))
    for cte in reversed(ctes):
        weight = counts.pop(cte.alias_or_name, 0)
        if not inlines_ctes:
            weight = min(weight, 1)
        for name, n in _count_table_scans(cte.this, inlines_ctes).items():
            counts[name] += weight * n
    return counts


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


def _make_remote_replacer(
    expr: Expr, scope: RemoteTableScope, read_record_batches_kwargs: dict
) -> Replacer:
    """Build the per-node replacer that swaps each `RemoteTable` for a placeholder
    table fed by a cached, cast stream, adopting every resource into the
    caller-owned ``scope``.

    ``read_record_batches_kwargs`` are forwarded to ``read_record_batches``
    (e.g. Snowflake's ``database=(catalog, db)``). The replacer never closes
    ``scope`` -- teardown stays with the caller that created it.
    """
    # A required pre-pass, not a foldable extra walk: ``max_readers`` must be
    # known when each ``StreamCache`` is constructed below (it is immutable after
    # -- backed by a Rust ``_PyStreamCache``), and the count needs a *compiled*
    # SQL AST because one graph reference can become several physical scans (a
    # self-join over one RemoteTable). So it cannot be merged into the
    # materializing walk. REMOTE_PASS's ``when`` keeps it (and this walk) from
    # running at all when no RemoteTable is reachable at this boundary.
    reader_counts = count_remote_table_readers(expr)

    # Emit the historical ``remote_table.replace`` event on the active
    # ``transform.remote`` span so trace/dashboards keyed on the RemoteTable
    # fan-out keep firing. Count the boundary RemoteTables directly (== the
    # eventual ``scope.table_count``, matching this pass's own ``find``-based
    # ``when`` guard) rather than ``len(reader_counts)``: the count returns ``{}``
    # for a compiler-less backend or a failed SQL compile even when RemoteTables
    # are present, so gating on it would silently drop the event in exactly that
    # case (see gh-2160).
    remote_table_count = len(expr.op().find(RemoteTable))
    if remote_table_count:
        get_current_span().add_event(
            "remote_table.replace", {"remote_table.count": remote_table_count}
        )

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
                cache, table_name=table_name, **read_record_batches_kwargs
            )
            return result.op()

        # ``replacer``'s ``kwargs`` (node-recreate args) shadows the builder's
        # ``read_record_batches_kwargs``; captured above so they reach the backend.
        if kwargs:
            node = node.__recreate__(kwargs)

        return node

    return replacer


def _has_boundary_remote_table(expr: Expr, ctx: TransformCtx) -> bool:
    """Whether a ``RemoteTable`` is reachable at *this* execution boundary.

    Uses ``op.find`` (not the opaque-descending ``walk_nodes``) so it matches the
    pass's own ``op.replace`` reach exactly: a RemoteTable buried inside an opaque
    sub-expr (e.g. ``CachedNode.parent``) is transformed when that node executes,
    not here, so the pass -- and its ``count_remote_table_readers`` pre-walk plus
    SQL compile -- would only no-op on it. Skipping keeps both walks off every
    expr that has no boundary RemoteTable (the common case, and any remote nested
    only inside a cache).
    """
    return bool(expr.op().find(RemoteTable))


# BOUNDARY: mark_remote_table has side effects, so it must not descend into opaque
# sub-exprs (e.g. ExprScalarUDF.computed_kwargs_expr) -- see Traversal. Runs after
# tee so a tee placeholder registered earlier is torn down by the same shared scope
# if this pass fails.
REMOTE_PASS = TransformPass(
    name="remote",
    traversal=Traversal.BOUNDARY,
    build=lambda expr, ctx: _make_remote_replacer(
        expr, ctx.scope, ctx.read_record_batches_kwargs
    ),
    produces_resources=True,
    when=_has_boundary_remote_table,
    after=("tee",),
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
