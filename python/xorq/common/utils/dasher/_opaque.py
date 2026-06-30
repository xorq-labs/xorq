"""Opaque-leaf placeholder rewriting and Expr / ScalarUDF normalizers.

Used by both the global ``HASHER`` (via the registered ``Expr`` rule) and
``SnapshotStrategy`` (which sets ``_current_hasher`` so transitive tokenize
calls inside ``_parent_token`` propagate the snapshot-flavored rules instead
of falling back to the global, data-sensitive HASHER).
"""

from __future__ import annotations

import contextlib
import contextvars
import itertools
import logging
from typing import TYPE_CHECKING

import xxhash
from xorq_dasher.rules.expr import normalize_inmemorytable


if TYPE_CHECKING:
    from typing import Literal, TypedDict

    from xorq.vendor.ibis.common.collections import FrozenOrderedDict
    from xorq.vendor.ibis.expr.operations.core import Node
    from xorq.vendor.ibis.expr.operations.relations import (
        DatabaseTable,
        InMemoryTable,
    )
    from xorq.vendor.ibis.expr.operations.udf import AggUDF, ScalarUDF
    from xorq.vendor.ibis.expr.relations import Read
    from xorq.vendor.ibis.expr.schema import Schema
    from xorq.vendor.ibis.expr.types.core import Expr

    SlotKind = Literal["Read", "DatabaseTable", "InMemoryTable"]

    class SlotDict(TypedDict):
        index: int
        kind: SlotKind
        name: str
        hash: str

    from xorq.expr.relations import CacheTag, HashingTag, TeeNode

    class ExprMetadata(TypedDict):
        version: Literal[4]
        structural_hash: str
        slots: list[SlotDict]


logger = logging.getLogger(__name__)


class _MissingSentinel:
    def __dasher_tokenize__(self):
        return ("_MISSING",)

    def __repr__(self):
        return "_MISSING"


_MISSING = _MissingSentinel()

# Per-outer-call memo for ``_parent_token``.  Cross-engine nested expressions
# (``RemoteTable`` containing a ``RemoteTable`` containing …) trigger a fresh
# ``hasher.tokenize`` of every opaque parent at every level
_parent_token_memo: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "_xorq_parent_token_memo", default=None
)

# Per-outer-call memo for ``_normalize_expr_xorq`` keyed by ``op``.
_expr_normalize_memo: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "_xorq_expr_normalize_memo", default=None
)

# When True, TeeNode writer identity is folded into the structural hash.
# Set by ``get_expr_hash`` (build hash path) so that different writers produce
# different build artifacts, while the cache hash path leaves this False.
_include_tee_nodes: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_xorq_include_tee_nodes", default=False
)


@contextlib.contextmanager
def include_tee_nodes() -> contextlib.AbstractContextManager[None]:
    token = _include_tee_nodes.set(True)
    try:
        yield
    finally:
        _include_tee_nodes.reset(token)


def _rename_unbound_xorq(op: Node, prefix: str = "static") -> Node:
    """Rewrite UnboundTable nodes to sequential placeholder names.

    Equivalent of ``xorq_dasher.rules.expr._rename_unbound`` but with a correct
    op.replace callback signature: dasher 0.1.0's version uses ``**kwargs``
    which captures nothing when ibis passes the rewritten-children dict as a
    positional, then crashes when ``__recreate__({})`` is called on ops with
    required fields (e.g. ``Field`` needs ``rel`` and ``name``).
    """

    from xorq.vendor.ibis.expr.operations.relations import UnboundTable  # noqa: PLC0415

    count = itertools.count()

    def rename(node, _kwargs=None, **_kw):
        if isinstance(node, UnboundTable):
            return node.copy(name=f"{prefix}-{next(count)}")
        if _kwargs:
            return node.__recreate__(_kwargs)
        return node

    return op.replace(rename)


def _stable_opaque_name(prefix: str, *parts: str | Schema | FrozenOrderedDict) -> str:
    """Build a deterministic placeholder name from xxhash of structural parts.

    xorq_dasher 0.1.0's ``_opaque_to_placeholder`` uses ``id(node)`` for some
    leaf names, which breaks across catalog reloads (different Python object
    identities for semantically-identical Reads). This helper keys on a
    content-stable hash of the supplied parts instead.
    """
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return f"{prefix}-{xxhash.xxh128(payload).hexdigest()[:16]}"


def _parent_token(thing: Expr | Node | _MissingSentinel) -> str:
    """Tokenize an opaque sub-expression's parent / inner expr structurally.

    Used to fold the inner expression's identity into the placeholder name so
    two opaque wrappers with the same schema/cache-type/etc. but different
    inner expressions do not collide. Accepts either Op or Expr.

    Uses ``_current_hasher`` when set (so snapshot tokenize propagates its
    data-blind rules into recursive parent normalization); otherwise falls
    back to the global HASHER.

    Memoized per outer call via :data:`_parent_token_memo` — without this,
    deep cross-engine ``into_backend`` chains pay O(depth²) re-tokenization
    of shared parent sub-expressions.  Key includes ``id(hasher)`` so
    snapshot-flavored and default-flavored calls don't share entries.
    """
    # Lazy: HASHER is constructed in ``__init__`` *after* this module is
    # imported, so a top-level import here would create a bootstrap cycle.
    from xorq.common.utils.dasher import HASHER, _current_hasher  # noqa: PLC0415

    memo = _parent_token_memo.get()
    if hasattr(thing, "to_expr") and not hasattr(thing, "op"):
        thing = thing.to_expr()
    hasher = _current_hasher.get() or HASHER
    op = thing.op() if hasattr(thing, "op") else thing
    key = (id(hasher), op)
    if memo is not None and key in memo:
        return memo[key]
    try:
        tok = hasher.tokenize(thing)
    except RecursionError:
        logger.warning(
            "RecursionError tokenizing %r in _parent_token; falling back "
            "to type+schema hash.  Investigate the op graph for cycles "
            "or unbounded nesting.",
            type(thing).__name__,
        )
        typ = type(thing)
        fallback_op = thing.op() if hasattr(thing, "op") else thing
        schema = getattr(thing, "schema", getattr(fallback_op, "schema", ""))
        payload = f"{typ.__module__}.{typ.__qualname__}|{schema}"
        tok = xxhash.xxh128(payload.encode("utf-8")).hexdigest()
    if memo is not None:
        memo[key] = tok
    return tok


def _xorq_opaque_to_placeholder(node, _kwargs=None, **_kw):
    """Replace opaque leaf nodes with UnboundTable placeholders.

    Mirrors xorq_dasher.rules.expr._opaque_to_placeholder but
    (a) uses content-stable hashes instead of ``id()`` so tokenize is
    reproducible across catalog reloads, and
    (b) folds the *parent/inner* expression's structural token into each
    placeholder name so wrappers with identical schema but distinct inner
    expressions do not collide.

    Callable from both ibis ``op.replace`` (positional ``(node, kwargs)``)
    and xorq's ``replace_nodes`` (same shape); for non-opaque nodes with
    rewritten children, ``_kwargs`` is the children-dict and we recreate.
    """
    import xorq.expr.operations as xops  # noqa: PLC0415
    from xorq.expr import api  # noqa: PLC0415
    from xorq.expr.relations import (  # noqa: PLC0415
        CachedNode,
        CacheTag,
        FlightExpr,
        FlightUDXF,
        Read,
        RemoteTable,
        cache_tag_identity_parts,
    )

    match node:
        case CacheTag():
            # A pinned read is a leaf: its placeholder is keyed on the cache key
            # (the frozen read's table name), which is base_path-independent and
            # needs no source. Returning a placeholder here also stops the
            # rewrite from descending ``parent``'s absolute path or ``uncached``'s
            # discarded upstream into the SQL. See CacheTag.__dasher_tokenize__.
            # Identity fields shared via cache_tag_identity_parts; the "cachetag"
            # prefix is this layer's own (distinct from the "cache-tag" token,
            # since it lands in the structural SQL).
            name = _stable_opaque_name("cachetag", *cache_tag_identity_parts(node))
        case CachedNode():
            name = _stable_opaque_name(
                "cached",
                node.schema,
                type(node.cache).__name__,
                _parent_token(node.parent),
            )
        case Read():
            read_kwargs = dict(node.read_kwargs)
            rp = read_kwargs.get("read_path")
            anchor = rp if rp is not None else read_kwargs["hash_path"]
            name = _stable_opaque_name("read", node.schema, anchor)
        case RemoteTable():
            name = _stable_opaque_name(
                "remote",
                node.schema,
                _parent_token(node.remote_expr),
                getattr(node.source, "name", ""),
            )
        case FlightExpr():
            # unbound_expr names are user-chosen and may differ between two
            # FlightExprs that should hash identically (see
            # test_flight_expr_name_doesnt_matter). Canonicalize via
            # _rename_unbound_xorq before folding into the placeholder name.
            name = _stable_opaque_name(
                "flight-expr",
                node.schema,
                _parent_token(node.input_expr),
                _parent_token(_rename_unbound_xorq(node.unbound_expr.op()).to_expr()),
            )
        case FlightUDXF():
            # See ``_dispatch_databasetable``: fold type-identity alongside
            # ``exchange_f`` so two UDXF classes that both lack
            # ``exchange_f`` don't collide.
            name = _stable_opaque_name(
                "flight-udxf",
                node.schema,
                _parent_token(node.input_expr),
                type(node.udxf).__qualname__,
                _parent_token(getattr(node.udxf, "exchange_f", _MISSING)),
            )
        case xops.NamedScalarParameter():
            # Replace with a typed NULL so SQL compilation works without a
            # translation rule for NamedScalarParameter.  A bare Literal
            # (no .name()) is required — Project forbids Alias values.
            # Parameter identity (label, dtype, default) is folded into the
            # structural hash separately via _collect_param_anchors.
            return api.literal(value=None, type=node.dtype).op()
        case _:
            if _kwargs:
                return node.__recreate__(_kwargs)
            return node
    return api.table(node.schema, name=name).op()


def _normalize_computed_kwargs_expr(cke):
    """Content-stable, structural-only normalization of a ``computed_kwargs_expr``.

    The default Expr rule routes through SQL compilation, which embeds the
    auto-generated counter-suffixed dynamic class names that ibis assigns to
    each ScalarUDF (``_inner_fit_0`` vs ``_inner_fit_3``). Two pipelines built
    in different import orders would therefore tokenize differently — a
    correctness regression for ML fit-then-predict flows.

    Per ADR-0010, this helper is **data-free**: data identity for any leaf
    reachable through ``ExprScalarUDF.computed_kwargs_expr`` flows into the
    outer expression's leaves (via the recursive HASHER tokenization of the
    enclosing expression), so here we contribute only structural shape
    (schemas, op types, UDF function identity, cache class).

    Decomposition mirrors the legacy
    ``dask_normalize_expr._normalize_computed_kwargs_expr`` (deleted with the
    ``dask_normalize/`` package); ScalarUDF normalization recurses through
    this helper so the structural-only contract holds transitively across
    nested ``ExprScalarUDF`` steps.
    """
    from xorq.expr.relations import CachedNode, Read  # noqa: PLC0415
    from xorq.vendor.ibis.expr.operations import relations as _rel  # noqa: PLC0415
    from xorq.vendor.ibis.expr.operations.udf import AggUDF, ScalarUDF  # noqa: PLC0415
    from xorq.vendor.ibis.expr.types import Table  # noqa: PLC0415

    op = cke.op()
    mems = op.find(_rel.InMemoryTable)
    agg_udfs = op.find(AggUDF)
    scalar_udfs = op.find(ScalarUDF)
    reads = op.find(Read)
    cached = op.find(CachedNode)

    # Strip path identity from read_kwargs — the path's data identity lives
    # in the outer expression's data leaves, not here.
    _path_keys = ("hash_path", "read_path")
    read_structural = tuple(
        (
            r.schema,
            r.method_name,
            tuple((k, v) for k, v in r.read_kwargs if k not in _path_keys),
            r.normalize_method,
        )
        for r in reads
    )
    return (
        "normalize_computed_kwargs_expr",
        cke.schema() if isinstance(cke, Table) else cke.type(),
        tuple(m.schema for m in mems),
        agg_udfs,
        scalar_udfs,
        read_structural,
        tuple((c.schema, type(c.cache).__name__) for c in cached),
    )


def _normalize_scalar_udf_xorq(udf):
    """ScalarUDF normalizer that routes ``computed_kwargs_expr`` through the
    data-free :func:`_normalize_computed_kwargs_expr` helper.

    Dasher 0.1.0's rule returns the raw ``computed_kwargs_expr`` and lets
    recursion through the Expr rule normalize it via SQL — which embeds the
    counter-suffixed dynamic class name (see :func:`_normalize_computed_kwargs_expr`
    for context).
    """
    typs = tuple(arg.dtype for arg in udf.args)
    computed_kwargs_expr = udf.__config__.get("computed_kwargs_expr")
    cke_token = (
        _normalize_computed_kwargs_expr(computed_kwargs_expr)
        if computed_kwargs_expr is not None
        else None
    )
    return (
        "ibis.ScalarUDF",
        typs,
        udf.dtype,
        udf.__func__,
        cke_token,
    )


def _normalize_expr_xorq(expr):
    """Deterministic Expr normalizer; replaces dasher's id()-based version.

    Memoized per outer call via :data:`_expr_normalize_memo` keyed by ``op``
    (the same underlying op tree always produces the same normalization).
    """
    op = expr.op()
    memo = _expr_normalize_memo.get()
    if memo is not None and op in memo:
        return memo[op]
    result = _normalize_expr_xorq_impl(expr, op)
    if memo is not None:
        memo[op] = result
    return result


def _collect_param_anchors(op: Node) -> tuple[str, ...]:
    """Return a stable tuple of per-parameter identity strings.

    Walks the op graph for NamedScalarParameter nodes and produces a
    content-stable anchor for each occurrence (preserving graph order).
    These anchors are folded into the structural hash so that two
    same-dtype parameters produce distinct tokens even though their
    placeholders (typed NULLs) are identical in SQL.
    """
    import xorq.expr.operations as xops  # noqa: PLC0415
    from xorq.common.utils.graph_utils import walk_nodes  # noqa: PLC0415

    return tuple(
        _stable_opaque_name("param", p.label, str(p.dtype), str(p.default))
        for p in walk_nodes(xops.NamedScalarParameter, op)
    )


def _decompose_expr(
    expr: Expr, op: Node
) -> tuple[
    str,
    tuple[Read, ...],
    tuple[DatabaseTable, ...],
    tuple[AggUDF | ScalarUDF, ...],
    tuple[InMemoryTable, ...],
    tuple[str, ...],
    tuple[HashingTag, ...],
    tuple[TeeNode, ...],
    tuple[CacheTag, ...],
]:
    """Split an expression into structural SQL, data leaves, UDFs, and identity nodes.

    Returns ``(sql, reads, dts, udfs, mems, param_anchors, hashing_tags, tee_nodes, cache_tags)``
    where *reads*/*dts*/*mems* are the data-carrying leaf ops, *udfs* are
    structural code-identity ops, *param_anchors* are stable identity
    strings for each NamedScalarParameter in graph order, *hashing_tags*
    carry user-supplied metadata, and *tee_nodes* carry writer identity.
    """
    from xorq.common.utils.graph_utils import replace_nodes, walk_nodes  # noqa: PLC0415
    from xorq.expr.api import get_compiler, to_sql  # noqa: PLC0415
    from xorq.expr.relations import (  # noqa: PLC0415
        CachedNode,
        CacheTag,
        HashingTag,
        Read,
        TeeNode,
    )
    from xorq.vendor.ibis.expr.operations.relations import (  # noqa: PLC0415
        DatabaseTable,
        InMemoryTable,
    )
    from xorq.vendor.ibis.expr.operations.udf import AggUDF, ScalarUDF  # noqa: PLC0415

    compiler = get_compiler(expr)
    # Collect param anchors *before* replacement erases the identity.
    param_anchors = _collect_param_anchors(op)
    # Use replace_nodes (not op.replace) so the opaque-placeholder rewrite
    # descends into Any-typed sub-expressions (RemoteTable.remote_expr,
    # CachedNode.parent, FlightExpr/UDXF.input_expr,
    # ExprScalarUDF.computed_kwargs_expr). Without this, inner opaque nodes
    # keep their gen_name()-randomized names and leak randomness into SQL.
    rewritten = replace_nodes(_xorq_opaque_to_placeholder, op)
    sql = str(to_sql(rewritten.to_expr().unbind(), compiler=compiler))
    # walk_nodes descends through the same Any-typed boundaries, so leaves
    # reachable only through opaque sub-expressions still contribute their
    # data identity to the hash.
    reads = tuple(walk_nodes(Read, op))
    dts = tuple(
        n
        for n in walk_nodes(DatabaseTable, op)
        if not isinstance(n, (CachedNode, Read))
    )
    udfs = tuple(walk_nodes((AggUDF, ScalarUDF), op))
    mems = tuple(walk_nodes(InMemoryTable, op))
    hashing_tags = tuple(walk_nodes(HashingTag, op))
    tee_nodes = tuple(walk_nodes(TeeNode, op))
    # A pinned read (CacheTag) is a hash *leaf*: its identity is the cache key,
    # folded in _hash_expr_components via __dasher_tokenize__. Prune every leaf
    # living under a CacheTag -- both ``parent`` (the cache-file read, whose
    # absolute path is base_path-dependent) and ``uncached`` (the discarded
    # upstream, whose sources would be stat'd) -- so a pin hashes consistently
    # across cache dirs and without its original sources still existing.
    # Pruning is by node identity; a no-op when the expr has no pins.
    cache_tags = tuple(walk_nodes(CacheTag, op))
    if cache_tags:
        # The leaf types pruned here are exactly the data/identity collections
        # gathered above; one walk per tag into a set (which dedups nodes shared
        # by nested pins), then a single membership filter over each collection.
        pinned_leaf_types = (
            Read,
            DatabaseTable,
            InMemoryTable,
            AggUDF,
            ScalarUDF,
            HashingTag,
            TeeNode,
        )
        pinned = {
            node for ct in cache_tags for node in walk_nodes(pinned_leaf_types, ct)
        }
        reads, dts, udfs, mems, hashing_tags, tee_nodes = (
            tuple(n for n in coll if n not in pinned)
            for coll in (reads, dts, udfs, mems, hashing_tags, tee_nodes)
        )
    return (
        sql,
        reads,
        dts,
        udfs,
        mems,
        param_anchors,
        hashing_tags,
        tee_nodes,
        cache_tags,
    )


def _hash_expr_components(expr: Expr, op: Node) -> tuple[str, list[SlotDict]]:
    from xorq.common.utils.dasher import HASHER, _current_hasher  # noqa: PLC0415

    (
        sql,
        reads,
        dts,
        udfs,
        mems,
        param_anchors,
        hashing_tags,
        tee_nodes,
        cache_tags,
    ) = _decompose_expr(expr, op)
    hasher = _current_hasher.get() or HASHER

    hash_args = ("ibis.Expr.structural", sql, udfs)
    if param_anchors:
        hash_args += (param_anchors,)
    if hashing_tags:
        # By design, a HashingTag tokenizes by (schema, metadata) only -- not its
        # parent (see HashingTag.__dasher_tokenize__). Graph position is already
        # captured by `sql` above, so dropping the parent here cannot collapse two
        # structurally-different exprs; it only means same-metadata tags at
        # different points contribute the same token, which is correct.
        hash_args += (tuple(hasher.tokenize(ht) for ht in hashing_tags),)
    if _include_tee_nodes.get() and tee_nodes:
        hash_args += (tuple(hasher.tokenize(tn) for tn in tee_nodes),)
    if cache_tags:
        # A pinned read contributes only its cache-key identity (base_path-
        # independent, source-free) via CacheTag.__dasher_tokenize__; its
        # subtree leaves were pruned in _decompose_expr.
        hash_args += (tuple(hasher.tokenize(ct) for ct in cache_tags),)
    structural_hash = hasher.tokenize(*hash_args)

    def _read_name(r: Read) -> str:
        read_kwargs = dict(r.read_kwargs)
        rp = read_kwargs.get("read_path")
        name = rp if rp is not None else read_kwargs.get("hash_path", "")
        if isinstance(name, (list, tuple)):
            name = ", ".join(str(p) for p in name) if name else ""
        return str(name)

    labeled = itertools.chain(
        (("Read", _read_name(r), hasher.tokenize(r)) for r in reads),
        (("DatabaseTable", getattr(dt, "name", ""), hasher.tokenize(dt)) for dt in dts),
        (
            (
                "InMemoryTable",
                getattr(m, "name", ""),
                hasher.tokenize(normalize_inmemorytable(m)),
            )
            for m in mems
        ),
    )
    slots: list[SlotDict] = [
        {"index": idx, "kind": kind, "name": name, "hash": h}
        for idx, (kind, name, h) in enumerate(labeled)
    ]

    return structural_hash, slots


def _normalize_expr_xorq_impl(expr: Expr, op: Node) -> tuple[str, ...]:
    structural_hash, slots = _hash_expr_components(expr, op)
    slot_hashes = tuple(s["hash"] for s in slots)
    return ("ibis.Expr.v4", structural_hash, *slot_hashes)


def expr_metadata(expr: Expr) -> ExprMetadata:
    """Produce serializable metadata for cross-environment token recomputation.

    Returns a dict of the form::

        {
          "version": 4,
          "structural_hash": "<xxh128 hex>",
          "slots": [
              {"index": 0, "kind": "Read", "name": "...", "hash": "<xxh128 hex>"},
              ...
          ],
        }

    UDFs (``AggUDF``, ``ScalarUDF``), HashingTags (via ``__dasher_tokenize__``),
    and (when ``_include_tee_nodes`` is set) TeeNodes (via
    ``__dasher_tokenize__``, which delegates to each ``WriteThrough``'s own
    ``__dasher_tokenize__``) all contribute to ``structural_hash`` rather
    than appearing as separate slots.

    The expression token can be recomputed from this dict using
    :func:`~xorq.common.utils.dasher._recompute.compute_expr_token`, which
    only needs ``xxhash`` and ``struct`` — no xorq or ibis import required.
    """
    op = expr.op()
    structural_hash, slots = _hash_expr_components(expr, op)

    return {
        "version": 4,
        "structural_hash": structural_hash,
        "slots": slots,
    }


__all__ = [
    "_normalize_computed_kwargs_expr",
    "_normalize_expr_xorq",
    "_normalize_scalar_udf_xorq",
    "_parent_token",
    "_rename_unbound_xorq",
    "_stable_opaque_name",
    "_xorq_opaque_to_placeholder",
    "expr_metadata",
    "include_tee_nodes",
]
