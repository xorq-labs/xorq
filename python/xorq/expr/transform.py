"""Declarative driver for the ``to_pyarrow_batches`` (tier-1) transform passes.

Historically the six effectful/pure passes in ``_transform_expr`` were a hand-run
sequence where three correctness rules lived only in prose comments repeated at
each call site:

- **traversal kind** -- descend into opaque sub-exprs (``replace_nodes``) vs stop
  at them (``op.replace``); the rule and its full rationale live on
  :class:`~xorq.expr.enums.Traversal`.
- **ordering** -- e.g. cache resolution must precede the tee pass (a cache hit
  prunes the ``TeeNode`` before its write fires); deferred reads resolve last.
- **resource ownership** -- effectful passes adopt what they materialize into a
  single shared ``RemoteTableScope`` (see ``remote_table_exec``).

This module turns those rules into *data*: each pass is a :class:`TransformPass`
record carrying its :class:`Traversal` kind, whether it ``produces_resources``
(and so needs the scope), and its ``after`` dependencies. :func:`apply_pass`
selects the traversal from the record (so no call site can pick the wrong walk),
and :func:`run_transform_passes` folds the ordered table while asserting every
``after`` constraint holds -- a reorder now raises loudly instead of silently
breaking correctness.

Adjacent *fusable* passes -- pure structural rewrites (``DESCEND`` and
``not produces_resources``) -- are coalesced into a single ``replace_nodes``
walk whose composed replacer applies each pass's rewrite in table order at every
node. This is the one place ``produces_resources`` is read: it keeps an
effectful DESCEND pass out of the shared walk. Fusion only collapses *how many*
graph traversals a run of pure rewrites costs, never their order or result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from attr.validators import deep_iterable, instance_of, is_callable, optional
from attrs import field, frozen

from xorq.common.exceptions import InternalError
from xorq.common.utils.otel_utils import tracer
from xorq.expr.enums import Traversal
from xorq.vendor.ibis import Expr
from xorq.vendor.ibis.expr.operations import Node


if TYPE_CHECKING:
    # remote_table_exec imports this module, so a module-level import would be
    # circular; used for annotations and the scope validator's lazy check.
    from xorq.expr.remote_table_exec import RemoteTableScope


# A replacer is the per-node rewrite ``(node, kwargs) -> node`` consumed by both
# ``replace_nodes`` and ``op.replace``; a builder produces one from the expr +
# context (closing over the scope, bound params, reader counts, etc.).
Replacer = Callable[[Node, "dict | None"], Node]
ReplacerBuilder = Callable[["Expr", "TransformCtx"], Replacer]
Predicate = Callable[["Expr", "TransformCtx"], bool]


def _validate_scope(instance: Any, attribute: Any, value: Any) -> None:
    """Lazy ``instance_of(RemoteTableScope)``.

    The class is imported inside the validator, not at module level, to keep
    ``transform`` out of the ``remote_table_exec`` import cycle. The import is
    cheap by the time any ``TransformCtx`` is constructed (every module is loaded).
    """
    from xorq.expr.remote_table_exec import RemoteTableScope  # noqa: PLC0415

    instance_of(RemoteTableScope)(instance, attribute, value)


@frozen
class TransformCtx:
    """Per-transform context threaded to every pass builder.

    ``scope`` owns every resource the effectful passes materialize (created once,
    up front, in ``_transform_expr``); it is ``None`` only when driving a subset
    of passes that produce nothing (e.g. a lone pure pass in a test).
    ``name_values`` are the resolved parameter bindings;
    ``read_record_batches_kwargs`` are the backend read kwargs that must reach
    ``read_record_batches`` (e.g. Snowflake's ``database=``).
    """

    scope: RemoteTableScope | None = field(
        default=None, validator=optional(_validate_scope)
    )
    name_values: dict = field(factory=dict, validator=instance_of(dict))
    read_record_batches_kwargs: dict = field(factory=dict, validator=instance_of(dict))


@frozen
class TransformPass:
    """One tier-1 pass as data: what it does (``build``), how it walks
    (``traversal``), whether it owns resources (``produces_resources``), when it
    applies (``when``), and what must precede it (``after``)."""

    name: str = field(validator=instance_of(str))
    traversal: Traversal = field(validator=instance_of(Traversal))
    build: ReplacerBuilder = field(validator=is_callable())
    produces_resources: bool = field(default=False, validator=instance_of(bool))
    when: Predicate | None = field(default=None, validator=optional(is_callable()))
    after: tuple[str, ...] = field(
        default=(), validator=deep_iterable(instance_of(str), instance_of(tuple))
    )


def _is_fusable(pass_: TransformPass) -> bool:
    """Whether ``pass_`` can share a single descend-walk with its neighbours.

    True for a pure structural rewrite: it descends (``DESCEND``) and
    materializes nothing (``not produces_resources``). This is the sole consumer
    of ``produces_resources`` -- the flag exists to keep an effectful DESCEND
    pass out of the fused walk, where its side effect would fire under a shared
    traversal it does not own. A BOUNDARY pass is never fusable: it must fire
    once, at the execution boundary (see :class:`~xorq.expr.enums.Traversal`).
    """
    return pass_.traversal is Traversal.DESCEND and not pass_.produces_resources


def _fusion_groups(
    passes: tuple[TransformPass, ...],
) -> list[tuple[TransformPass, ...]]:
    """Partition ``passes`` into ordered groups: each maximal run of adjacent
    fusable passes becomes one group, every other pass a singleton. Order is
    preserved, so passes still apply in table order."""
    groups: list[tuple[TransformPass, ...]] = []
    run: list[TransformPass] = []
    for pass_ in passes:
        if _is_fusable(pass_):
            run.append(pass_)
            continue
        if run:
            groups.append(tuple(run))
            run = []
        groups.append((pass_,))
    if run:
        groups.append(tuple(run))
    return groups


def _fuse_replacers(replacers: list[Replacer]) -> Replacer:
    """Chain per-node replacers into one, applied in list order at each node.

    The fused replacer does the single bottom-up ``__recreate__`` (rebuild the
    node with its already-transformed children) itself, then applies every pass
    replacer to that recreated node with ``kwargs=None`` -- so recreation happens
    exactly once *regardless of pass order*: no replacer has to be a
    "kwargs-consuming head", so reordering the group (or adding a pass that skips
    the recreate on some branch, as ``remove_tags`` does for a ``Tag``) cannot
    silently drop the transformed children.

    Applying r1-then-r2 at each node of one bottom-up walk equals r1-everywhere
    then r2-everywhere because a fusable pass is a *node-local* rewrite -- its
    output at a node depends only on that node and its already-transformed
    children, never on the global post-r1 tree -- which pure DESCEND passes satisfy.
    """

    def fused(node, kwargs):
        if kwargs:
            node = node.__recreate__(kwargs)
        for replacer in replacers:
            node = replacer(node, None)
        return node

    return fused


def _apply_active_pass(pass_: TransformPass, expr: Expr, ctx: TransformCtx) -> Expr:
    """Build and run one pass whose ``when`` is already evaluated (or absent),
    selecting the walk from ``pass_.traversal``.

    A ``produces_resources`` pass may adopt readers/caches/tables into
    ``ctx.scope`` before failing; on any exception the whole shared scope (this
    pass's resources and earlier passes') is torn down before the error
    propagates, so a direct caller that has not yet entered its own ``with scope:``
    does not leak. ``close`` is idempotent, so an outer guard closing again is a
    no-op.
    """
    # lazy: graph_utils imports xorq.expr.relations, which imports this module at
    # module level -- importing it here keeps that edge out of the import cycle.
    from xorq.common.utils.graph_utils import replace_nodes  # noqa: PLC0415

    with tracer.start_as_current_span(f"transform.{pass_.name}"):
        try:
            replacer = pass_.build(expr, ctx)
            if pass_.traversal is Traversal.DESCEND:
                return replace_nodes(replacer, expr).to_expr()
            return expr.op().replace(replacer).to_expr()
        except BaseException:
            if pass_.produces_resources and ctx.scope is not None:
                ctx.scope.close()
            raise


def apply_pass(pass_: TransformPass, expr: Expr, ctx: TransformCtx) -> Expr:
    """Run a single pass: skip it when its ``when`` predicate is False, else build
    and walk it (selecting the traversal from ``pass_.traversal``).

    A resource-producing pass tears its (and earlier passes') scope resources
    down on failure; see :func:`_apply_active_pass`.
    """
    if pass_.when is not None and not pass_.when(expr, ctx):
        return expr
    return _apply_active_pass(pass_, expr, ctx)


def _assert_after_ordering(passes: tuple[TransformPass, ...]) -> None:
    """Raise unless every pass's ``after`` deps are positioned earlier in the
    tuple. Positioning is by table order, not execution: a pass skipped via
    ``when`` still counts as positioned, so a reorder raises immediately instead
    of silently mis-transforming."""
    positioned: set[str] = set()
    for pass_ in passes:
        missing = tuple(dep for dep in pass_.after if dep not in positioned)
        if missing:
            raise InternalError(
                f"transform pass {pass_.name!r} must run after {missing}, "
                f"but they are not positioned earlier (seen: {sorted(positioned)})"
            )
        positioned.add(pass_.name)


def _apply_group(
    expr: Expr, group: tuple[TransformPass, ...], ctx: TransformCtx
) -> Expr:
    """Apply one fusion group. A singleton goes through :func:`apply_pass`; a
    group whose ``when`` predicates leave a single active pass runs that one pass
    directly (its ``when`` already evaluated); two or more active passes share one
    ``replace_nodes`` walk under a composed replacer.
    """
    if len(group) == 1:
        return apply_pass(group[0], expr, ctx)
    # lazy import: see _apply_active_pass.
    from xorq.common.utils.graph_utils import replace_nodes  # noqa: PLC0415

    active = tuple(p for p in group if p.when is None or p.when(expr, ctx))
    if not active:
        return expr
    if len(active) == 1:
        # ``when`` already evaluated above; skip ``apply_pass`` so its predicate
        # (a graph walk) is not re-run.
        return _apply_active_pass(active[0], expr, ctx)
    label = "+".join(p.name for p in active)
    with tracer.start_as_current_span(f"transform.fuse[{label}]"):
        replacers = [p.build(expr, ctx) for p in active]
        return replace_nodes(_fuse_replacers(replacers), expr).to_expr()


def run_transform_passes(
    expr: Expr, passes: tuple[TransformPass, ...], ctx: TransformCtx
) -> Expr:
    """Fold the ordered ``passes`` over ``expr``, asserting ``after`` deps.

    Ordering is validated up front (:func:`_assert_after_ordering`), then passes
    execute in fusion groups (:func:`_fusion_groups`): adjacent pure DESCEND
    rewrites share one graph walk, everything else applies on its own. The result
    is identical to applying every pass singly -- fusion only saves traversals.
    """
    _assert_after_ordering(passes)
    for group in _fusion_groups(passes):
        expr = _apply_group(expr, group, ctx)
    return expr
