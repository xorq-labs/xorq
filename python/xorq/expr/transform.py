"""Declarative driver for the ``to_pyarrow_batches`` (tier-1) transform passes.

Historically the six effectful/pure passes in ``_transform_expr`` were a hand-run
sequence where three correctness rules lived only in prose comments repeated at
each call site:

- **traversal kind** -- a pure structural rewrite descends into opaque sub-exprs
  (``replace_nodes``); an effectful or boundary-resolved pass must stop at opaque
  nodes (``op.replace``) so it fires exactly once, at this execution boundary.
  The full rationale lives on :class:`~xorq.expr.enums.Traversal`.
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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from attr.validators import deep_iterable, instance_of, is_callable, optional
from attrs import field, frozen

from xorq.common.exceptions import InternalError
from xorq.common.utils.otel_utils import tracer
from xorq.expr.enums import Traversal
from xorq.vendor.ibis import Expr


if TYPE_CHECKING:
    # remote_table_exec imports this module, so a module-level import would be
    # circular; used for annotations and the scope validator's lazy check.
    from xorq.expr.remote_table_exec import RemoteTableScope


# A replacer is the per-node rewrite ``(node, kwargs) -> node`` consumed by both
# ``replace_nodes`` and ``op.replace``; a builder produces one from the expr +
# context (closing over the scope, bound params, reader counts, etc.).
Replacer = Callable[[Any, "dict | None"], Any]
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
    up front, in ``_transform_expr``). ``name_values`` are the resolved parameter
    bindings; ``read_record_batches_kwargs`` are the backend read kwargs that must
    reach ``read_record_batches`` (e.g. Snowflake's ``database=``).
    """

    scope: RemoteTableScope = field(validator=_validate_scope)
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


def apply_pass(p: TransformPass, expr: Expr, ctx: TransformCtx) -> Expr:
    """Run a single pass, selecting the traversal from ``p.traversal``.

    The caller owns ``ctx.scope`` teardown; a failure here just propagates.
    """
    # lazy: graph_utils imports xorq.expr.relations, which imports this module at
    # module level -- importing it here keeps that edge out of the import cycle.
    from xorq.common.utils.graph_utils import replace_nodes  # noqa: PLC0415

    if p.when is not None and not p.when(expr, ctx):
        return expr
    with tracer.start_as_current_span(f"transform.{p.name}"):
        replacer = p.build(expr, ctx)
        if p.traversal is Traversal.DESCEND:
            return replace_nodes(replacer, expr).to_expr()
        return expr.op().replace(replacer).to_expr()


def run_transform_passes(
    expr: Expr, passes: tuple[TransformPass, ...], ctx: TransformCtx
) -> Expr:
    """Fold the ordered ``passes`` over ``expr``, asserting ``after`` deps.

    The ordering constraints are checked against passes *already positioned*
    earlier in the tuple (whether or not they no-op'd via ``when``), so a
    reordered table raises immediately instead of silently mis-transforming.
    """
    done: set[str] = set()
    for p in passes:
        missing = tuple(dep for dep in p.after if dep not in done)
        if missing:
            raise InternalError(
                f"transform pass {p.name!r} must run after {missing}, "
                f"but they are not positioned earlier (seen: {sorted(done)})"
            )
        expr = apply_pass(p, expr, ctx)
        done.add(p.name)
    return expr
