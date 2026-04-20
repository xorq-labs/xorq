"""from_tag_node registry — recover domain objects from expression tags.

An ExprBuilder is an expression whose tags carry domain-specific metadata.
Handlers are ``TagHandler`` instances that declare their ``tag_names`` and
provide:

- ``extract_metadata(tag_node) → dict`` — sidecar metadata for the catalog
- ``from_tag_node(tag_node) → object``  — recover a live domain object

Both callbacks are optional (but at least one is required).  If only
``from_tag_node`` is provided, a minimal metadata dict ``{"type": tag_name}``
is generated automatically.

Built-in handlers (ML pipeline) are declared in ``_builtin_handlers()``.
Third-party packages register via the ``"xorq.from_tag_node"`` entry point
group — the loaded object must be a ``TagHandler``.

Entry point example (pyproject.toml)::

    [project.entry-points."xorq.from_tag_node"]
    my_plugin = "my_package.handlers:my_handler"

Handler example::

    from xorq.expr.builders import TagHandler, register_tag_handler

    my_handler = TagHandler(
        tag_names=("my_tag",),
        extract_metadata=lambda tag_node: {"type": "my_model", ...},
        from_tag_node=lambda tag_node: MyModel.from_tag(tag_node),
    )
    register_tag_handler(my_handler)
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable
from typing import Optional

from attr import field, frozen
from attr.validators import deep_iterable, instance_of, is_callable, optional


@frozen
class TagHandler:
    tag_names: tuple[str, ...] = field(
        converter=tuple,
        validator=deep_iterable(instance_of(str), instance_of(tuple)),
    )
    extract_metadata: Optional[Callable] = field(
        default=None, validator=optional(is_callable())
    )
    from_tag_node: Optional[Callable] = field(
        default=None, validator=optional(is_callable())
    )
    reemit: Optional[Callable] = field(default=None, validator=optional(is_callable()))

    def __attrs_post_init__(self):
        if not self.tag_names:
            raise ValueError("TagHandler must declare at least one tag_name")
        if self.extract_metadata is None and self.from_tag_node is None:
            raise ValueError(
                "TagHandler must have at least one of extract_metadata or from_tag_node"
            )


_FROM_TAG_NODE_REGISTRY: dict[str, TagHandler] = {}
_BUILTIN_KEYS: frozenset[str] = frozenset()
_initialized = False


def _register_handler(handler, *, builtin=False, override=False):
    """Expand handler.tag_names into the registry."""
    for name in handler.tag_names:
        if not builtin and name in _BUILTIN_KEYS:
            raise ValueError(f"{name!r} is a protected builtin tag key")
        if not override and name in _FROM_TAG_NODE_REGISTRY:
            raise ValueError(f"tag handler already registered for {name!r}")
        _FROM_TAG_NODE_REGISTRY[name] = handler


def _ensure_initialized():
    global _initialized, _BUILTIN_KEYS
    if _initialized:
        return
    _initialized = True
    for handler in _builtin_handlers():
        _register_handler(handler, builtin=True)
    _BUILTIN_KEYS = frozenset(_FROM_TAG_NODE_REGISTRY)
    for handler in _discover_from_tag_node():
        _register_handler(handler)


def _reset_registry():
    """For testing only. Clears and reinitializes the registry."""
    global _initialized, _BUILTIN_KEYS
    _FROM_TAG_NODE_REGISTRY.clear()
    _BUILTIN_KEYS = frozenset()
    _initialized = False


def register_tag_handler(handler, *, override=False):
    """Register a ``TagHandler``. Raises if any tag_name conflicts."""
    _ensure_initialized()
    if not isinstance(handler, TagHandler):
        raise TypeError(f"expected TagHandler, got {type(handler).__name__}")
    _register_handler(handler, override=override)


def _get_from_tag_node_registry():
    _ensure_initialized()
    return _FROM_TAG_NODE_REGISTRY


def _discover_from_tag_node():
    """Load handlers from entry points (group "xorq.from_tag_node")."""
    handlers = []
    for ep in importlib.metadata.entry_points(group="xorq.from_tag_node"):
        try:
            handler = ep.load()
            if not isinstance(handler, TagHandler):
                raise TypeError(
                    f"entry point {ep.name!r} must be a TagHandler, "
                    f"got {type(handler).__name__}"
                )
            if any(name in _BUILTIN_KEYS for name in handler.tag_names):
                import structlog  # noqa: PLC0415

                structlog.get_logger().warning(
                    "entry point tried to override builtin tag key -- skipped",
                    entry_point=ep.name,
                    tag_names=handler.tag_names,
                )
                continue
            handlers.append(handler)
        except Exception:
            import structlog  # noqa: PLC0415

            structlog.get_logger().warning(
                "failed to load from_tag_node entry point",
                entry_point=ep.name,
                exc_info=True,
            )
    return handlers


# ---------------------------------------------------------------------------
# Public dispatch functions
# ---------------------------------------------------------------------------


def extract_builder_metadata(tag_node):
    """Look up the tag on *tag_node* in the registry and return sidecar metadata dict, or None."""
    tag_name = tag_node.metadata.get("tag")
    registry = _get_from_tag_node_registry()
    handler = registry.get(tag_name)
    if handler is None:
        return None
    if handler.extract_metadata is not None:
        return handler.extract_metadata(tag_node)
    return {"type": tag_name}


_SINGLE_OUTPUT_DISPATCH = "single_output"


def get_rebuild_dispatch(tag_node):
    """Return a rebuild dispatch for *tag_node*, or ``None``.

    Dispatch order:
      1. Handler-level ``reemit`` callable (e.g., BSL-shape builders whose
         ``from_tag_node`` does not carry the full recipe).
      2. Domain-object ``reemit(tag_node, rebuild_subexpr)`` method
         (multi-output builders like ``FittedPipeline``).
      3. Domain-object ``with_inputs_translated(remap, to_catalog)`` +
         ``expr`` (single-output builders like ``ExprComposer``).

    Returns
    -------
    callable | tuple | None
        - ``callable(rebuild_subexpr) -> Expr`` for paths 1 and 2.
        - ``(_SINGLE_OUTPUT_DISPATCH, builder)`` sentinel for path 3; the
          driver must supply ``remap`` / ``to_catalog``.
        - ``None`` when no handler matches or no rebuild path is available.
    """
    registry = _get_from_tag_node_registry()
    handler = registry.get(tag_node.metadata.get("tag"))
    if handler is None:
        return None
    if callable(handler.reemit):
        return lambda rebuild_subexpr: handler.reemit(tag_node, rebuild_subexpr)
    if handler.from_tag_node is None:
        return None
    builder = handler.from_tag_node(tag_node)
    if builder is None:
        return None
    if callable(getattr(builder, "reemit", None)):
        return lambda rebuild_subexpr: builder.reemit(tag_node, rebuild_subexpr)
    if callable(getattr(builder, "with_inputs_translated", None)) and hasattr(
        builder, "expr"
    ):
        return (_SINGLE_OUTPUT_DISPATCH, builder)
    return None


def _resolve_builder_from_tag(expr):
    """Walk tags on *expr*, dispatch to registry, return the first domain object.

    Tags are visited in graph-walk order (outermost first). The first handler
    that returns a non-None result wins.

    Raises ``ValueError`` if no handler with ``from_tag_node`` matches.
    """
    from xorq.common.utils.graph_utils import walk_nodes  # noqa: PLC0415
    from xorq.expr.relations import HashingTag, Tag  # noqa: PLC0415

    registry = _get_from_tag_node_registry()
    tag_nodes = walk_nodes((Tag, HashingTag), expr)

    for tag_node in tag_nodes:
        tag_name = tag_node.metadata.get("tag")
        handler = registry.get(tag_name)
        if handler is not None and handler.from_tag_node is not None:
            result = handler.from_tag_node(tag_node)
            if result is not None:
                return result
    raise ValueError("No builder tags found in expression")


# ---------------------------------------------------------------------------
# Built-in handler factories
# ---------------------------------------------------------------------------


def _ml_pipeline_extract_metadata(tag_node):
    from xorq.expr.ml.enums import FittedPipelineTagKey  # noqa: PLC0415

    all_steps_raw = tag_node.metadata.get(str(FittedPipelineTagKey.ALL_STEPS), ())
    dicts = tuple(dict(step_items) for step_items in all_steps_raw)
    steps = tuple(d.get("name", "unknown") for d in dicts)
    targets = tuple(d["target"] for d in dicts if d.get("target"))
    features_all = tuple(d["features"] for d in dicts if d.get("features"))
    return {
        "type": "fitted_pipeline",
        "description": f"{len(steps)} steps",
        "steps": steps,
        "features": features_all[-1] if features_all else (),
        "target": targets[-1] if targets else None,
    }


def _ml_from_tag_node(tag_node):
    from xorq.expr.ml.pipeline_lib import FittedPipeline  # noqa: PLC0415

    return FittedPipeline.from_tag_node(tag_node)


def _builtin_handlers():
    """Declare built-in handlers for ML pipeline tags."""
    from xorq.expr.ml.enums import FittedPipelineTagKey  # noqa: PLC0415

    return (
        TagHandler(
            tag_names=tuple(
                str(k)
                for k in FittedPipelineTagKey
                if k
                not in (FittedPipelineTagKey.ALL_STEPS, FittedPipelineTagKey.TRAINING)
            ),
            extract_metadata=_ml_pipeline_extract_metadata,
            from_tag_node=_ml_from_tag_node,
        ),
    )
