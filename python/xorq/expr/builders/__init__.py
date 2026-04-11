"""from_tag_node registry — recover domain objects from expression tags.

An ExprBuilder is an expression whose tags carry domain-specific metadata.
Handlers are ``TagHandler`` instances that declare their ``tag_names`` and
provide:

- ``extract_metadata(tag_node) → dict`` — sidecar metadata for the catalog
- ``from_tag_node(tag_node) → object``  — recover a live domain object

Both callbacks are optional (but at least one is required).  If only
``from_tag_node`` is provided, a minimal metadata dict ``{"type": tag_name}``
is generated automatically.

Built-in handlers (BSL, ML pipeline) are declared in ``_builtin_handlers()``.
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


def _bsl_extract_metadata(tag_node):
    meta = tag_node.metadata
    table_meta = meta
    while table_meta.get("bsl_op_type") != "SemanticTableOp":
        source = table_meta.get("source")
        if source is None:
            break
        table_meta = dict(source) if isinstance(source, tuple) else source
    dims = tuple(d[0] for d in table_meta.get("dimensions", ()))
    measures = tuple(m[0] for m in table_meta.get("measures", ()))
    return {
        "type": "semantic_model",
        "description": f"{len(dims)} dims, {len(measures)} measures",
        "dimensions": dims,
        "measures": measures,
    }


def _bsl_from_tag_node(tag_node):
    # TODO: BSL's from_tag_node returns the full query chain (SemanticAggregate),
    # not the SemanticModel. We reconstruct only the base SemanticTableOp to
    # get the SemanticModel back so callers can issue new .query() calls.
    try:
        from boring_semantic_layer.serialization import (  # noqa: PLC0415
            BSLSerializationContext,
            extract_xorq_metadata,
            reconstruct_bsl_operation,
        )
    except ImportError:
        raise ImportError(
            "boring-semantic-layer is required to recover BSL models -- "
            "install it with: uv pip install boring-semantic-layer"
        ) from None

    expr = tag_node.to_expr()
    ctx = BSLSerializationContext()
    metadata = extract_xorq_metadata(expr)
    # Walk to innermost source (SemanticTableOp)
    _MAX_DEPTH = 100
    for _ in range(_MAX_DEPTH):
        src = ctx.parse_field(metadata, "source")
        if not src:
            break
        metadata = src
    else:
        raise RuntimeError(
            f"_bsl_from_tag_node exceeded {_MAX_DEPTH} nesting levels; "
            "possible circular metadata"
        )
    return reconstruct_bsl_operation(metadata, expr, ctx)


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
    """Declare built-in handlers for BSL and ML pipeline tags."""
    from xorq.expr.ml.enums import FittedPipelineTagKey  # noqa: PLC0415

    return (
        TagHandler(
            tag_names=("bsl",),
            extract_metadata=_bsl_extract_metadata,
            from_tag_node=_bsl_from_tag_node,
        ),
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
