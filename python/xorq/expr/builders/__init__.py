"""from_tagged registry — recover domain objects from expression tags.

An ExprBuilder is an expression whose tags carry domain-specific metadata.
Handlers are ``TagHandler`` instances registered by tag_name, providing:

- ``extract_metadata(tag_node) → dict`` — sidecar metadata for the catalog
- ``from_tagged(tag_node) → object``    — recover a live domain object

Both are optional.  If only ``from_tagged`` is provided, a minimal metadata
dict ``{"type": tag_name}`` is generated automatically.

Built-in handlers (BSL, ML pipeline) are registered at import time.
Third-party packages register via the ``"xorq.from_tagged"`` entry point
group — the loaded object must be a ``TagHandler``.

Entry point example (pyproject.toml)::

    [project.entry-points."xorq.from_tagged"]
    my_tag = "my_package.handlers:my_handler"

Handler example::

    from xorq.expr.builders import TagHandler

    my_handler = TagHandler(
        extract_metadata=lambda tag_node: {"type": "my_model", ...},
        from_tagged=lambda tag_node: MyModel.from_tag(tag_node),
    )
"""

from __future__ import annotations

import importlib.metadata
from typing import Optional

from attr import field, frozen
from attr.validators import optional as optional_v


@frozen
class TagHandler:
    """Handler for a tagged expression builder.

    Attributes
    ----------
    extract_metadata : callable, optional
        ``(tag_node) → dict`` — produce sidecar metadata for the catalog.
    from_tagged : callable, optional
        ``(tag_node) → object`` — recover a live domain object from the tag.
    """

    extract_metadata: Optional[callable] = field(
        default=None, validator=optional_v(lambda inst, attr, val: callable(val))
    )
    from_tagged: Optional[callable] = field(
        default=None, validator=optional_v(lambda inst, attr, val: callable(val))
    )


_FROM_TAGGED_REGISTRY: dict[str, TagHandler] = {}


def register_tag_handler(tag_name, tag_handler, *, override=False):
    """Register a ``TagHandler`` for *tag_name*."""
    if not override and tag_name in _FROM_TAGGED_REGISTRY:
        raise ValueError(f"tag handler already registered for {tag_name!r}")
    _FROM_TAGGED_REGISTRY[tag_name] = tag_handler


def get_from_tagged_registry():
    """Return the handler registry, discovering entry points on first call."""
    if not _FROM_TAGGED_REGISTRY:
        _register_builtins()
        _discover_from_tagged()
    return _FROM_TAGGED_REGISTRY


def _discover_from_tagged():
    """Discover handlers from entry points (group "xorq.from_tagged")."""
    for ep in importlib.metadata.entry_points(group="xorq.from_tagged"):
        try:
            handler = ep.load()
            _FROM_TAGGED_REGISTRY[ep.name] = handler
        except Exception:
            import structlog  # noqa: PLC0415

            structlog.get_logger().warning(
                "failed to load from_tagged entry point",
                entry_point=ep.name,
                exc_info=True,
            )


# ---------------------------------------------------------------------------
# Public dispatch functions
# ---------------------------------------------------------------------------


def extract_builder_metadata(tag_name, tag_node):
    """Look up *tag_name* in the registry and return sidecar metadata dict, or None."""
    registry = get_from_tagged_registry()
    handler = registry.get(tag_name)
    if handler is None:
        return None
    if handler.extract_metadata is not None:
        return handler.extract_metadata(tag_node)
    return {"type": tag_name}


def from_tagged_dispatch(expr):
    """Walk tags on *expr*, dispatch to registry, return the first domain object.

    Raises ``ValueError`` if no handler with ``from_tagged`` matches.
    """
    from xorq.common.utils.graph_utils import walk_nodes  # noqa: PLC0415
    from xorq.expr.relations import HashingTag, Tag  # noqa: PLC0415

    registry = get_from_tagged_registry()
    tag_nodes = walk_nodes((Tag, HashingTag), expr)

    for tag_node in tag_nodes:
        tag_name = tag_node.metadata.get("tag")
        handler = registry.get(tag_name)
        if handler is not None and handler.from_tagged is not None:
            result = handler.from_tagged(tag_node)
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


def _bsl_from_tagged(tag_node):
    # TODO: BSL's from_tagged returns the full query chain (SemanticAggregate),
    # not the SemanticModel. We reconstruct only the base SemanticTableOp to
    # get the SemanticModel back so callers can issue new .query() calls.
    from boring_semantic_layer.serialization import (  # noqa: PLC0415
        BSLSerializationContext,
        extract_xorq_metadata,
        reconstruct_bsl_operation,
    )

    expr = tag_node.to_expr()
    ctx = BSLSerializationContext()
    metadata = extract_xorq_metadata(expr)
    # Walk to innermost source (SemanticTableOp)
    while src := ctx.parse_field(metadata, "source"):
        metadata = src
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


def _ml_from_tagged(tag_node):
    from xorq.expr.ml.pipeline_lib import FittedPipeline  # noqa: PLC0415

    return FittedPipeline.from_expr(tag_node.to_expr())


def _register_builtins():
    """Register built-in handlers for BSL and ML pipeline tags."""
    from xorq.expr.ml.enums import FittedPipelineTagKey  # noqa: PLC0415

    _FROM_TAGGED_REGISTRY["bsl"] = TagHandler(
        extract_metadata=_bsl_extract_metadata,
        from_tagged=_bsl_from_tagged,
    )

    ml_handler = TagHandler(
        extract_metadata=_ml_pipeline_extract_metadata,
        from_tagged=_ml_from_tagged,
    )
    for key in FittedPipelineTagKey:
        if key not in (FittedPipelineTagKey.ALL_STEPS, FittedPipelineTagKey.TRAINING):
            _FROM_TAGGED_REGISTRY[str(key)] = ml_handler
