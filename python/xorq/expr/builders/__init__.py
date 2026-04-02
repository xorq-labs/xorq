"""from_tagged registry — recover domain objects from expression tags.

An ExprBuilder is an expression whose tags carry domain-specific metadata.
Recovery dispatches through a registry of tag_name → from_tagged callables,
discovered via the ``"xorq.from_tagged"`` entry point group.
"""

from __future__ import annotations

import importlib.metadata


_FROM_TAGGED_REGISTRY: dict[str, callable] = {}


def register_from_tagged(tag_name):
    """Register a from_tagged callable by tag name."""

    def decorator(fn):
        _FROM_TAGGED_REGISTRY[tag_name] = fn
        return fn

    return decorator


def get_from_tagged_registry():
    """Return the from_tagged registry, discovering entry points on first call."""
    if not _FROM_TAGGED_REGISTRY:
        _discover_from_tagged()
    return _FROM_TAGGED_REGISTRY


def _discover_from_tagged():
    """Discover from_tagged callables from entry points (group "xorq.from_tagged")."""
    for ep in importlib.metadata.entry_points(group="xorq.from_tagged"):
        try:
            fn = ep.load()
            _FROM_TAGGED_REGISTRY[ep.name] = fn
        except Exception:
            import structlog  # noqa: PLC0415

            structlog.get_logger().warning(
                "failed to load from_tagged entry point",
                entry_point=ep.name,
                exc_info=True,
            )


def from_tagged_dispatch(expr):
    """Walk tags on *expr*, dispatch to registry, return the first domain object found.

    Checks BSL tags first (hardcoded), then ML pipeline tags, then the registry.
    Raises ValueError if no builder tags are found.
    """
    from xorq.common.utils.graph_utils import walk_nodes  # noqa: PLC0415
    from xorq.expr.relations import HashingTag, Tag  # noqa: PLC0415

    registry = get_from_tagged_registry()
    tag_nodes = walk_nodes((Tag, HashingTag), expr)

    for tag_node in tag_nodes:
        tag_name = tag_node.metadata.get("tag")
        # BSL recovery
        if tag_name == "bsl":
            from boring_semantic_layer import from_tagged  # noqa: PLC0415

            return from_tagged(tag_node.to_expr())
        # Registry dispatch
        if tag_name in registry:
            return registry[tag_name](tag_node)
    raise ValueError("No builder tags found in expression")
