"""Builder framework — factories that produce expressions.

A Builder is an attrs @frozen class that knows how to:
- build_expr(**kwargs) → Expr — produce an expression from builder-specific selections
- to_build_dir(path)       — serialize to a catalog build directory (writes builder_meta.json)
- from_build_dir(path)     — reconstruct from a catalog build directory
- from_tagged(tag_node)    — recover builder from expression tags (provenance)

Built-in builders are registered via entry points in pyproject.toml
(group "xorq.builders"). Third-party packages register theirs the same way.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

from attr import field, frozen
from attr.validators import instance_of


try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum

from xorq.vendor.ibis import Expr


BUILDER_META_FILENAME = "builder_meta.json"


class BuilderKind(StrEnum):
    SemanticModel = "semantic_model"
    FittedPipeline = "fitted_pipeline"


_BUILDER_REGISTRY: dict[str, type[Builder]] = {}


def register_builder(name):
    """Register a Builder subclass by name."""

    def decorator(cls):
        _BUILDER_REGISTRY[name] = cls
        return cls

    return decorator


def get_registry():
    """Return the current builder registry, discovering entry points on first call."""
    if not _BUILDER_REGISTRY:
        _discover_builders()
    return _BUILDER_REGISTRY


def _discover_builders():
    """Discover builder classes from entry points (group "xorq.builders")."""
    for ep in importlib.metadata.entry_points(group="xorq.builders"):
        try:
            adapter_cls = ep.load()
            _BUILDER_REGISTRY[ep.name] = adapter_cls
        except Exception:
            # TODO: determine if this should hard-fail when a builder entry
            # point is malformed (missing tag_name, import error, etc.)
            # rather than silently skipping — currently best-effort so one
            # broken third-party package doesn't block the whole registry.
            import structlog  # noqa: PLC0415

            structlog.get_logger().warning(
                "failed to load builder entry point",
                entry_point=ep.name,
                exc_info=True,
            )


@frozen
class Builder:
    """Base class for expression builders (factories).

    Subclasses must set ``tag_name`` as a class-level default and implement
    ``build_expr``, ``from_tagged``, ``from_build_dir``, and ``to_build_dir``.
    """

    tag_name: str = field(validator=instance_of(str))

    def build_expr(self, **kwargs) -> Expr:
        """Produce an expression. Args are builder-specific."""
        raise NotImplementedError

    @classmethod
    def from_tagged(cls, tag_node) -> Builder:
        """Recover builder from expression tags (for provenance detection)."""
        raise NotImplementedError

    @classmethod
    def from_build_dir(cls, path: Path) -> Builder:
        """Reconstruct builder from a catalog build directory."""
        raise NotImplementedError

    def to_build_dir(self, path: Path) -> None:
        """Write builder files to a build directory.

        Must write ``builder_meta.json`` plus any builder-specific files.
        The JSON must include ``"type"`` and ``"description"`` keys.
        """
        raise NotImplementedError
