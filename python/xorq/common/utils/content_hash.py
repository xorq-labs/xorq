"""Content hash for expression-graph nodes.

Single source of truth for a node's content-addressed identity. The hash is the
dasher token of the node's *untagged*, snapshot-normalized representation, with
per-type special-casing so structurally-equal nodes collapse to one identity.

Shared by ``ibis_yaml`` (expr.yaml node labels / ``snapshot_hash``) and lineage
extraction, so a node keys identically in both artifacts and can be
cross-referenced by hash.
"""

from __future__ import annotations

from typing import Any

import xorq.vendor.ibis.expr.operations as ops
from xorq.caching.strategy import SnapshotStrategy
from xorq.common.utils.dasher import tokenize
from xorq.expr.relations import CacheTag, HashingTag, Read, Tag
from xorq.vendor.ibis.expr.schema import Schema


def content_hash(node: Any) -> str:
    """Return the content hash of *node*.

    The hash is derived purely from ``node``, so the two documented callers
    (``ibis_yaml`` serialization and lineage extraction) compute the same value
    for the same node and it can be cross-referenced by hash. A plain ``Tag``
    folds in its raw ``node.metadata`` -- never a serialization-specific form --
    so both callers agree.
    """
    # Schema is not a graph node and has no to_expr(); hash it directly.
    if isinstance(node, Schema):
        return tokenize(node)

    strategy = SnapshotStrategy()
    expr = node.to_expr()
    match node:
        case HashingTag():
            tagged_repr = expr.ls.untagged
            with strategy.normalization_context(expr):
                return tokenize(tagged_repr)
        case CacheTag():
            untagged_repr = ("CacheTag", node.parent.to_expr(), node.uncached)
            with strategy.normalization_context(expr):
                return tokenize(untagged_repr)
        case Tag():
            untagged_repr = ("Tag", node.parent.to_expr(), node.metadata)
            with strategy.normalization_context(expr):
                return tokenize(untagged_repr)
        case ops.JoinReference():
            with strategy.normalization_context(expr):
                parent_hash = tokenize(expr.ls.untagged)
            return tokenize((parent_hash, node.identifier))
        case Read():
            # Include node.name so two Reads with identical content but different
            # table names get distinct identities (prevents silent dedup).
            untagged_repr = expr.ls.untagged
            with strategy.normalization_context(expr):
                return tokenize((untagged_repr, node.name))
        case _:
            untagged_repr = expr.ls.untagged
            with strategy.normalization_context(expr):
                return tokenize(untagged_repr)
