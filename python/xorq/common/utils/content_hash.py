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


def content_hash(node: Any, *, tag_metadata: Any = None) -> str:
    """Return the content hash of *node*.

    ``tag_metadata`` overrides the metadata folded into a plain ``Tag``'s hash.
    ``ibis_yaml`` passes the already-serialized ``node_dict["metadata"]`` here so
    expr.yaml hashes stay byte-identical; callers without it (lineage) fall back
    to the node's raw ``metadata``. Only the plain ``Tag`` branch consults it;
    every other node type derives its hash purely from ``node``.
    """
    match node:
        case HashingTag():
            tagged_repr = node.to_expr().ls.untagged
            with SnapshotStrategy().normalization_context(node.to_expr()):
                return tokenize(tagged_repr)
        case CacheTag():
            untagged_repr = ("CacheTag", node.parent.to_expr(), node.uncached)
            with SnapshotStrategy().normalization_context(node.to_expr()):
                return tokenize(untagged_repr)
        case Tag():
            metadata = tag_metadata if tag_metadata is not None else node.metadata
            untagged_repr = ("Tag", node.parent.to_expr(), metadata)
            with SnapshotStrategy().normalization_context(node.to_expr()):
                return tokenize(untagged_repr)
        case Schema():
            return tokenize(node)
        case ops.JoinReference():
            parent_expr = node.to_expr()
            with SnapshotStrategy().normalization_context(parent_expr):
                parent_hash = tokenize(parent_expr.ls.untagged)
            return tokenize((parent_hash, node.identifier))
        case Read():
            # Include node.name so two Reads with identical content but different
            # table names get distinct identities (prevents silent dedup).
            untagged_repr = node.to_expr().ls.untagged
            with SnapshotStrategy().normalization_context(node.to_expr()):
                return tokenize((untagged_repr, node.name))
        case _:
            untagged_repr = node.to_expr().ls.untagged
            with SnapshotStrategy().normalization_context(node.to_expr()):
                return tokenize(untagged_repr)
