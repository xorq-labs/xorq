"""Content hash for expression-graph nodes.

A node's identity is its serialized form: the yaml dict ``ibis_yaml`` builds
for it, in which every child relation is already a reference named by the
child's hash and everything else is a primitive. ``Registry.register_node``
names a node by ``tokenize(node_dict)``, the scheme it already uses for dtypes
and schemas, so hashing every node of an expression costs what serializing it
costs: one translation, children first. :func:`content_hash` translates a node
and reads that hash back, so ``ibis_yaml`` (expr.yaml node labels /
``snapshot_hash``) and lineage extraction key a node identically and can
cross-reference it by hash.

Leaves keep the identity their handlers give them (a backend table its
profile and name, a ``Read`` its kwargs, a memtable its contents). A value op
(``Field``, ``JoinLink``, ...) is serialized inline rather than registered and
hashes the same way. A ``Tag`` is a node like any other, so a tagged relation
and its untagged twin get different hashes.
"""

from __future__ import annotations

from typing import Any

from xorq.caching.strategy import SnapshotStrategy
from xorq.common.utils.dasher import tokenize
from xorq.ibis_yaml.enums import NodeKey, RefEnum
from xorq.vendor.ibis.common.graph import Node
from xorq.vendor.ibis.expr.schema import Schema


def content_hash(node: Node | Schema, context: Any = None) -> str:
    """Return the content hash of *node*.

    Parameters
    ----------
    node
        An operation node, or a ``Schema`` (hashed directly).
    context
        A ``TranslationContext`` to translate through. Share one across the
        nodes of an expression so the graph is translated once; a fresh one
        is used otherwise.

    Contract
    --------
    Raises what the node's yaml handler raises (``NotImplementedError`` for a
    type with no translation rule). Translation recurses into the node's
    inputs, so a chain of a few hundred ops needs the serializer's
    children-first warm-up to stay within the stack.
    """
    if isinstance(node, Schema):
        return tokenize(node)
    # ibis_yaml imports this module; resolve the cycle at call time
    import xorq.ibis_yaml.translate  # noqa: PLC0415, F401
    from xorq.ibis_yaml.common import TranslationContext  # noqa: PLC0415

    context = TranslationContext() if context is None else context
    with SnapshotStrategy().normalization_context(node):
        translated = context.translate_to_yaml(node)
    match translated:
        case {RefEnum.node_ref: ref}:
            return context.registry.nodes[ref][NodeKey.snapshot_hash]
        case _:
            return tokenize(translated)
