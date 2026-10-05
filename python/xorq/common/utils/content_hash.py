"""Content hash for expression-graph nodes.

Single source of truth for a node's content-addressed identity. The hash is the
dasher token of the node's *untagged*, snapshot-normalized representation, with
per-type special-casing so structurally-equal nodes collapse to one identity.

Consumed by ``ibis_yaml`` (expr.yaml node labels / ``snapshot_hash``) and by
lineage extraction (``lineage_utils``), so a relation keys identically in both
artifacts and can be cross-referenced by hash. Lineage only calls this for
relations; value ops (``Field``, ``SortKey``, …) never appear in expr.yaml and are
not all normalizable, so they get a structural token instead.

The hash is *incremental*: a relation is hashed in isolation, with each child
relation swapped for a placeholder table named by that child's own content
hash (``PLACEHOLDER_PREFIX``). Snapshot normalization (``SnapshotStrategy``)
therefore compiles one level of SQL per node instead of the whole subtree, and
hashing every relation of an expression costs time linear in its size. A
:class:`ContentHasher` memoizes across calls; its children-first fill is
iterative, so stack depth does not grow with the expression either.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any

from attrs import field, frozen
from attrs.validators import instance_of

import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.operations as ops
from xorq.caching.strategy import SnapshotStrategy
from xorq.common.utils.dasher import fqn, tokenize
from xorq.common.utils.graph_utils import (
    BACKEND_LEAF_NODE_TYPES,
    OPAQUE_EDGES,
    child_relations,
    get_ordered_unique_sources,
)
from xorq.expr.relations import CacheTag, HashingTag, Read, Tag
from xorq.vendor.ibis.common.graph import Graph, Node
from xorq.vendor.ibis.expr.schema import Schema
from xorq.vendor.ibis.expr.types import Expr


PLACEHOLDER_PREFIX = "__xorq_content_hash_"

# ``ExprScalarUDF.computed_kwargs_expr`` lives in ``__config__``, not in
# ``__args__``, so the isolation rewrite cannot swap it for a placeholder; its
# (data-free) identity is folded in by the UDF's own dasher rule instead. Treat
# the UDF as a leaf here so that edge is neither hashed separately nor half-rewritten.
HASH_EDGES = {**OPAQUE_EDGES, udf.ExprScalarUDF: ()}


def _is_transparent_tag(node: Node) -> bool:
    """A plain ``Tag`` is stripped before hashing; ``HashingTag``/``CacheTag`` are not."""
    return isinstance(node, Tag) and not isinstance(node, (HashingTag, CacheTag))


def through_tags(node: Node) -> Node:
    """The nearest ancestor-or-self that survives ``expr.ls.untagged``."""
    while _is_transparent_tag(node):
        node = node.parent
    return node


class ContentHashPlaceholder(ops.DatabaseTable):
    """Stand-in leaf for an already-hashed child relation (see :func:`_placeholder`).

    A ``DatabaseTable`` so that it carries a backend (compiler selection and
    the snapshot hasher's per-backend rules see what the real subtree has),
    but normalized by :func:`normalize_content_hash_placeholder` rather than
    as a real table: its name already *is* the child's identity, and a real
    table's normalization may introspect the backend (Redshift resolves an
    unqualified table's schema with a query), which must not run once per
    relation for tables that do not exist.
    """


def normalize_content_hash_placeholder(dt: ContentHashPlaceholder) -> tuple:
    return ("content-hash-placeholder", dt.name, dt.schema)


@frozen
class _PlaceholderSnapshotStrategy(SnapshotStrategy):
    """Snapshot normalization plus the placeholder rule, declared first so it
    wins the MRO tie against the ``DatabaseTable`` rule."""

    def declared_rules(self) -> tuple:
        return (
            (fqn(ContentHashPlaceholder), normalize_content_hash_placeholder),
            *super().declared_rules(),
        )


def _own_backends(node: Node) -> tuple:
    if isinstance(node, BACKEND_LEAF_NODE_TYPES):
        return get_ordered_unique_sources((node,))
    return ()


def _placeholder(child: Node, hashes: Mapping, backends: Mapping) -> Node:
    """A stand-in for *child*: an identity ``Project`` over a leaf table named by
    the child's content hash.

    The table carries one of the child's backends when it has any, so compiler
    (dialect) selection and the snapshot hasher's per-backend rules see the
    same backends they would on the full subtree. The ``Project`` gives the
    stand-in ``values`` (a bare table has none), which subquery ops read.
    """
    name = f"{PLACEHOLDER_PREFIX}{hashes[through_tags(child)]}"
    if cons := backends[child]:
        table = ContentHashPlaceholder(name=name, schema=child.schema, source=cons[0])
    else:
        table = ops.UnboundTable(name=name, schema=child.schema)
    return ops.Project(
        parent=table, values={name: ops.Field(table, name) for name in child.schema}
    )


def isolate(node: Node, placeholders: Mapping[Node, Node]) -> Node:
    """Rebuild *node* with every relation in *placeholders* swapped for its stand-in.

    Value ops and ``Reference`` wrappers (``JoinReference``) are rebuilt on the
    way down; a relation not in *placeholders* (such as a Flight op's
    ``unbound_expr`` root) is left as it is, never descended, so the rewrite
    touches one level of the relation graph.
    """
    memo: dict = {}

    def rewrite(value: Any) -> Any:
        match value:
            case Node() if value in placeholders:
                return placeholders[value]
            case ops.Relation() if not isinstance(value, ops.Reference):
                return value
            case Node():
                if value not in memo:
                    memo[value] = value.__recreate__(
                        {
                            name: rewrite(arg)
                            for name, arg in zip(value.__argnames__, value.__args__)
                        }
                    )
                return memo[value]
            case Expr():
                return rewrite(value.op()).to_expr()
            case dict():
                return type(value)((key, rewrite(v)) for key, v in value.items())
            case tuple() | list():
                return type(value)(rewrite(v) for v in value)
            case _:
                return value

    return node.__recreate__(
        {name: rewrite(arg) for name, arg in zip(node.__argnames__, node.__args__)}
    )


def _reference_target(rel: Node) -> Node:
    """The relation a ``Reference`` wrapper (``JoinReference``) stands for.

    References are rebuilt around a placeholder rather than replaced by one,
    since ``JoinChain`` requires them; the placeholder goes on their parent.
    """
    while isinstance(rel, ops.Reference):
        rel = rel.parent
    return rel


def _hash_isolated(node: Node, hashes: Mapping, backends: Mapping) -> str:
    """Hash *node* given the hashes (and backends) of every relation it refers to."""
    match node:
        case ops.JoinReference():
            return tokenize((hashes[through_tags(node.parent)], node.identifier))
        case CacheTag():
            return tokenize(
                ("CacheTag", hashes[node.parent], hashes[node.uncached.op()])
            )
        case Tag() if not isinstance(node, HashingTag):
            return tokenize(("Tag", hashes[through_tags(node.parent)], node.metadata))
    targets = dict.fromkeys(
        _reference_target(child)
        for child in child_relations(node, opaque_edges=HASH_EDGES)
    )
    placeholders = {rel: _placeholder(rel, hashes, backends) for rel in targets}
    expr = isolate(node, placeholders).to_expr()
    match node:
        case Read():
            # Include node.name so two Reads with identical content but different
            # table names get distinct identities (prevents silent dedup).
            untagged_repr = (expr.ls.untagged, node.name)
        case _:
            untagged_repr = expr.ls.untagged
    with _PlaceholderSnapshotStrategy().normalization_context(expr):
        return tokenize(untagged_repr)


def _dedupe_backends(cons: tuple) -> tuple:
    # id()-based like get_ordered_unique_sources: backend __eq__/__hash__ are
    # not identity-safe across same-class instances.
    seen: set = set()
    out = ()
    for con in cons:
        if id(con) not in seen:
            seen.add(id(con))
            out += (con,)
    return out


@frozen
class ContentHasher:
    """Memoizing ``node -> content hash`` for one expression (or many sharing nodes).

    Call it on any relation: it fills the memo for that relation's whole
    subtree children-first and iteratively, so neither time nor stack depth
    grows faster than the number of relations.
    """

    hashes: dict = field(factory=dict, validator=instance_of(dict))
    backends: dict = field(factory=dict, validator=instance_of(dict))

    def __call__(self, node: Any) -> str:
        # Schema is not a graph node and has no to_expr(); hash it directly.
        if isinstance(node, Schema):
            return tokenize(node)
        if node not in self.hashes:
            self._fill(node)
        return self.hashes[node]

    def _fill(self, root: Node) -> None:
        children_of: dict[Node, tuple] = {}
        queue = deque((root,))
        while queue:
            node = queue.popleft()
            if node in children_of or node in self.hashes:
                continue
            children_of[node] = child_relations(node, opaque_edges=HASH_EDGES)
            queue.extend(children_of[node])
        pending = Graph(
            {
                node: tuple(child for child in children if child not in self.hashes)
                for node, children in children_of.items()
            }
        )
        ordered, _ = pending.toposort()
        for node in ordered:
            inherited = tuple(
                con for child in children_of[node] for con in self.backends[child]
            )
            self.backends[node] = _dedupe_backends(_own_backends(node) + inherited)
            self.hashes[node] = _hash_isolated(node, self.hashes, self.backends)


def content_hash(node: Any) -> str:
    """Return the content hash of *node*.

    The hash is derived purely from ``node``, so any caller (``ibis_yaml``
    serialization, lineage extraction) computes the same value for the same
    node and it can be cross-referenced by hash. A plain ``Tag`` folds in its
    raw ``node.metadata`` -- never a serialization-specific form -- so callers
    agree regardless of how they reach this helper.

    To hash many nodes of one expression, share a :class:`ContentHasher` so
    each relation is hashed once.
    """
    return ContentHasher()(node)
