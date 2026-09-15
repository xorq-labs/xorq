"""Read a build record's external source leaves without loading its expression.

A catalog entry freezes its source schemas at build time.  ``inspection`` reads
those frozen sources back out of the archive so a later pass can compare them
against the world as it is now.

Extraction is pure dict traversal over the serialized build record.  It
constructs no xorq expression node, unpickles no UDF, and imports no backend.
Only two members of the archive are needed -- ``expr.yaml`` and
``profiles.yaml`` -- and both are read in place, so there is no tempdir to
create and none to clean up.  That is what makes source reporting work on an
entry whose expression can no longer load at all (a corrupt UDF blob, a
missing dependency, a backend that is no longer installed).

Leaves are enumerated by a reachability walk from ``expression`` through
``node_ref`` edges rather than by scanning the flat node registry, so a node
the registry holds but the expression does not reach can never be reported.
They deduplicate by construction: the registry keys on content hash, so a
table read twice in a self-join is one node and one leaf.  That is correct --
it is one source.

The bundled/external split is ADR-0006's: the presence of the ``read_path``
key is the signal.  See xorq-labs/xorq#2293 for the epic this feeds.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml12
from attr import field, frozen
from attr.validators import deep_iterable, in_, instance_of, optional

from xorq.catalog.enums import LeafKind
from xorq.catalog.zip_utils import BuildZip
from xorq.ibis_yaml.enums import (
    BundledSourceTypes,
    DocKey,
    DumpFiles,
    NamespaceKey,
    NodeKey,
    ReadKwarg,
    RefEnum,
    RegistryEnum,
)
from xorq.vendor.ibis.expr.schema import Schema


if TYPE_CHECKING:
    from xorq.catalog.catalog import CatalogEntry
    from xorq.ibis_yaml.common import TranslationContext


# The `op` field holds the exact class name the translator serialized.  Keep
# this a string comparison: `Read`, `DatabaseTableView`, `CachedNode`,
# `RemoteTable`, `FlightExpr` and `FlightUDXF` all subclass
# `ops.DatabaseTable`, so an `isinstance(node, ops.DatabaseTable)` test
# silently swallows caches, cross-backend transfers and Flight nodes -- none of
# which are external sources.
LEAF_OPS = frozenset(LeafKind)


def translation_context(doc: dict) -> TranslationContext:
    """A ``TranslationContext`` over ``doc``'s definitions, for dtypes only.

    Reusing the translator's own registry is what makes a recorded schema
    render its parameters -- ``decimal(10, 2)``, ``array<int64>``,
    ``timestamp('UTC')`` -- for free.  ``get_schema`` constructs dtypes only;
    it never reaches a node def, so no UDF blob is touched.
    """
    from xorq.ibis_yaml.common import Registry, TranslationContext  # noqa: PLC0415
    from xorq.ibis_yaml.compiler import _ensure_translate_registered  # noqa: PLC0415

    _ensure_translate_registered()
    return TranslationContext(registry=Registry(**doc[DocKey.definitions]))


def reachable_node_refs(doc: dict) -> tuple[str, ...]:
    """Every ``node_ref`` reachable from ``expression``, in sorted order.

    ``definitions.nodes`` is flat; the graph lives in the ``node_ref`` values
    scattered through the node defs.  Walking from the root rather than
    iterating the registry makes a phantom leaf impossible rather than merely
    unobserved.
    """
    nodes = doc[DocKey.definitions][RegistryEnum.nodes]
    seen: set[str] = set()
    stack: list[Any] = [doc[DocKey.expression]]
    while stack:
        cur = stack.pop()
        match cur:
            case dict():
                if (ref := cur.get(RefEnum.node_ref)) is not None and ref not in seen:
                    seen.add(ref)
                    stack.append(nodes[ref])
                stack.extend(cur.values())
            case list() | tuple():
                stack.extend(cur)
            case _:
                pass
    return tuple(sorted(seen))


@frozen
class SourceLeaf:
    """One source as the build record describes it.

    ``recorded`` is the schema frozen at build time.  ``bundled`` says the
    bytes already live inside the archive (a relocated read, a materialized
    memory-backend table, a memtable), so there is nothing outside to drift,
    and ``bundle_kind`` names which bundle holds them.
    """

    node_ref = field(validator=instance_of(str))
    kind = field(validator=in_(tuple(LeafKind)))
    name = field(validator=instance_of(str))
    profile = field(validator=optional(instance_of(str)))
    bundled = field(validator=instance_of(bool))
    bundle_kind = field(validator=optional(in_(tuple(BundledSourceTypes))))
    recorded = field(validator=optional(instance_of(Schema)))
    table = field(validator=optional(instance_of(str)))
    namespace = field(
        validator=deep_iterable(optional(instance_of(str)), instance_of(tuple)),
        converter=tuple,
    )
    method_name = field(validator=optional(instance_of(str)))
    read_kwargs = field(
        validator=deep_iterable(instance_of(tuple), instance_of(tuple)),
        converter=lambda kwargs: tuple(map(tuple, kwargs)),
    )

    @classmethod
    def from_node_def(
        cls, node_ref: str, node_def: dict, context: TranslationContext
    ) -> SourceLeaf:
        kind = LeafKind(node_def[NodeKey.op])
        read_kwargs = tuple(map(tuple, node_def.get(NodeKey.read_kwargs, ())))
        kw = dict(read_kwargs)
        # ADR-0006: the presence of `read_path` *is* the bundled signal -- a
        # one-key test, not a path-prefix guess.
        read_path = kw.get(ReadKwarg.read_path)
        namespace = dict(node_def.get(NodeKey.namespace) or {})
        catalog = namespace.get(NamespaceKey.catalog)
        database = namespace.get(NamespaceKey.database)
        match kind:
            case LeafKind.database_table:
                table = node_def[NodeKey.table]
                name = ".".join(p for p in (catalog, database, table) if p)
            case LeafKind.read:
                table = kw.get(ReadKwarg.table_name)
                # The path or URI, never the generated table name.  Both
                # deferred-read constructors normalize their path parameter
                # into `hash_path`, so the fallback is defensive only.
                name = str(kw.get(ReadKwarg.hash_path, node_def[NodeKey.name]))
        return cls(
            node_ref=node_ref,
            kind=kind,
            name=name,
            profile=node_def.get(NodeKey.profile),
            bundled=read_path is not None,
            bundle_kind=(
                None
                if read_path is None
                else BundledSourceTypes(Path(read_path).parts[0])
            ),
            recorded=context.get_schema(node_def[RefEnum.schema_ref]),
            table=table,
            namespace=(catalog, database),
            method_name=node_def.get(NodeKey.method_name),
            read_kwargs=read_kwargs,
        )


def iter_source_leaves(doc: dict) -> tuple[SourceLeaf, ...]:
    """Reachable ``DatabaseTable`` / ``Read`` node defs of ``doc``, in ref order."""
    context = translation_context(doc)
    nodes = doc[DocKey.definitions][RegistryEnum.nodes]
    return tuple(
        SourceLeaf.from_node_def(node_ref, nodes[node_ref], context)
        for node_ref in reachable_node_refs(doc)
        if nodes[node_ref][NodeKey.op] in LEAF_OPS
    )


@frozen
class BuildRecord:
    """The two archive members source inspection needs, read in place and parsed.

    Neither member is extracted to disk and neither is loaded into an
    expression, so a record whose UDF blob is corrupt still reports its
    sources.
    """

    expr_doc = field(validator=instance_of(dict))
    profiles = field(validator=instance_of(dict))

    @cached_property
    def source_leaves(self) -> tuple[SourceLeaf, ...]:
        return iter_source_leaves(self.expr_doc)

    @property
    def external_leaves(self) -> tuple[SourceLeaf, ...]:
        """Leaves whose bytes live outside the archive, so can drift."""
        return tuple(leaf for leaf in self.source_leaves if not leaf.bundled)

    @property
    def bundled_counts(self) -> tuple[tuple[str, int], ...]:
        """``(bundle_kind, count)`` pairs over the bundled leaves, sorted."""
        kinds = tuple(leaf.bundle_kind for leaf in self.source_leaves if leaf.bundled)
        return tuple(sorted((kind, kinds.count(kind)) for kind in set(kinds)))

    def get_profile_dict(self, leaf: SourceLeaf) -> dict[str, Any] | None:
        """The serialized profile ``leaf`` needs to be reached, if it names one."""
        return self.profiles.get(leaf.profile) if leaf.profile else None

    @classmethod
    def from_build_zip(cls, build_zip: BuildZip) -> BuildRecord:
        prefix = build_zip.internal_prefix
        (expr_doc, profiles) = (
            build_zip.read_member(f"{prefix}/{dump_file}", yaml12.parse_yaml)
            for dump_file in (DumpFiles.expr, DumpFiles.profiles)
        )
        return cls(expr_doc=expr_doc, profiles=profiles or {})

    @classmethod
    def from_catalog_entry(cls, catalog_entry: CatalogEntry) -> BuildRecord:
        if not catalog_entry.is_content_local:
            catalog_entry.fetch()
        return cls.from_build_zip(BuildZip(catalog_entry.catalog_path))


def get_source_leaves(catalog_entry: CatalogEntry) -> tuple[SourceLeaf, ...]:
    """Source leaves of ``catalog_entry``, without loading its expression."""
    return BuildRecord.from_catalog_entry(catalog_entry).source_leaves
