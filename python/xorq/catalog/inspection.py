"""Read a build record's external source leaves without loading its expression.

A catalog entry freezes its source schemas at build time. This module reads them
back out of the archive so a later pass can compare them against the world now.

Extraction is pure dict traversal over the serialized record: it constructs no
xorq node, unpickles no UDF, imports no backend. It reads two archive members,
``expr.yaml`` and ``profiles.yaml``, both in place, so there is no tempdir to
clean up. That is what makes source reporting work on an entry whose expression
can no longer load at all (corrupt UDF blob, missing dependency, uninstalled
backend).

Leaves come from a reachability walk out of ``expression`` along ``node_ref``
edges rather than a scan of the flat node registry, so a node the registry holds
but the expression does not reach can never be reported. They deduplicate by
construction: the registry keys on content hash, so a table read twice in a
self-join is one node and one leaf, which is correct. It is one source.

The walk stops at a pin the way the build hash does. A ``CacheTag``'s
``uncached`` branch is the upstream the pin discarded (see
``graph_utils.exclusively_pinned_leaves``), so its sources are not this record's,
and the pin's ``parent`` reads the cache artifact, machine-local rather than a
user source, so it is reported drift-exempt. Both edges are pruned only *at the
pin*: a leaf also reachable from a live branch stays live and checkable.

The bundled/external split is ADR-0006's: the presence of the ``read_path`` key
is the signal. See xorq-labs/xorq#2293 for the epic this feeds.
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import TYPE_CHECKING, Any

import yaml12
from attr import field, frozen
from attr.validators import deep_iterable, in_, instance_of, optional

from xorq.catalog.enums import LeafKind, PinKey, PinOp
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
from xorq.ibis_yaml.utils import freeze
from xorq.vendor.ibis.expr.schema import Schema


if TYPE_CHECKING:
    from xorq.catalog.catalog import CatalogEntry
    from xorq.ibis_yaml.common import TranslationContext


# See `LeafKind` for why these stay string comparisons against the `op` field.
LEAF_OPS = frozenset(LeafKind)
# The registry sections `get_schema` can reach. `nodes` is deliberately absent:
# it holds the UDF pickle blobs this module never touches, and the translator's
# unbounded cache would pin whatever registry it is handed. `dtypes` is here for
# an archive that used dtype refs; `_datatype_to_yaml` inlines them when
# `context is None`, which is how `register_schema` writes them today.
SCHEMA_REGISTRY_KEYS = frozenset({RegistryEnum.dtypes, RegistryEnum.schemas})
# The `CacheTag` edges each walk cuts. The source walk keeps the pin's frozen
# read and drops the upstream it discarded; the live walk drops the pin whole, so
# their difference is what only the pin reaches. Mirrors
# `graph_utils._EXEC_EDGES` / `_SKIP_PINS_EDGES`.
SOURCE_WALK_PRUNED = frozenset({PinKey.UNCACHED})
LIVE_WALK_PRUNED = frozenset({PinKey.PARENT, PinKey.UNCACHED})


def get_doc_key(doc: dict, key: str) -> Any:
    """``doc[key]``, naming the missing member rather than raising ``KeyError``.

    A truncated or corrupt ``expr.yaml`` must fail loudly. A drift check that
    silently reports zero sources for an unreadable record is a false pass.
    """
    if (value := doc.get(key)) is None:
        raise ValueError(f"document has no {key}")
    return value


def get_nodes(doc: dict) -> dict:
    """``definitions.nodes`` of ``doc``; an empty registry is legal, a missing one is not."""
    definitions = get_doc_key(doc, DocKey.definitions)
    if (nodes := definitions.get(RegistryEnum.nodes)) is None:
        raise ValueError(f"{DocKey.definitions} has no {RegistryEnum.nodes}")
    return nodes


def translation_context(doc: dict) -> TranslationContext:
    """A ``TranslationContext`` over ``doc``'s definitions, for dtypes only.

    Reusing the translator's own registry renders a recorded schema with its
    parameters (``decimal(10, 2)``, ``array<int64>``, ``timestamp('UTC')``) for
    free. ``get_schema`` constructs dtypes only and never reaches a node def, so
    no UDF blob is touched.

    Passing only the sections schema resolution reads makes "no node def is
    reachable from the translator" structural rather than incidental, and bounds
    what the context retains: ``translate_from_yaml`` is an unbounded
    ``lru_cache`` keyed on the frozen context, so any registry handed to it is
    pinned for the life of the process, once per entry over a ``check-sources``
    sweep.
    """
    from xorq.ibis_yaml.common import Registry, TranslationContext  # noqa: PLC0415
    from xorq.ibis_yaml.compiler import _ensure_translate_registered  # noqa: PLC0415

    _ensure_translate_registered()
    definitions = get_doc_key(doc, DocKey.definitions)
    # A registry section a newer xorq added is not ours to pass along: drop it
    # rather than let `Registry.__init__` raise an unnamed TypeError.
    known = {k: v for k, v in definitions.items() if k in SCHEMA_REGISTRY_KEYS}
    return TranslationContext(registry=Registry(**known))


def walk_node_refs(doc: dict, pruned_at_pin: frozenset[str]) -> frozenset[str]:
    """Every ``node_ref`` reachable from ``expression``, cutting a pin's edges.

    ``definitions.nodes`` is flat; the graph lives in the ``node_ref`` values
    scattered through the node defs. Walking from the root rather than iterating
    the registry makes a phantom leaf impossible rather than merely unobserved.

    ``pruned_at_pin`` names the keys not descended *at a ``CacheTag`` only*.
    Everywhere else every value is descended, so a node the pin shares with a
    live branch is still reached.
    """
    nodes = get_nodes(doc)
    seen: set[str] = set()
    stack: list[Any] = [get_doc_key(doc, DocKey.expression)]
    while stack:
        cur = stack.pop()
        match cur:
            case dict():
                if (ref := cur.get(RefEnum.node_ref)) is not None and ref not in seen:
                    seen.add(ref)
                    if (node_def := nodes.get(ref)) is None:
                        raise ValueError(
                            f"node_ref {ref!r} has no definition in "
                            f"{DocKey.definitions}.{RegistryEnum.nodes}"
                        )
                    stack.append(node_def)
                pruned = pruned_at_pin if cur.get(NodeKey.op) == PinOp.CACHE_TAG else ()
                stack.extend(value for key, value in cur.items() if key not in pruned)
            case list() | tuple():
                stack.extend(cur)
            case _:
                pass
    return frozenset(seen)


def has_pin(doc: dict) -> bool:
    """Whether any node def is a ``CacheTag``.

    A registry scan, not a walk: it only gates the pair of walks whose difference
    is the pinned set, and a pin the expression cannot reach prunes nothing, so
    an over-broad ``True`` costs two walks with an empty difference and cannot
    change the answer.
    """
    return any(
        node_def.get(NodeKey.op) == PinOp.CACHE_TAG
        for node_def in get_nodes(doc).values()
        if isinstance(node_def, dict)
    )


def pinned_node_refs(doc: dict) -> frozenset[str]:
    """Refs reachable ONLY through a pin: the frozen cache-artifact reads.

    Same ``under_pin - live`` rule as ``graph_utils.exclusively_pinned_leaves``,
    so a node the pin shares with a live branch keeps its live status. Both walks
    are taken here rather than handed in: which prune set each ref set came from
    is the whole correctness of the difference, and ``has_pin`` already keeps the
    common pin-free record to one walk.
    """
    if not has_pin(doc):
        return frozenset()
    return walk_node_refs(doc, SOURCE_WALK_PRUNED) - walk_node_refs(
        doc, LIVE_WALK_PRUNED
    )


def get_bundle_kind(read_path: str) -> BundledSourceTypes | None:
    """Which bundle dir holds ``read_path``, or ``None`` when its prefix is unknown.

    An archive written by another version, or on Windows (where the separator is
    stored verbatim), must not abort the report for every other leaf. A bundled
    leaf is drift-exempt either way, so the kind is only informational.
    """
    prefix = str(read_path).replace("\\", "/").split("/")[0]
    try:
        return BundledSourceTypes(prefix)
    except ValueError:
        return None


def get_node_key(node_ref: str, node_def: dict, key: str) -> Any:
    """``node_def[key]``, naming the node rather than raising a bare ``KeyError``.

    Same contract as ``get_doc_key``: every required member of a node def is read
    through here, so a truncated record always fails with the ref that cannot be
    read.
    """
    if (value := node_def.get(key)) is None:
        raise ValueError(f"node {node_ref!r} has no {key}")
    return value


def get_read_kwargs(node_ref: str, node_def: dict) -> tuple[tuple, ...]:
    """``read_kwargs`` as ``(key, value)`` pairs, naming the node on a bad entry.

    The serialized form is a list of string-keyed pairs, and the shape is checked
    *before* anything is coerced, so every corrupt shape raises with the ref
    rather than only some of them. Left to ``dict()`` and ``tuple()`` these are
    the unattributable tracebacks this module exists to avoid: "dictionary update
    sequence element #0 has length 9" for a wrong-length entry, a bare
    ``TypeError`` for a scalar one, "unhashable type: 'list'" for a bad key. A
    mapping or a two-character string is worse still, passing a length test by
    coercing into a plausible-looking pair (``tuple({"a": 1, "b": 2})`` is
    ``("a", "b")``), and a fabricated kwarg is worse than a named failure.

    Only an *absent* key is legal: a ``DatabaseTable`` never carries one, while
    ``_read_to_yaml`` writes ``read_kwargs`` for every ``Read``. A
    present-but-null value is a damaged archive and raises rather than reading as
    empty, which would drop ``read_path`` and turn a bundled read into a
    plausible *external* leaf that a drift check then falsely flags.
    """
    entries = node_def.get(NodeKey.read_kwargs, ())
    if not isinstance(entries, (list, tuple)) or not all(
        isinstance(entry, (list, tuple))
        and len(entry) == 2
        and isinstance(entry[0], str)
        for entry in entries
    ):
        raise ValueError(
            f"node {node_ref!r} has a malformed {NodeKey.read_kwargs} entry"
        )
    return tuple(map(tuple, entries))


@frozen
class SourceLeaf:
    """One source as the build record describes it.

    ``recorded`` is the schema frozen at build time. ``bundled`` says the bytes
    already live inside the archive (a relocated read, a materialized
    memory-backend table, a memtable), so there is nothing outside to drift, and
    ``bundle_kind`` names which bundle holds them. ``pinned`` says the leaf is a
    ``CacheTag``'s frozen read of its cache artifact, a machine-local cache path
    rather than a user source, so equally nothing to check. Both are
    ``drift_exempt``.
    """

    node_ref = field(validator=instance_of(str))
    kind = field(validator=in_(tuple(LeafKind)))
    name = field(validator=instance_of(str))
    profile = field(validator=optional(instance_of(str)))
    bundled = field(validator=instance_of(bool))
    bundle_kind = field(validator=optional(in_(tuple(BundledSourceTypes))))
    pinned = field(validator=instance_of(bool))
    recorded = field(validator=optional(instance_of(Schema)))
    table = field(validator=optional(instance_of(str)))
    namespace = field(
        validator=deep_iterable(optional(instance_of(str)), instance_of(tuple)),
        converter=tuple,
    )
    method_name = field(validator=optional(instance_of(str)))
    read_kwargs = field(
        validator=deep_iterable(instance_of(tuple), instance_of(tuple)),
        # Deep-freeze: serialized values are often dicts (`columns`, `schema`)
        # and `@frozen` generates a `__hash__`, so a shallow tuple() would make
        # the leaf unhashable the moment a consumer puts it in a set.
        converter=lambda kwargs: tuple((k, freeze(v)) for k, v in kwargs),
    )

    @property
    def drift_exempt(self) -> bool:
        """Nothing outside the archive this leaf could drift against."""
        return self.bundled or self.pinned

    @classmethod
    def from_node_def(
        cls,
        node_ref: str,
        node_def: dict,
        context: TranslationContext,
        pinned: bool = False,
    ) -> SourceLeaf:
        kind = LeafKind(get_node_key(node_ref, node_def, NodeKey.op))
        read_kwargs = get_read_kwargs(node_ref, node_def)
        kw = dict(read_kwargs)
        # ADR-0006: the presence of `read_path` *is* the bundled signal. A
        # one-key test, not a path-prefix guess.
        read_path = kw.get(ReadKwarg.read_path)
        namespace = dict(node_def.get(NodeKey.namespace) or {})
        catalog = namespace.get(NamespaceKey.catalog)
        database = namespace.get(NamespaceKey.database)
        match kind:
            case LeafKind.DATABASE_TABLE:
                table = get_node_key(node_ref, node_def, NodeKey.table)
                name = ".".join(p for p in (catalog, database, table) if p)
            case LeafKind.READ:
                # `read_kwargs["table_name"]` is not the reliable spelling:
                # `make_read_kwargs` fills it by binding the backend method's
                # signature, so a backend naming that parameter differently omits
                # it, and `_read_to_yaml` only rewrites it on a uid rename.
                # `NodeKey.name` is the final table name and is written
                # unconditionally, which keeps this as populated as the
                # `DatabaseTable` branch's `table`. Read with `.get`: a node def
                # missing `name` is still readable here as long as it carries a
                # path, same as `path` below.
                table = kw.get(ReadKwarg.table_name) or node_def.get(NodeKey.name)
                # The recorded path, never the generated table name. For a
                # bundled read that is the bundle-relative `read_path` the
                # registry rewrote `hash_path` to, not the original source. Both
                # deferred-read constructors normalize their path parameter into
                # `hash_path`, so the fallback is defensive only, and must stay
                # lazy or a node def missing `name` would fail the normal path
                # too.
                path = kw.get(ReadKwarg.hash_path) or get_node_key(
                    node_ref, node_def, NodeKey.name
                )
                name = (
                    ", ".join(map(str, path))
                    if isinstance(path, (list, tuple))
                    else str(path)
                )
            # Unreachable while `LeafKind` has exactly the two members matched
            # above: `LeafKind(...)` on `op` already rejected anything else. It
            # fires only once `LeafKind` grows a member this match forgot, which
            # is why no test reaches it.
            case _:
                raise ValueError(f"no leaf extraction for kind {kind}")
        return cls(
            node_ref=node_ref,
            kind=kind,
            name=name,
            profile=node_def.get(NodeKey.profile),
            bundled=read_path is not None,
            bundle_kind=None if read_path is None else get_bundle_kind(read_path),
            pinned=pinned,
            recorded=context.get_schema(
                get_node_key(node_ref, node_def, RefEnum.schema_ref)
            ),
            table=table,
            namespace=(catalog, database),
            method_name=node_def.get(NodeKey.method_name),
            read_kwargs=read_kwargs,
        )


def iter_source_leaves(doc: dict) -> tuple[SourceLeaf, ...]:
    """Reachable ``DatabaseTable`` / ``Read`` node defs of ``doc``, in ref order."""
    context = translation_context(doc)
    nodes = get_nodes(doc)
    source_refs = walk_node_refs(doc, SOURCE_WALK_PRUNED)
    pinned = pinned_node_refs(doc)
    pairs = ((node_ref, nodes[node_ref]) for node_ref in sorted(source_refs))
    return tuple(
        SourceLeaf.from_node_def(node_ref, node_def, context, pinned=node_ref in pinned)
        for node_ref, node_def in pairs
        if get_node_key(node_ref, node_def, NodeKey.op) in LEAF_OPS
    )


def read_document(
    build_zip: BuildZip, dump_file: DumpFiles, empty_ok: bool = False
) -> dict:
    """One YAML member of ``build_zip``, parsed, named on failure.

    An empty or non-mapping member would otherwise reach the ``BuildRecord``
    validators as an unnamed ``TypeError``. ``empty_ok`` allows the one member
    that is legitimately empty, a record with no profiles.
    """
    doc = build_zip.read_dump_file(dump_file, yaml12.parse_yaml)
    if doc is None and empty_ok:
        return {}
    if not isinstance(doc, dict):
        raise ValueError(
            f"{dump_file} did not parse to a document, got {type(doc).__name__}"
        )
    return doc


@frozen
class BuildRecord:
    """The two archive members source inspection needs, read in place and parsed.

    Neither member is extracted to disk and neither is loaded into an expression,
    so a record whose UDF blob is corrupt still reports its sources.

    Both are deep-frozen on the way in. ``source_leaves`` is a
    ``cached_property``, so a plain dict would let a caller mutate the document
    after first access and silently keep the stale leaves; it would also make
    ``@frozen``'s generated ``__hash__`` raise on a ``dict`` field.
    ``FrozenOrderedDict`` subclasses ``dict``, so every ``.get`` / ``[]``
    traversal here reads it unchanged.

    That freeze is the one place the module pays for the whole document: it
    rebuilds and eagerly hashes every mapping, so ``definitions.nodes``, where
    the UDF pickle blobs live, is copied and hashed though no blob is ever read.
    Peak cost is ~2x the parsed doc, once per record, which a ``check-sources``
    sweep pays per entry. It buys a record that cannot go stale and can be used
    as a key. Profile a sweep with that in mind rather than expecting the
    traversal to be the only cost.
    """

    expr_doc = field(validator=instance_of(dict), converter=freeze)
    profiles = field(validator=instance_of(dict), converter=freeze)

    @cached_property
    def source_leaves(self) -> tuple[SourceLeaf, ...]:
        return iter_source_leaves(self.expr_doc)

    @property
    def external_leaves(self) -> tuple[SourceLeaf, ...]:
        """Leaves whose bytes live outside the archive, so can drift."""
        return tuple(leaf for leaf in self.source_leaves if not leaf.drift_exempt)

    @property
    def bundled_counts(self) -> tuple[tuple[BundledSourceTypes | None, int], ...]:
        """``(bundle_kind, count)`` pairs over the bundled leaves, sorted.

        An unknown kind sorts last, so a leaf written by another version cannot
        crash the summary alongside the kinds this version recognizes.
        """
        kinds = Counter(leaf.bundle_kind for leaf in self.source_leaves if leaf.bundled)
        return tuple(sorted(kinds.items(), key=lambda kv: (kv[0] is None, kv[0] or "")))

    def get_profile_dict(self, leaf: SourceLeaf) -> dict[str, Any] | None:
        """The serialized profile ``leaf`` needs to be reached, if it names one.

        ``None`` means the leaf names no profile. A leaf naming one the record
        does not hold is a dangling ref in a corrupt record, so it raises rather
        than degrading to the same ``None``.
        """
        if not leaf.profile:
            return None
        if (profile := self.profiles.get(leaf.profile)) is None:
            raise ValueError(
                f"node {leaf.node_ref!r} names profile {leaf.profile!r}, "
                f"which {DumpFiles.profiles} does not hold"
            )
        return profile

    @classmethod
    def from_build_zip(cls, build_zip: BuildZip) -> BuildRecord:
        (expr_doc, profiles) = (
            read_document(
                build_zip, dump_file, empty_ok=dump_file == DumpFiles.profiles
            )
            for dump_file in (DumpFiles.expr, DumpFiles.profiles)
        )
        return cls(expr_doc=expr_doc, profiles=profiles)

    @classmethod
    def from_catalog_entry(cls, catalog_entry: CatalogEntry) -> BuildRecord:
        if not catalog_entry.is_content_local:
            catalog_entry.fetch()
        return cls.from_build_zip(BuildZip(catalog_entry.catalog_path))


def get_source_leaves(catalog_entry: CatalogEntry) -> tuple[SourceLeaf, ...]:
    """Source leaves of ``catalog_entry``, without loading its expression."""
    return BuildRecord.from_catalog_entry(catalog_entry).source_leaves
