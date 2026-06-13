"""Pin caches into the expression DAG (cache-to-read).

A ``CachedNode`` holds its result in a parquet file outside the graph; a
cache miss silently recomputes the parent, so shipping a build ships the
*compute*. ``pin_caches`` executes each selected cache once and swaps the
``CachedNode`` for a deferred read on the cache parquet, wrapped in a pin
``Tag`` carrying the cache's provenance. The read is relocatable so
``xorq build`` packs the parquet into the artifact: a pinned build never
recomputes and loads lazily. The read is md5-normalized, so the pinned
expression's hash follows the pinned bytes rather than the upstream compute.

``keys`` selects individual caches by cache key; pinning an outer cache
absorbs caches nested inside it. Caches inside ``FlightExpr.unbound_expr``
are unreachable and cannot be pinned. ``unpin`` reverses a pin by rebuilding
the original compute from a recipe the pin tag carries (recovered via
``PinnedCache.from_tag_node``, the ``ls.builder`` protocol); the recovered
compute must read from portable sources to be rebuilt into a new artifact.

Caveat: the content-stable identity holds for the in-process expression.
After build + load the packed read is resolved into a table tokenized by its
path, so a cache left unpinned downstream of a pinned read (a partial
``keys`` pin) can re-key across loads; ``xorq pin`` reports caches it leaves
unpinned.
"""

import json
from pathlib import Path

from attr import (
    asdict,
    field,
    fields,
    frozen,
)
from attr.validators import (
    instance_of,
)

from xorq.caching.storage import ParquetStorage
from xorq.common.utils.dasher import tokenize
from xorq.common.utils.defer_utils import (
    deferred_read_parquet,
    normalize_read_path_md5sum,
)
from xorq.common.utils.file_utils import file_digest
from xorq.common.utils.graph_utils import find_all_sources, replace_nodes, walk_nodes
from xorq.expr.relations import CachedNode, Read, Tag
from xorq.vendor.ibis.expr.types import Expr


# a pin is a base ``Tag`` with this name carrying the provenance keys below
PIN_TAG = "pinned_cache"


@frozen
class PinInfo:
    """The provenance a pin ``Tag`` carries; field names are the metadata keys.

    ``content_token`` is the md5 file digest of the pinned parquet (the same
    value that names the packed ``reads/<digest>.parquet`` file), so verify is
    a cheap file-digest comparison rather than a re-materialization.
    """

    content_token: str = field(validator=instance_of(str))
    source_token: str = field(validator=instance_of(str))
    cache_key: str = field(validator=instance_of(str))

    def attach(self, expr, recipe):
        """Wrap *expr* in a pin ``Tag`` carrying this provenance and recipe."""
        return expr.tag(PIN_TAG, recipe=recipe, **asdict(self))

    @classmethod
    def field_names(cls):
        return tuple(f.name for f in fields(cls))

    @classmethod
    def from_tag_node(cls, tag_node):
        names = cls.field_names()
        if missing := set(names) - set(tag_node.metadata):
            raise ValueError(f"pin tag is missing keys {sorted(missing)}")
        return cls(**{name: tag_node.metadata[name] for name in names})


def _dump_recipe(expr) -> str:
    """Serialize *expr* (the pre-pin compute) to a self-contained string.

    Stored in the pin tag so ``unpin`` can rebuild the original expression
    without the source script. Carries the expression yaml plus its source
    profiles, the same pair the build artifact persists.
    """
    from xorq.ibis_yaml.compiler import (  # noqa: PLC0415
        YamlExpressionTranslator,
        _to_yaml_safe,
        dehydrate_cons,
    )

    payload = {
        # allow_relocate: the compute may itself contain nested pinned reads
        "expression": YamlExpressionTranslator.to_yaml(expr, allow_relocate=True),
        "profiles": dehydrate_cons(find_all_sources(expr)),
    }
    return json.dumps(_to_yaml_safe(payload), sort_keys=True)


def _load_recipe(recipe: str):
    from xorq.ibis_yaml.compiler import (  # noqa: PLC0415
        YamlExpressionTranslator,
        hydrate_cons,
    )

    payload = json.loads(recipe)
    return YamlExpressionTranslator.from_yaml(
        payload["expression"], profiles=hydrate_cons(payload["profiles"])
    )


def _is_pin_tag(node) -> bool:
    # the key check guards against an unrelated user tag reusing the name
    return (
        isinstance(node, Tag)
        and node.tag == PIN_TAG
        and set(PinInfo.field_names()) <= set(node.metadata)
    )


def pinned_tag_nodes(expr) -> tuple[Tag, ...]:
    """All pin ``Tag`` nodes in the expression graph, one per pinned cache."""
    return tuple(node for node in walk_nodes((Tag,), expr) if _is_pin_tag(node))


def pin_infos(expr) -> tuple[PinInfo, ...]:
    """Pin provenance for every pinned cache in *expr*."""
    return tuple(PinInfo.from_tag_node(node) for node in pinned_tag_nodes(expr))


def _pin_cached_node(op, relocate):
    storage = op.cache.storage
    if not isinstance(storage, ParquetStorage):
        raise ValueError(
            f"cache storage {type(storage).__name__} is not a local "
            "ParquetStorage; pin requires a local parquet cache file"
        )
    cached = op.to_expr()
    # the key and source token must be computed before executing: a parent
    # whose pickled closure mutates during execution (e.g. a counter)
    # tokenizes, and therefore keys, differently afterwards, while execution
    # populates the cache under the pre-execution key
    key = op.cache.calc_key(cached)
    source_token = tokenize(op.parent)
    # skip the materialization on a warm cache; the post-execute re-check
    # catches a fit whose tokenization drifted across execution
    if not storage.exists(key):
        cached.execute()
        if not storage.exists(key):
            raise ValueError(
                f"cache parquet not found at {storage.get_path(key)}: the "
                f"cache was populated under a different key than {key!r} (is "
                "the parent's tokenization unstable across execution?)"
            )
    path = storage.get_path(key)
    read = deferred_read_parquet(
        path,
        storage.source,
        table_name=f"pinned_{key}",
        schema=op.schema,
        normalize_method=normalize_read_path_md5sum,
        **({"relocate": True} if relocate else {}),
    )
    return PinInfo(
        content_token=file_digest(path),
        source_token=source_token,
        cache_key=key,
    ).attach(read, recipe=_dump_recipe(cached))


def pin_caches(expr, keys=None, relocate: bool = True):
    """Materialize each selected cache in *expr* and pin it into the DAG.

    Each selected ``CachedNode`` is executed once (hitting the cache
    parquet when it is already populated) and replaced by a deferred read
    on the cache parquet, wrapped in a pin ``Tag`` recording the cache's
    provenance. The resulting expression never recomputes the pinned
    subtree: the data is part of the program, and downstream cache keys
    derive from the pinned content rather than the upstream compute.

    Parameters
    ----------
    expr : ir.Expr
        Expression containing cached nodes.
    keys : iterable of str, optional
        Cache keys (``cache.calc_key``) selecting which caches to pin.
        Default None pins every cache backed by a local ``ParquetStorage``
        and skips the rest; an explicit key whose cache has non-parquet
        storage raises, as does a key matching no cache.
    relocate : bool, default True
        Mark the pinned read relocatable so ``xorq build`` copies the
        parquet inside the build artifact. Pass False to leave the read
        pointing at the cache path (e.g. on shared storage).

    Returns
    -------
    ir.Expr
        A new expression with the selected caches pinned.
    """
    pinned, _ = _pin_caches(expr, keys=keys, relocate=relocate)
    return pinned


def _pin_caches(expr, keys=None, relocate=True):
    """Pin caches and report how many *new* pins were created.

    Returns ``(pinned_expr, n_pinned)`` where ``n_pinned`` counts the
    distinct cached nodes newly pinned this call. Shared caches and caches
    absorbed by pinning an enclosing cache each count once, so the count is
    honest even when pinning an outer cache subsumes a nested pin (where a
    before/after pin-tag tally would under- or over-count).
    """
    keys = frozenset(keys) if keys is not None else None
    key_cache = {}

    def key_of(op):
        # memoized so the validation walk and selected() below don't each pay
        # the calc_key tokenization; an unkeyable cache (broken source) is None
        if op not in key_cache:
            try:
                key_cache[op] = op.cache.calc_key(op.to_expr())
            except Exception:
                key_cache[op] = None
        return key_cache[op]

    if keys is not None:
        # validate up front so a typo (or a key for a cache absorbed by an
        # enclosing pin) fails before any pin executes, not after
        available = {key_of(node) for node in walk_nodes((CachedNode,), expr)}
        if missing := keys - available:
            already = missing & {info.cache_key for info in pin_infos(expr)}
            if already:
                raise ValueError(
                    f"key(s) {sorted(already)} are already pinned; pin "
                    "operates on unpinned caches"
                )
            raise ValueError(f"no cached node matches key(s) {sorted(missing)}")

    memo = {}

    def selected(op):
        if keys is None:
            return isinstance(op.cache.storage, ParquetStorage)
        return key_of(op) in keys

    def replacer(op, kwargs):
        if kwargs:
            op = op.__recreate__(kwargs)
        if isinstance(op, CachedNode) and selected(op):
            if op not in memo:
                memo[op] = _pin_cached_node(op, relocate)
            return memo[op].op()
        return op

    pinned = replace_nodes(replacer, expr).to_expr()
    return pinned, len(memo)


@frozen
class PinVerification:
    cache_key: str = field(validator=instance_of(str))
    expected_token: str = field(validator=instance_of(str))
    actual_token: str = field(validator=instance_of(str))

    @property
    def ok(self):
        return self.expected_token == self.actual_token


def _pinned_parquet_path(tag_node, info, reads_dir):
    """The on-disk parquet to digest for *tag_node*, or None if unavailable.

    With *reads_dir* (a build's packed ``reads/`` dir) the file is named by
    its digest: ``reads_dir/<content_token>.parquet``. Otherwise fall back to
    the pinned read's own path, which is present pre-build but gone once the
    read has been resolved into a registered table on load.
    """
    if reads_dir is not None:
        return Path(reads_dir) / f"{info.content_token}.parquet"
    parent = tag_node.parent
    if isinstance(parent, Read):
        hash_path = dict(parent.read_kwargs).get("hash_path")
        if isinstance(hash_path, (list, tuple)):
            hash_path = hash_path[0] if hash_path else None
        if hash_path is not None:
            return Path(hash_path)
    return None


def _verify_one(tag_node, reads_dir) -> PinVerification:
    info = PinInfo.from_tag_node(tag_node)
    path = _pinned_parquet_path(tag_node, info, reads_dir)
    actual = file_digest(path) if path is not None and path.is_file() else ""
    return PinVerification(
        cache_key=info.cache_key,
        expected_token=info.content_token,
        actual_token=actual,
    )


def verify_pinned(expr, reads_dir=None) -> tuple[PinVerification, ...]:
    """Compare each pinned cache's parquet digest against the pinned value.

    Returns a tuple of ``PinVerification``, one per pinned cache. A mismatch
    (or a missing file, reported as an empty actual token) means the parquet
    behind the pinned read no longer holds the bytes that were pinned (the
    file was rewritten or corrupted); the pin's tag metadata remains the
    authority on what was pinned.

    Parameters
    ----------
    reads_dir : path-like, optional
        A build's packed ``reads/`` directory. Required to verify an
        expression loaded from a build (where the pinned read has been
        resolved into a registered table and no longer carries a path); for
        an in-process pinned expression the read's own path is used.
    """
    return tuple(
        _verify_one(tag_node, reads_dir) for tag_node in pinned_tag_nodes(expr)
    )


@frozen
class PinnedCache:
    """A pinned cache recovered from its tag: provenance plus the original.

    ``expr`` is the pre-pin compute rebuilt from the tag's recipe, so
    ``unpin`` can swap the pinned read back for the expression that produced
    it. Recovered via :meth:`from_tag_node` (the ``ls.builder`` protocol).
    """

    info: PinInfo = field(validator=instance_of(PinInfo))
    expr: Expr = field(validator=instance_of(Expr))

    @classmethod
    def from_tag_node(cls, tag_node):
        recipe = tag_node.metadata.get("recipe")
        if recipe is None:
            raise ValueError("pin tag carries no recipe; cannot unpin")
        return cls(info=PinInfo.from_tag_node(tag_node), expr=_load_recipe(recipe))


def unpin(expr):
    """Swap every pinned read back for the compute that produced it.

    The inverse of :func:`pin_caches`: each pin tag is replaced by the
    original ``CachedNode`` rebuilt from the tag's recipe, so executing the
    result recomputes (and re-caches) rather than reading the pinned file.
    Returns *expr* unchanged when nothing is pinned.
    """
    tag_nodes = pinned_tag_nodes(expr)
    if not tag_nodes:
        return expr
    originals = {node: PinnedCache.from_tag_node(node).expr.op() for node in tag_nodes}

    def replacer(op, kwargs):
        if op in originals:
            return originals[op]
        if kwargs:
            return op.__recreate__(kwargs)
        return op

    return replace_nodes(replacer, expr).to_expr()
