import difflib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

import yaml
from attrs import asdict, field, frozen
from attrs.validators import deep_iterable, instance_of, optional
from toolz import curry

import xorq as xo
from xorq.ibis_yaml.compiler import load_expr as _load_expr
from xorq.vendor.ibis import Expr


@frozen
class CatalogMetadata:
    """Catalog metadata."""

    catalog_id: str = field(validator=instance_of(str))
    created_at: str = field(validator=instance_of(str))
    updated_at: str = field(validator=instance_of(str))
    tool_version: str = field(validator=instance_of(str))

    def with_updated_timestamp(self) -> "CatalogMetadata":
        """Return new metadata with updated timestamp."""
        return self.clone(updated_at=datetime.now(timezone.utc).isoformat())

    def clone(self, **kwargs) -> "CatalogMetadata":
        """Create a copy with specified changes."""
        return type(self)(**asdict(self) | kwargs)


@frozen
class Build:
    """Build information."""

    build_id: Optional[str] = field(default=None, validator=optional(instance_of(str)))
    path: Optional[str] = field(default=None, validator=optional(instance_of(str)))

    def clone(self, **kwargs) -> "Build":
        return type(self)(**asdict(self) | kwargs)


@frozen
class Revision:
    """Revision data."""

    revision_id: str = field(validator=instance_of(str))
    created_at: str = field(validator=instance_of(str))
    build: Optional[Build] = field(default=None, validator=optional(instance_of(Build)))
    expr_hashes: Optional[Dict[str, str]] = field(default=None)
    node_hashes: Tuple[str, ...] = field(
        factory=tuple, validator=deep_iterable(instance_of(str), instance_of(tuple))
    )
    meta_digest: Optional[str] = field(
        default=None, validator=optional(instance_of(str))
    )
    metadata: Optional[Dict[str, Any]] = field(default=None)
    schema_fingerprint: Optional[str] = field(
        default=None, validator=optional(instance_of(str))
    )

    def clone(self, **kwargs) -> "Revision":
        return type(self)(**asdict(self) | kwargs)


@frozen
class Entry:
    """Entry in the catalog."""

    entry_id: str = field(validator=instance_of(str))
    current_revision: Optional[str] = field(
        default=None, validator=optional(instance_of(str))
    )
    history: Tuple[Revision, ...] = field(
        factory=tuple,
        validator=deep_iterable(instance_of(Revision), instance_of(tuple)),
    )

    def with_revision(self, revision: Revision) -> "Entry":
        """Add a new revision to history."""
        return self.clone(
            current_revision=revision.revision_id, history=self.history + (revision,)
        )

    def maybe_get_revision(self, revision_id: str) -> Optional[Revision]:
        """Get revision by ID if it exists."""
        return next((r for r in self.history if r.revision_id == revision_id), None)

    def maybe_current_revision(self) -> Optional[Revision]:
        """Get current revision if it exists."""
        if self.current_revision is None:
            return None
        return self.maybe_get_revision(self.current_revision)

    def clone(self, **kwargs) -> "Entry":
        return type(self)(**asdict(self) | kwargs)


@frozen
class Alias:
    entry_id: str = field(validator=instance_of(str))
    revision_id: Optional[str] = field(
        default=None, validator=optional(instance_of(str))
    )

    def clone(self, **kwargs) -> "Alias":
        return type(self)(**asdict(self) | kwargs)


@frozen
class XorqCatalog:
    """Xorq Catalog container."""

    metadata: CatalogMetadata = field(validator=instance_of(CatalogMetadata))
    api_version: str = field(default="xorq.dev/v1", validator=instance_of(str))
    kind: str = field(default="XorqCatalog", validator=instance_of(str))
    aliases: Mapping[str, Alias] = field(factory=dict)
    entries: Tuple[Entry, ...] = field(
        factory=tuple, validator=deep_iterable(instance_of(Entry), instance_of(tuple))
    )

    def with_entry(self, entry: Entry) -> "XorqCatalog":
        """Add or update an entry."""
        existing_entries = tuple(
            e for e in self.entries if e.entry_id != entry.entry_id
        )
        return self.clone(entries=existing_entries + (entry,))

    def with_alias(self, name: str, alias: Alias) -> "XorqCatalog":
        """Add or update an alias."""
        new_aliases = dict(self.aliases)
        new_aliases[name] = alias
        return self.clone(aliases=new_aliases)

    def maybe_get_entry(self, entry_id: str) -> Optional[Entry]:
        """Get entry by ID if it exists."""
        return next((e for e in self.entries if e.entry_id == entry_id), None)

    def maybe_get_alias(self, name: str) -> Optional[Alias]:
        """Get alias by name if it exists."""
        return self.aliases.get(name)

    def with_updated_metadata(self) -> "XorqCatalog":
        """Return catalog with updated timestamp."""
        return self.clone(metadata=self.metadata.with_updated_timestamp())

    def get_entry_ids(self) -> Tuple[str, ...]:
        """Get all entry IDs."""
        return tuple(e.entry_id for e in self.entries)

    def clone(self, **kwargs) -> "XorqCatalog":
        return type(self)(**asdict(self) | kwargs)

    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> "XorqCatalog":
        """Load catalog like the CLI does, or create a default if none exists.
        Supports the legacy raw dict format (entries+aliases) written by xorq CLI.
        """
        raw = load_catalog(path)
        entries_raw = raw.get("entries", []) or []
        raw_aliases = raw.get("aliases", {}) or {}
        # Clean alias data to only the fields known to Alias
        aliases_clean: dict[str, dict[str, Optional[str]]] = {
            name: {"entry_id": a.get("entry_id"), "revision_id": a.get("revision_id")}
            for name, a in raw_aliases.items()
        }
        if entries_raw or aliases_clean:
            # Build a full catalog dict and delegate to _dict_to_catalog
            metadata_dict = asdict(make_default_catalog().metadata)
            full = {
                "metadata": metadata_dict,
                "apiVersion": raw.get("apiVersion", "xorq.dev/v1"),
                "kind": raw.get("kind", "XorqCatalog"),
                "entries": entries_raw,
                "aliases": aliases_clean,
            }
            return _dict_to_catalog(full)
        # No catalog file (or empty), fall back to complete loader or default
        return load_catalog_or_default(path)

    def list_aliases(self, *, with_tags: bool = True) -> List[Mapping[str, Any]]:
        """
        List aliases with optional semantic tags, similar to 'xorq catalog ls --with-tags'.
        Returns a list of mappings with keys:
        'alias', 'revision', 'build_id', 'unbind_tag', 'tags' (Dict[str, Set[str]]),
        and 'breadcrumb' (formatted tag breadcrumb).
        """
        infos: List[Mapping[str, Any]] = []
        for name, alias in self.aliases.items():
            entry = self.maybe_get_entry(alias.entry_id)
            rev = (
                entry.maybe_get_revision(alias.revision_id)
                if entry and alias.revision_id
                else None
            )
            if rev is None or rev.build is None:
                continue
            build_id = rev.build.build_id
            # Determine unbind tag from metadata or default split tag
            unbind_tag = ""
            meta = getattr(entry, "metadata", {}) or getattr(entry, "meta", {})
            serve_meta = meta.get("serve", {})
            if (
                serve_meta.get("kind") == "UNBIND_VARIANT"
                and serve_meta.get("unbind", {}).get("type") == "split"
            ):
                unbind_tag = serve_meta.get("unbind", {}).get("tag", "")
            else:
                # fallback to first split tag
                tags0: Dict[str, Set[str]] = {}
                build_dir0 = maybe_resolve_build_dir(name, self)
                if build_dir0:
                    tags0 = collect_semantic_tags(build_dir0 / "expr.yaml")
                split_tags = sorted(tags0.get("split", []))
                unbind_tag = split_tags[0] if split_tags else ""

            # Collect tags and breadcrumb if requested
            tags_dict: Dict[str, Set[str]] = {}
            breadcrumb = ""
            if with_tags:
                build_dir = maybe_resolve_build_dir(name, self)
                if build_dir:
                    tags_dict = collect_semantic_tags(build_dir / "expr.yaml")
                    breadcrumb = format_tag_breadcrumb(tags_dict)

            infos.append(
                {
                    "alias": name,
                    "revision": alias.revision_id,
                    "build_id": build_id,
                    "unbind_tag": unbind_tag,
                    "tags": tags_dict,
                    "breadcrumb": breadcrumb,
                }
            )
        return infos

    def load_expr(self, target: str) -> Expr:
        """
        Load expression for given catalog target (alias, entry ID, or 'alias@revision').
        """
        build_dir = maybe_resolve_build_dir(target, self)
        if build_dir is None or not build_dir.is_dir():
            raise ValueError(f"Build directory not found for target: {target}")
        return _load_expr(build_dir)


@frozen
class Target:
    entry_id: str = field(validator=instance_of(str))
    rev: Optional[str] = field(default=None, validator=optional(instance_of(str)))
    alias: bool = field(default=False, validator=instance_of(bool))


@frozen
class ExportIndex:
    alias: Optional[str] = field(default=None, validator=optional(instance_of(str)))
    entry_id: Optional[str] = field(default=None, validator=optional(instance_of(str)))
    revision_id: Optional[str] = field(
        default=None, validator=optional(instance_of(str))
    )
    build_id: Optional[str] = field(default=None, validator=optional(instance_of(str)))
    expr_digest: Optional[str] = field(
        default=None, validator=optional(instance_of(str))
    )
    meta_digest: Optional[str] = field(
        default=None, validator=optional(instance_of(str))
    )


def get_catalog_path(path: Optional[Union[str, Path]] = None) -> Path:
    """Return the catalog file path, using XDG_CONFIG_HOME if set or default to ~/.config."""
    if path:
        return Path(path)
    # Determine config home directory at runtime
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        config_home = Path(xdg)
    else:
        config_home = Path.home() / ".config"
    return config_home / "xorq" / "catalog.yaml"


# Keep DEFAULT_CATALOG_PATH for backward compatibility
DEFAULT_CATALOG_PATH = get_catalog_path()


def do_save_catalog(
    catalog: XorqCatalog, path: Optional[Union[str, Path]] = None
) -> None:
    catalog_path = get_catalog_path(path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    catalog_dict = _catalog_to_dict(catalog.with_updated_metadata())

    with catalog_path.open("w") as f:
        yaml.safe_dump(catalog_dict, f, sort_keys=False)


def do_write_tree(root: Path, files: Mapping[str, bytes]) -> None:
    """Write file tree to disk (side effect)."""
    for rel_path, data in files.items():
        full_path = root / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(data)


def make_default_catalog() -> XorqCatalog:
    """Create a new default catalog."""
    now = datetime.now(timezone.utc).isoformat()
    metadata = CatalogMetadata(
        catalog_id=str(uuid.uuid4()),
        created_at=now,
        updated_at=now,
        tool_version=xo.__version__,
    )
    return XorqCatalog(metadata=metadata)


def maybe_load_catalog(
    path: Optional[Union[str, Path]] = None,
) -> Optional[XorqCatalog]:
    """Load catalog from disk, return None if not found."""
    catalog_path = get_catalog_path(path)
    if not catalog_path.exists():
        return None

    try:
        # Delegate to classmethod load to support both legacy raw and typed catalog formats
        return XorqCatalog.load(path)
    except Exception:
        return None


def load_catalog_or_default(path: Optional[Union[str, Path]] = None) -> XorqCatalog:
    """Load catalog or create default if not found."""
    catalog = maybe_load_catalog(path)
    return catalog if catalog is not None else make_default_catalog()


def maybe_resolve_target(target: str, catalog: XorqCatalog) -> Optional[Target]:
    """Parse and resolve a target string, return None if invalid."""
    if "@" in target:
        base, rev = target.split("@", 1)
    else:
        base, rev = target, None

    # Try alias first
    alias = catalog.maybe_get_alias(base)
    if alias is not None:
        return Target(entry_id=alias.entry_id, rev=rev, alias=rev is None)

    # Try entry ID
    if base in catalog.get_entry_ids():
        return Target(entry_id=base, rev=rev, alias=False)

    return None


def collect_semantic_tags(expr_path: Union[str, Path]) -> Dict[str, Set[str]]:
    """Collect semantic tags from a built expression using graph utilities."""
    import xorq.expr.relations as rel
    from xorq.common.utils.graph_utils import walk_nodes
    from xorq.ibis_yaml.compiler import load_expr

    p = Path(expr_path)
    build_dir = p if p.is_dir() else p.parent
    expr = load_expr(build_dir)

    tags: Dict[str, Set[str]] = {}

    for tn in walk_nodes(rel.Tag, expr):
        ttype = tn.metadata.get("type")
        tagname = tn.tag
        step = tn.metadata.get("step_name")

        if ttype == rel.TagType.SOURCE and tagname:
            tags.setdefault(rel.TagType.SOURCE.value, set()).add(tagname)
        elif ttype == rel.TagType.CACHE and tagname:
            tags.setdefault(rel.TagType.CACHE.value, set()).add(tagname)
        elif ttype == rel.TagType.TRANSFORM and step:
            tags.setdefault(rel.TagType.TRANSFORM.value, set()).add(step)
        elif ttype == rel.TagType.PREDICT and step:
            tags.setdefault(rel.TagType.PREDICT.value, set()).add(step)
        elif ttype == rel.TagType.SPLIT and tagname:
            tags.setdefault(rel.TagType.SPLIT.value, set()).add(tagname)
        elif ttype == rel.TagType.UDF and tagname:
            tags.setdefault(rel.TagType.UDF.value, set()).add(tagname)
        elif tagname:
            tags.setdefault("generic", set()).add(tagname)

    return tags


def print_tag_tree(tags: Dict[str, Set[str]], title: str) -> None:
    """Print a tree view of semantic tags under the given title."""

    # FIXME: Get enum values in order, then add any others alphabetically
    cats = tags

    print(f"Tags for {title}")
    for i, cat in enumerate(cats):
        is_last_cat = i == len(cats) - 1
        prefix = "└──" if is_last_cat else "├──"
        print(f"{prefix} {cat}")
        items = sorted(tags[cat])
        for j, item in enumerate(items):
            is_last_item = j == len(items) - 1
            indent = "    " if is_last_cat else "│   "
            sub = "└──" if is_last_item else "├──"
            print(f"{indent}{sub} {item}")


def format_tag_breadcrumb(tags: Dict[str, Set[str]]) -> str:
    """Format semantic tags as a one-line breadcrumb string."""
    parts: List[str] = []
    for cat in tags:  # Use natural dict order (insertion order in Python 3.7+)
        for item in sorted(tags[cat]):
            parts.append(f"{cat}/{item}")
    return " → ".join(parts)


def maybe_resolve_build_dir(token: str, catalog: XorqCatalog) -> Optional[Path]:
    """Resolve build directory from token, return None if not found."""
    # Try an explicit directory literal
    path = Path(token)
    if path.exists() and path.is_dir():
        return path

    # Next, try a known build ID lookup
    build_path = _maybe_find_build_by_id(token, catalog)
    if build_path is not None:
        # If stored as relative, interpret under the CLI config dir
        if not build_path.is_absolute():
            build_path = get_catalog_path().parent / build_path
        return build_path

    # Try alias/entry resolution
    target = maybe_resolve_target(token, catalog)
    if target is None:
        return None

    entry = catalog.maybe_get_entry(target.entry_id)
    if entry is None:
        return None

    # Determine which revision to use
    revision_id = target.rev or entry.current_revision
    if revision_id is None:
        return None

    revision = entry.maybe_get_revision(revision_id)
    if revision is None or revision.build is None:
        return None

    build_path = revision.build.path
    if not build_path:
        return None
    pth = Path(build_path)
    if not pth.is_absolute():
        pth = get_catalog_path().parent / pth
    return pth


def dump_yaml(obj: Any) -> str:
    """Dump object to YAML string."""
    return yaml.safe_dump(obj, sort_keys=False)


def dump_json(obj: Any) -> str:
    """Dump object to JSON string."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def unified_dir_diff(left_root: Path, right_root: Path) -> Tuple[bool, str]:
    """Compare two directories and return unified diff."""
    left_files = _get_file_map(left_root)
    right_files = _get_file_map(right_root)
    all_paths = sorted(set(left_files) | set(right_files))

    chunks = []
    different = False

    for rel_path in all_paths:
        left_path = left_files.get(rel_path)
        right_path = right_files.get(rel_path)

        diff_result = _compare_files(left_path, right_path, rel_path)
        if diff_result is not None:
            different = True
            chunks.extend(diff_result)

    return different, "".join(chunks)


def _catalog_to_dict(catalog: XorqCatalog) -> Dict[str, Any]:
    """Convert catalog to dictionary for serialization."""
    return {
        "apiVersion": catalog.api_version,
        "kind": catalog.kind,
        "metadata": asdict(catalog.metadata),
        "aliases": {name: asdict(alias) for name, alias in catalog.aliases.items()},
        "entries": [_entry_to_dict(entry) for entry in catalog.entries],
    }


def _entry_to_dict(entry: Entry) -> Dict[str, Any]:
    """Convert entry to dictionary."""
    return {
        "entry_id": entry.entry_id,
        "current_revision": entry.current_revision,
        "history": [_revision_to_dict(rev) for rev in entry.history],
    }


def _revision_to_dict(revision: Revision) -> Dict[str, Any]:
    """Convert revision to dictionary."""
    result = {
        "revision_id": revision.revision_id,
        "created_at": revision.created_at,
    }

    if revision.build:
        result["build"] = asdict(revision.build)
    if revision.expr_hashes:
        result["expr_hashes"] = revision.expr_hashes
    if revision.node_hashes:
        result["node_hashes"] = list(revision.node_hashes)
    if revision.meta_digest:
        result["meta_digest"] = revision.meta_digest
    if revision.metadata:
        result["metadata"] = revision.metadata
    if revision.schema_fingerprint:
        result["schema_fingerprint"] = revision.schema_fingerprint

    return result


def _dict_to_catalog(data: Dict[str, Any]) -> XorqCatalog:
    """Convert dictionary to catalog."""
    metadata = CatalogMetadata(**data["metadata"])

    # Only keep fields known to Alias (entry_id, revision_id); ignore extras like updated_at
    aliases: dict[str, Alias] = {}
    for name, alias_data in data.get("aliases", {}).items():
        aliases[name] = Alias(
            entry_id=alias_data.get("entry_id"),
            revision_id=alias_data.get("revision_id"),
        )

    entries = tuple(
        _dict_to_entry(entry_data) for entry_data in data.get("entries", []) or []
    )

    return XorqCatalog(
        api_version=data.get("apiVersion", "xorq.dev/v1"),
        kind=data.get("kind", "XorqCatalog"),
        metadata=metadata,
        aliases=aliases,
        entries=entries,
    )


def _dict_to_entry(data: Dict[str, Any]) -> Entry:
    """Convert dictionary to entry."""
    history = tuple(_dict_to_revision(rev_data) for rev_data in data.get("history", []))

    return Entry(
        entry_id=data["entry_id"],
        current_revision=data.get("current_revision"),
        history=history,
    )


def _dict_to_revision(data: Dict[str, Any]) -> Revision:
    """Convert dictionary to revision."""
    build = None
    if "build" in data:
        build = Build(**data["build"])

    return Revision(
        revision_id=data["revision_id"],
        created_at=data["created_at"],
        build=build,
        expr_hashes=data.get("expr_hashes"),
        node_hashes=tuple(data.get("node_hashes", [])),
        meta_digest=data.get("meta_digest"),
        metadata=data.get("metadata"),
        schema_fingerprint=data.get("schema_fingerprint"),
    )


def _maybe_find_build_by_id(build_id: str, catalog: XorqCatalog) -> Optional[Path]:
    """Find build directory by build ID."""
    for entry in catalog.entries:
        for revision in entry.history:
            if revision.build and revision.build.build_id == build_id:
                path = revision.build.path
                return Path(path) if path else None
    return None


def _maybe_get_schema_fingerprint(revision: Revision) -> Optional[str]:
    """Extract schema fingerprint from revision."""
    if revision.schema_fingerprint:
        return revision.schema_fingerprint
    if revision.metadata:
        return revision.metadata.get("schema_fingerprint")
    return None


def _maybe_get_read_set(revision: Revision) -> Optional[List[Any]]:
    """Extract read set from revision metadata."""
    if revision.metadata:
        return revision.metadata.get("read_set")
    return None


def _get_file_map(root: Path) -> Dict[str, Path]:
    """Get mapping of relative paths to absolute paths."""
    files = [p for p in root.rglob("*") if p.is_file()]
    return {str(p.relative_to(root)): p for p in files}


def _read_text_lines(path: Path) -> List[str]:
    """Read file as text lines."""
    return path.read_text().splitlines(keepends=True)


def _compare_files(
    left_path: Optional[Path], right_path: Optional[Path], rel_path: str
) -> Optional[List[str]]:
    """Compare two files and return diff lines if different."""
    if left_path and right_path:
        if left_path.read_bytes() == right_path.read_bytes():
            return None

        diff = difflib.unified_diff(
            _read_text_lines(left_path),
            _read_text_lines(right_path),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            lineterm="",
        )
        return list(diff) + ["\n"]
    elif left_path and not right_path:
        return [f"--- a/{rel_path}\n+++ /dev/null\n"]
    else:
        return [f"--- /dev/null\n+++ b/{rel_path}\n"]


@curry
def with_entry(entry: Entry, catalog: XorqCatalog) -> XorqCatalog:
    """Curried version of with_entry for composition."""
    return catalog.with_entry(entry)


@curry
def with_alias(name: str, alias: Alias, catalog: XorqCatalog) -> XorqCatalog:
    """Curried version of with_alias for composition."""
    return catalog.with_alias(name, alias)


def write_tree(root: Path, files: Mapping[str, bytes]) -> None:
    """Write file tree to disk."""
    do_write_tree(root, files)


def load_catalog(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load catalog as raw dict; return minimal structure if not found."""
    catalog_path = get_catalog_path(path)
    if not catalog_path.exists():
        return {"entries": [], "aliases": {}}
    with catalog_path.open() as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("entries", [])
    data.setdefault("aliases", {})
    return data


def save_catalog(
    catalog: Dict[str, Any], path: Optional[Union[str, Path]] = None
) -> None:
    """Save raw catalog dict to disk."""
    catalog_path = get_catalog_path(path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with catalog_path.open("w") as f:
        yaml.safe_dump(catalog, f, sort_keys=False)


def resolve_target(target: str, catalog: Dict[str, Any]) -> Optional[Target]:
    """Resolve target string against raw catalog dict."""
    if "@" in target:
        base, rev = target.split("@", 1)
    else:
        base, rev = target, None
    # Special case: generic 'entry' resolves to the single entry if only one exists
    entries = catalog.get("entries", [])
    if base == "entry" and entries:
        entry_data = entries[0]
        rev_id = rev if rev is not None else entry_data.get("current_revision")
        return Target(entry_id=entry_data.get("entry_id"), rev=rev_id, alias=False)
    alias_map = catalog.get("aliases", {})
    if base in alias_map:
        alias_data = alias_map[base]
        entry_id = alias_data.get("entry_id")
        rev_id = rev if rev is not None else alias_data.get("revision_id")
        return Target(entry_id=entry_id, rev=rev_id, alias=True)
    entry_ids = [e.get("entry_id") for e in entries]
    if base in entry_ids:
        if rev is None:
            entry_data = next(e for e in entries if e.get("entry_id") == base)
            rev_id = entry_data.get("current_revision")
        else:
            rev_id = rev
        return Target(entry_id=base, rev=rev_id, alias=False)
    return None


def resolve_build_dir(token: str, catalog: Dict[str, Any]) -> Optional[Path]:
    """Resolve build directory from raw catalog dict."""
    path = Path(token)
    if path.exists() and path.is_dir():
        return path
    # Match on explicit build_id entries
    for entry in catalog.get("entries", []):
        for rev in entry.get("history", []):
            build = rev.get("build")
            if build and build.get("build_id") == token:
                p = build.get("path")
                if p:
                    pth = Path(p)
                    # If stored path is relative, interpret relative to catalog config directory
                    if not pth.is_absolute():
                        cfg_dir = get_catalog_path().parent
                        pth = cfg_dir / pth
                    return pth
    t = resolve_target(token, catalog)
    if t is None:
        return None
    # Match on entry@revision targets
    for entry in catalog.get("entries", []):
        if entry.get("entry_id") == t.entry_id:
            for rev in entry.get("history", []):
                if rev.get("revision_id") == t.rev:
                    build = rev.get("build")
                    if build and build.get("path"):
                        p = build.get("path")
                        pth = Path(p)
                        if not pth.is_absolute():
                            cfg_dir = get_catalog_path().parent
                            pth = cfg_dir / pth
                        return pth
    return None
