import os
import uuid
import yaml
import hashlib
import json
import difflib
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Union, Mapping, Sequence
from functools import singledispatch, partial
from toolz import curry, pipe
from attrs import frozen, field, asdict
from attrs.validators import instance_of, optional, deep_iterable

import xorq as xo

# =============================================================================
# Immutable Data Structures
# =============================================================================

@frozen
class CatalogMetadata:
    """Catalog metadata."""
    catalog_id: str = field(validator=instance_of(str))
    created_at: str = field(validator=instance_of(str))
    updated_at: str = field(validator=instance_of(str))
    tool_version: str = field(validator=instance_of(str))

    def with_updated_timestamp(self) -> 'CatalogMetadata':
        """Return new metadata with updated timestamp."""
        return self.clone(updated_at=datetime.now(timezone.utc).isoformat())

    def clone(self, **kwargs) -> 'CatalogMetadata':
        """Create a copy with specified changes."""
        return type(self)(**asdict(self) | kwargs)

@frozen
class Build:
    """ Build information."""
    build_id: Optional[str] = field(default=None, validator=optional(instance_of(str)))
    path: Optional[str] = field(default=None, validator=optional(instance_of(str)))

    def clone(self, **kwargs) -> 'Build':
        return type(self)(**asdict(self) | kwargs)

@frozen
class Revision:
    """ Revision data."""
    revision_id: str = field(validator=instance_of(str))
    created_at: str = field(validator=instance_of(str))
    build: Optional[Build] = field(default=None, validator=optional(instance_of(Build)))
    expr_hashes: Optional[Dict[str, str]] = field(default=None)
    node_hashes: Tuple[str, ...] = field(factory=tuple, validator=deep_iterable(instance_of(str), instance_of(tuple)))
    meta_digest: Optional[str] = field(default=None, validator=optional(instance_of(str)))
    metadata: Optional[Dict[str, Any]] = field(default=None)
    schema_fingerprint: Optional[str] = field(default=None, validator=optional(instance_of(str)))

    def clone(self, **kwargs) -> 'Revision':
        return type(self)(**asdict(self) | kwargs)

@frozen
class Entry:
    """Entry in the catalog."""
    entry_id: str = field(validator=instance_of(str))
    current_revision: Optional[str] = field(default=None, validator=optional(instance_of(str)))
    history: Tuple[Revision, ...] = field(factory=tuple, validator=deep_iterable(instance_of(Revision), instance_of(tuple)))

    def with_revision(self, revision: Revision) -> 'Entry':
        """Add a new revision to history."""
        return self.clone(
            current_revision=revision.revision_id,
            history=self.history + (revision,)
        )

    def maybe_get_revision(self, revision_id: str) -> Optional[Revision]:
        """Get revision by ID if it exists."""
        return next((r for r in self.history if r.revision_id == revision_id), None)

    def maybe_current_revision(self) -> Optional[Revision]:
        """Get current revision if it exists."""
        if self.current_revision is None:
            return None
        return self.maybe_get_revision(self.current_revision)

    def clone(self, **kwargs) -> 'Entry':
        return type(self)(**asdict(self) | kwargs)

@frozen
class Alias:
    entry_id: str = field(validator=instance_of(str))
    revision_id: Optional[str] = field(default=None, validator=optional(instance_of(str)))

    def clone(self, **kwargs) -> 'Alias':
        return type(self)(**asdict(self) | kwargs)

@frozen
class XorqCatalog:
    """ Xorq Catalog container."""
    metadata: CatalogMetadata = field(validator=instance_of(CatalogMetadata))
    api_version: str = field(default="xorq.dev/v1", validator=instance_of(str))
    kind: str = field(default="XorqCatalog", validator=instance_of(str))
    aliases: Mapping[str, Alias] = field(factory=dict)
    entries: Tuple[Entry, ...] = field(factory=tuple, validator=deep_iterable(instance_of(Entry), instance_of(tuple)))

    def with_entry(self, entry: Entry) -> 'XorqCatalog':
        """Add or update an entry."""
        existing_entries = tuple(e for e in self.entries if e.entry_id != entry.entry_id)
        return self.clone(entries=existing_entries + (entry,))

    def with_alias(self, name: str, alias: Alias) -> 'XorqCatalog':
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

    def with_updated_metadata(self) -> 'XorqCatalog':
        """Return catalog with updated timestamp."""
        return self.clone(metadata=self.metadata.with_updated_timestamp())

    def get_entry_ids(self) -> Tuple[str, ...]:
        """Get all entry IDs."""
        return tuple(e.entry_id for e in self.entries)

    def clone(self, **kwargs) -> 'XorqCatalog':
        return type(self)(**asdict(self) | kwargs)


@frozen
class Target:
    entry_id: str = field(validator=instance_of(str))
    rev: Optional[str] = field(default=None, validator=optional(instance_of(str)))
    alias: bool = field(default=False, validator=instance_of(bool))


@frozen
class ExportIndex:
    alias: Optional[str] = field(default=None, validator=optional(instance_of(str)))
    entry_id: Optional[str] = field(default=None, validator=optional(instance_of(str)))
    revision_id: Optional[str] = field(default=None, validator=optional(instance_of(str)))
    build_id: Optional[str] = field(default=None, validator=optional(instance_of(str)))
    expr_digest: Optional[str] = field(default=None, validator=optional(instance_of(str)))
    meta_digest: Optional[str] = field(default=None, validator=optional(instance_of(str)))


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

def do_save_catalog(catalog: XorqCatalog, path: Optional[Union[str, Path]] = None) -> None:
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
        tool_version=xo.__version__
    )
    return XorqCatalog(metadata=metadata)

def maybe_load_catalog(path: Optional[Union[str, Path]] = None) -> Optional[XorqCatalog]:
    """Load catalog from disk, return None if not found."""
    catalog_path = get_catalog_path(path)
    if not catalog_path.exists():
        return None

    try:
        with catalog_path.open() as f:
            data = yaml.safe_load(f)
            return _dict_to_catalog(data) if data else None
    except (yaml.YAMLError, KeyError, TypeError):
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
        return Target(
            entry_id=alias.entry_id,
            rev=rev,
            alias=rev is None
        )

    # Try entry ID
    if base in catalog.get_entry_ids():
        return Target(
            entry_id=base,
            rev=rev,
            alias=False
        )

    return None

def maybe_resolve_build_dir(token: str, catalog: XorqCatalog) -> Optional[Path]:
    """Resolve build directory from token, return None if not found."""
    # Try existing directory first
    path = Path(token)
    if path.exists() and path.is_dir():
        return path

    # Try build ID lookup
    build_path = _maybe_find_build_by_id(token, catalog)
    if build_path is not None:
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
    return Path(build_path) if build_path else None


def dump_yaml(obj: Any) -> str:
    """Dump object to YAML string."""
    return yaml.safe_dump(obj, sort_keys=False)

def dump_json(obj: Any) -> str:
    """Dump object to JSON string."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def build_virtual_export_tree(entry: Entry, revision: Revision) -> Mapping[str, bytes]:
    """Build virtual export file tree."""
    # Extract values safely
    build = revision.build
    expr_hashes = revision.expr_hashes or {}

    index = ExportIndex(
        alias=None,
        entry_id=entry.entry_id,
        revision_id=revision.revision_id,
        build_id=build.build_id if build else None,
        expr_digest=expr_hashes.get("expr"),
        meta_digest=revision.meta_digest
    )

    files = {}
    files["EXPORT_INDEX.yaml"] = dump_yaml(normalize_obj(asdict(index))).encode()

    # Expression files
    if index.expr_digest:
        files["expr/expr.yaml"] = dump_yaml({"root_expr": index.expr_digest}).encode()

        for node_hash in revision.node_hashes:
            files[f"expr/nodes/{node_hash}.yaml"] = dump_yaml({"id": node_hash}).encode()

    # Schema fingerprint
    schema_fp = _maybe_get_schema_fingerprint(revision)
    if schema_fp:
        files["schema/output.schema.json"] = dump_json({"fingerprint": schema_fp}).encode()

    # Read set
    read_set = _maybe_get_read_set(revision)
    if read_set:
        files["reads/sources.yaml"] = dump_yaml(normalize_obj(read_set)).encode()

    # Metadata
    files["meta/volatile.yaml"] = dump_yaml({"created_at": revision.created_at}).encode()

    return files

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
        "entries": [_entry_to_dict(entry) for entry in catalog.entries]
    }

def _entry_to_dict(entry: Entry) -> Dict[str, Any]:
    """Convert entry to dictionary."""
    return {
        "entry_id": entry.entry_id,
        "current_revision": entry.current_revision,
        "history": [_revision_to_dict(rev) for rev in entry.history]
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

    aliases = {}
    for name, alias_data in data.get("aliases", {}).items():
        aliases[name] = Alias(**alias_data)

    entries = tuple(_dict_to_entry(entry_data) for entry_data in data.get("entries", []))

    return XorqCatalog(
        api_version=data.get("apiVersion", "xorq.dev/v1"),
        kind=data.get("kind", "XorqCatalog"),
        metadata=metadata,
        aliases=aliases,
        entries=entries
    )

def _dict_to_entry(data: Dict[str, Any]) -> Entry:
    """Convert dictionary to entry."""
    history = tuple(_dict_to_revision(rev_data) for rev_data in data.get("history", []))

    return Entry(
        entry_id=data["entry_id"],
        current_revision=data.get("current_revision"),
        history=history
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
        schema_fingerprint=data.get("schema_fingerprint")
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

def _compare_files(left_path: Optional[Path], right_path: Optional[Path], rel_path: str) -> Optional[List[str]]:
    """Compare two files and return diff lines if different."""
    if left_path and right_path:
        if left_path.read_bytes() == right_path.read_bytes():
            return None

        diff = difflib.unified_diff(
            _read_text_lines(left_path),
            _read_text_lines(right_path),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            lineterm=""
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

load_catalog = load_catalog_or_default
resolve_target = maybe_resolve_target  # For backward compatibility - consider deprecating
resolve_build_dir = maybe_resolve_build_dir  # For backward compatibility - consider deprecating

def save_catalog(catalog: XorqCatalog, path: Optional[Union[str, Path]] = None) -> XorqCatalog:
    """Save catalog and return the updated version."""
    updated_catalog = catalog.with_updated_metadata()
    do_save_catalog(updated_catalog, path)
    return updated_catalog

def write_tree(root: Path, files: Mapping[str, bytes]) -> None:
    """Alias for do_write_tree for backward compatibility."""
    do_write_tree(root, files)
    
# Legacy dict-based (mapping) API for catalog operations
def _dict_load_catalog(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load catalog as raw dict; return minimal structure if not found."""
    catalog_path = get_catalog_path(path)
    if not catalog_path.exists():
        return {"entries": [], "aliases": {}}
    with catalog_path.open() as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("entries", [])
    data.setdefault("aliases", {})
    return data

def _dict_save_catalog(catalog: Dict[str, Any], path: Optional[Union[str, Path]] = None) -> None:
    """Save raw catalog dict to disk."""
    catalog_path = get_catalog_path(path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with catalog_path.open("w") as f:
        yaml.safe_dump(catalog, f, sort_keys=False)

def _dict_resolve_target(target: str, catalog: Dict[str, Any]) -> Optional[Target]:
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
        # Use explicit revision if provided, else alias's current revision_id
        if rev is not None:
            rev_id = rev
        else:
            rev_id = alias_data.get("revision_id")
        return Target(entry_id=entry_id, rev=rev_id, alias=True)
    entries = catalog.get("entries", [])
    entry_ids = [e.get("entry_id") for e in entries]
    if base in entry_ids:
        if rev is None:
            entry_data = next(e for e in entries if e.get("entry_id") == base)
            rev_id = entry_data.get("current_revision")
        else:
            rev_id = rev
        return Target(entry_id=base, rev=rev_id, alias=False)
    return None

def _dict_resolve_build_dir(token: str, catalog: Dict[str, Any]) -> Optional[Path]:
    """Resolve build directory from raw catalog dict."""
    path = Path(token)
    if path.exists() and path.is_dir():
        return path
    for entry in catalog.get("entries", []):
        for rev in entry.get("history", []):
            build = rev.get("build")
            if build and build.get("build_id") == token:
                p = build.get("path")
                if p:
                    return Path(p)
    t = resolve_target(token, catalog)
    if t is None:
        return None
    for entry in catalog.get("entries", []):
        if entry.get("entry_id") == t.entry_id:
            for rev in entry.get("history", []):
                if rev.get("revision_id") == t.rev:
                    build = rev.get("build")
                    if build and build.get("path"):
                        return Path(build.get("path"))
    return None

# Override public API for backward compatibility
load_catalog = _dict_load_catalog
save_catalog = _dict_save_catalog
resolve_target = _dict_resolve_target
resolve_build_dir = _dict_resolve_build_dir
