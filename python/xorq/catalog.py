import difflib
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import yaml
from attrs import evolve, field, frozen
from attrs.validators import deep_iterable, instance_of, optional

import xorq as xo


@frozen
class CatalogMetadata:
    """Catalog metadata."""

    # FIXME: make uuid
    catalog_id: str = field(validator=instance_of(str))
    # FIXME: make datetime.datetime
    created_at: str = field(validator=instance_of(str))
    updated_at: str = field(validator=instance_of(str))
    tool_version: str = field(validator=instance_of(str))

    def with_updated_timestamp(self) -> "CatalogMetadata":
        """Return new metadata with updated timestamp."""
        return self.evolve(updated_at=datetime.now(timezone.utc).isoformat())

    def evolve(self, **kwargs) -> "CatalogMetadata":
        """Create a copy with specified changes."""
        return evolve(self, **kwargs)


@frozen
class Build:
    """Build information."""

    build_id: Optional[str] = field(default=None, validator=optional(instance_of(str)))
    # FIXME: make Path
    path: Optional[str] = field(default=None, validator=optional(instance_of(str)))

    def evolve(self, **kwargs) -> "Build":
        """Create a copy with specified changes."""
        return evolve(self, **kwargs)


@frozen
class Revision:
    """Revision data."""

    # FIXME: make int
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

    def evolve(self, **kwargs) -> "Revision":
        """Create a copy with specified changes."""
        return evolve(self, **kwargs)


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
        return self.evolve(
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

    def evolve(self, **kwargs) -> "Entry":
        """Create a copy with specified changes."""
        return evolve(self, **kwargs)


@frozen
class Alias:
    entry_id: str = field(validator=instance_of(str))
    revision_id: Optional[str] = field(
        default=None, validator=optional(instance_of(str))
    )

    def evolve(self, **kwargs) -> "Alias":
        """Create a copy with specified changes."""
        return evolve(self, **kwargs)


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
        return self.evolve(entries=existing_entries + (entry,))

    def with_alias(self, name: str, alias: Alias) -> "XorqCatalog":
        """Add or update an alias."""
        new_aliases = dict(self.aliases)
        new_aliases[name] = alias
        return self.evolve(aliases=new_aliases)

    def maybe_get_entry(self, entry_id: str) -> Optional[Entry]:
        """Get entry by ID if it exists."""
        return next((e for e in self.entries if e.entry_id == entry_id), None)

    def maybe_get_alias(self, name: str) -> Optional[Alias]:
        """Get alias by name if it exists."""
        return self.aliases.get(name)

    def with_updated_metadata(self) -> "XorqCatalog":
        """Return catalog with updated timestamp."""
        return self.evolve(metadata=self.metadata.with_updated_timestamp())

    def get_entry_ids(self) -> Tuple[str, ...]:
        """Get all entry IDs."""
        return tuple(e.entry_id for e in self.entries)

    def evolve(self, **kwargs) -> "XorqCatalog":
        """Create a copy with specified changes."""
        return evolve(self, **kwargs)


@frozen
class Target:
    entry_id: str = field(validator=instance_of(str))
    rev: Optional[str] = field(default=None, validator=optional(instance_of(str)))
    alias: bool = field(default=False, validator=instance_of(bool))


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
