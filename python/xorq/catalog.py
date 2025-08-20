import difflib
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import yaml
from attrs import evolve, field, frozen
from attrs.validators import deep_iterable, instance_of, optional

import xorq as xo
from xorq.ibis_yaml.compiler import (
    BuildManager,
    load_expr,
)


# Keep DEFAULT_CATALOG_PATH for backward compatibility
DEFAULT_CATALOG_PATH = Path(
    os.environ.get("XDG_CONFIG_HOME") or Path.home().joinpath(".config")
).joinpath("xorq", "catalog.yaml")


@frozen
class CatalogMetadata:
    """Catalog metadata."""

    # FIXME: make uuid
    catalog_id: str = field(validator=instance_of(str), factory=uuid.uuid4)
    # FIXME: make datetime.datetime
    created_at: str = field(validator=instance_of(datetime), factory=datetime.now)
    updated_at: str = field(validator=instance_of(datetime), factory=datetime.now)
    tool_version: str = field(validator=instance_of(str), default=xo.__version__)

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
    created_at: str = field(validator=instance_of(datetime), factory=datetime.now)
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
        validator=deep_iterable(instance_of(Revision), instance_of(tuple)),
        factory=tuple,
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
        validator=deep_iterable(instance_of(Entry), instance_of(tuple)),
        factory=tuple,
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
    return Path(path) if path else DEFAULT_CATALOG_PATH


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


def resolve_target(
    target: str, catalog: Optional[Dict[str, Any]] = None
) -> Optional[Target]:
    """Resolve target string against raw catalog dict."""
    catalog = catalog or load_catalog()
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

    def get_next_absolute_path(paths):
        path = next(filter(None, paths), None)
        if path:
            path = Path(path)
            # If stored path is relative, interpret relative to catalog config directory
            if not path.is_absolute():
                path = DEFAULT_CATALOG_PATH.parent.joinpath(path)
            return path
        else:
            return None

    def get_explicit_build_entry(token, catalog):
        # Match on explicit build_id entries
        builds = (
            rev.get("build", {})
            for entry in catalog.get("entries", [])
            for rev in entry.get("history", [])
        )
        paths = (
            build.get("path") for build in builds if build.get("build_id") == token
        )
        return get_next_absolute_path(paths)

    def get_entry_revision_target(t, catalog):
        gen = (
            rev.get("build", {}).get("path")
            for entry in catalog.get("entries", [])
            for rev in entry.get("history", [])
            if (entry.get("entry_id") == t.entry_id and rev.get("revision_id") == t.rev)
        )
        return get_next_absolute_path(gen)

    path = Path(token)
    if path.exists() and path.is_dir():
        return path
    path = get_explicit_build_entry(token, catalog)
    if path:
        return path

    # Match on entry@revision targets
    t = resolve_target(token, catalog)
    if t is None:
        return None
    path = get_entry_revision_target(t, catalog)
    if path:
        return path
    return None


def unified_dir_diff(left_root: Path, right_root: Path) -> Tuple[bool, str]:
    """Compare two directories and return unified diff."""
    (left_files, right_files) = (
        tuple(p.relative_to(root) for p in root.rglob("*") if p.is_file())
        for root in (left_root, right_root)
    )
    rel_paths = sorted(set(left_files) | set(right_files))
    gen = (
        (left_root.joinpath(rel_path), right_root.joinpath(rel_path), rel_path)
        for rel_path in rel_paths
    )
    gen = (
        _compare_files(
            left_path if left_path.exists() else None,
            right_path if right_path.exists() else None,
            rel_path,
        )
        for (left_path, right_path, rel_path) in gen
    )
    chunks = tuple(filter(None, gen))
    changed = bool(chunks)
    joined = "".join(sum(chunks, start=[]))
    return (changed, joined)


def _compare_files(
    left_path: Optional[Path], right_path: Optional[Path], rel_path: str
) -> Optional[List[str]]:
    """Compare two files and return diff lines if different."""
    (fromfile, tofile) = (f"{which}/{rel_path}" for which in ("a", "b"))
    match [left_path, right_path]:
        case [None, None]:
            raise ValueError
        case [Path(), None]:
            return [f"--- {fromfile}\n+++ /dev/null\n"]
        case [None, Path()]:
            return [f"--- /dev/null\n+++ {tofile}\n"]
        case [Path(), Path()]:
            (left, right) = (
                el.read_text().splitlines(keepends=True) if el else None
                for el in (left_path, right_path)
            )
            if left == right:
                return None
            else:
                diff = difflib.unified_diff(
                    left,
                    right,
                    fromfile=fromfile,
                    tofile=tofile,
                    lineterm="",
                )
                return list(diff) + ["\n"]
        case _:
            raise ValueError


def maybe_resolve_build_dirs(
    left: str, right: str, catalog
) -> tuple[Path, Path] | None:
    try:
        left_dir = resolve_build_dir(left, catalog)
        right_dir = resolve_build_dir(right, catalog)
        # Interpret relative build paths from catalog as relative to config directory
        config_path = get_catalog_path()
        config_dir = config_path.parent
        # Adjust left_dir if token is not a literal directory and left_dir is relative
        token_path = Path(left)
        if (
            not token_path.is_dir()
            and left_dir is not None
            and not left_dir.is_absolute()
        ):
            left_dir = config_dir / left_dir
        # Adjust right_dir similarly
        token_path = Path(right)
        if (
            not token_path.is_dir()
            and right_dir is not None
            and not right_dir.is_absolute()
        ):
            right_dir = config_dir / right_dir
    except ValueError as e:
        print(f"Error: {e}")
        return None
    # Ensure resolution succeeded
    if left_dir is None:
        print(f"Build target not found: {left}")
        return None
    if right_dir is None:
        print(f"Build target not found: {right}")
        return None
    # Ensure paths exist and are directories
    if not left_dir.exists() or not left_dir.is_dir():
        print(f"Build directory not found: {left_dir}")
        return None
    if not right_dir.exists() or not right_dir.is_dir():
        print(f"Build directory not found: {right_dir}")
        return None
    return left_dir, right_dir


def do_diff_builds(
    left: str,
    right: str,
    files: list[str] | None,
    all_flag: bool,
) -> int:
    def get_keep_files(
        file_list: tuple[str, ...], left_dir: Path, right_dir: Path
    ) -> tuple[str, ...]:
        return tuple(
            f for f in file_list if (left_dir / f).exists() or (right_dir / f).exists()
        )

    def run_diffs(left_dir: Path, right_dir: Path, keep_files: tuple[str, ...]) -> int:
        exit_code = 0
        for f in keep_files:
            print(f"## Diff: {f}")
            lf = left_dir / f
            rf = right_dir / f
            ret = subprocess.call(["git", "diff", "--no-index", "--", str(lf), str(rf)])
            if ret == 1:
                exit_code = 1
            elif ret != 0:
                return ret
        return exit_code

    catalog = load_catalog()
    resolved = maybe_resolve_build_dirs(left, right, catalog)
    if not resolved:
        return 2
    left_dir, right_dir = resolved
    file_list = get_diff_file_list(left_dir, right_dir, files, all_flag)
    keep_files = get_keep_files(file_list, left_dir, right_dir)
    if not keep_files:
        print("No files to diff")
        return 2
    return run_diffs(left_dir, right_dir, keep_files)


def lineage_command(
    target: str,
):
    """
    Print per-column lineage trees for a single build.
    """
    catalog = load_catalog()
    build_dir = resolve_build_dir(target, catalog)
    if build_dir is None or not build_dir.exists() or not build_dir.is_dir():
        print(f"Build target not found: {target}")
        sys.exit(2)
    # Load serialized expression
    expr = load_expr(build_dir)
    # Build and print lineage trees
    from xorq.common.utils.lineage_utils import build_column_trees, print_tree

    trees = build_column_trees(expr)
    for column, tree in trees.items():
        print(f"Lineage for column '{column}':")
        print_tree(tree)
        print()


def catalog_command(args):
    """
    Manage build catalog subcommands: add, ls, inspect.
    """
    # Determine canonical catalog path in config directory
    config_path = get_catalog_path()
    config_dir = config_path.parent
    if args.subcommand == "add":
        # Use absolute path for build directory
        build_path = Path(args.build_path).resolve()
        alias = args.alias
        # Validate build and extract metadata (expr hash recalculated fresh later)
        build_id, meta_digest, metadata_preview = BuildManager.validate_build(
            build_path
        )
        # Ensure local catalog directory and builds subdirectory exist
        config_dir.mkdir(parents=True, exist_ok=True)
        # Store all builds under a canonical folder
        builds_dir = config_dir / "catalog-builds"
        builds_dir.mkdir(parents=True, exist_ok=True)
        target_dir = builds_dir / build_id
        # Copy into a temporary directory then atomically rename to avoid partial state
        temp_dir = builds_dir / f".{build_id}.tmp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        try:
            shutil.copytree(build_path, temp_dir)
            # Remove old target if exists, then atomically replace
            if target_dir.exists():
                shutil.rmtree(target_dir)
            os.replace(str(temp_dir), str(target_dir))
        except Exception:
            # Clean up temp on failure
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        # Store relative path to build artifacts
        build_path_str = str(Path("catalog-builds") / build_id)
        # Load existing catalog (empty if not exists)
        catalog = load_catalog(path=config_path)
        now = datetime.now(timezone.utc).isoformat()
        # If alias exists, append a new revision to that entry
        if alias and alias in (catalog.get("aliases") or {}):
            mapping = catalog["aliases"][alias]
            entry_id = mapping["entry_id"]
            entry = next(
                e for e in catalog.get("entries", []) if e.get("entry_id") == entry_id
            )
            # Determine next revision number
            existing = [r.get("revision_id", "r0") for r in entry.get("history", [])]
            nums = [int(r[1:]) for r in existing if r.startswith("r")]
            next_num = max(nums, default=0) + 1
            revision_id = f"r{next_num}"
            revision = {
                "revision_id": revision_id,
                "created_at": now,
                "build": {"build_id": build_id, "path": build_path_str},
                # expr_hash recalculated fresh when inspecting
                "meta_digest": meta_digest,
            }
            if metadata_preview:
                revision["metadata"] = metadata_preview
            # Update entry
            entry.setdefault("history", []).append(revision)
            entry["current_revision"] = revision_id
            # Update alias mapping timestamp
            mapping["revision_id"] = revision_id
            mapping["updated_at"] = now
        else:
            # New entry and optional alias
            entry_id = str(uuid.uuid4())
            revision_id = "r1"
            revision = {
                "revision_id": revision_id,
                "created_at": now,
                "build": {"build_id": build_id, "path": build_path_str},
                # expr_hash recalculated fresh when inspecting
                "meta_digest": meta_digest,
            }
            if metadata_preview:
                revision["metadata"] = metadata_preview
            entry = {
                "entry_id": entry_id,
                "created_at": now,
                "current_revision": revision_id,
                "history": [revision],
            }
            catalog.setdefault("entries", []).append(entry)
            if alias:
                catalog.setdefault("aliases", {})[alias] = {
                    "entry_id": entry_id,
                    "revision_id": revision_id,
                    "updated_at": now,
                }
        # Save updated catalog to local catalog file
        save_catalog(catalog, path=config_path)
        print(f"Added build {build_id} as entry {entry_id} revision {revision_id}")

    elif args.subcommand == "ls":
        # Load catalog from local catalog file
        catalog = load_catalog(path=config_path)
        aliases = catalog.get("aliases", {})
        if aliases:
            print("Aliases:")
            for al, mapping in aliases.items():
                print(f"{al}\t{mapping['entry_id']}\t{mapping['revision_id']}")
        print("Entries:")
        for entry in catalog.get("entries", []):
            ent_id = entry.get("entry_id")
            curr_rev = entry.get("current_revision")
            build_id = None
            for rev in entry.get("history", []):
                if rev.get("revision_id") == curr_rev:
                    build_id = rev.get("build", {}).get("build_id")
                    break
            print(f"{ent_id}\t{curr_rev}\t{build_id}")

    elif args.subcommand == "inspect":
        # Load catalog from local catalog file
        catalog = load_catalog(path=config_path)

        target = resolve_target(args.entry, catalog)
        if target is None:
            print(f"Entry {args.entry} not found in catalog")
            return
        entry_id = target.entry_id
        revision_id = args.revision or target.rev
        # Find entry
        entry = next(
            (e for e in catalog.get("entries", []) if e.get("entry_id") == entry_id),
            None,
        )
        if entry is None:
            print(f"Entry {entry_id} not found in catalog")
            return
        # Determine revision
        if not revision_id:
            revision_id = entry.get("current_revision")
        revision = next(
            (
                r
                for r in entry.get("history", [])
                if r.get("revision_id") == revision_id
            ),
            None,
        )
        if revision is None:
            print(f"Revision {revision_id} not found for entry {entry_id}")
            return
        # Only show summary when not focusing on specific sections
        if args.full or not (args.plan or args.profiles or args.hashes):
            print("Summary:")
            print(f"  {'Entry ID':<13}: {entry_id}")
            entry_created = entry.get("created_at")
            if entry_created:
                print(f"  Entry Created: {entry_created}")
            print(f"  {'Revision ID':<13}: {revision_id}")
            revision_created = revision.get("created_at")
            if revision_created:
                print(f"  Revision Created: {revision_created}")
            expr_hash = (revision.get("expr_hashes") or {}).get("expr") or revision.get(
                "build", {}
            ).get("build_id")
            print(f"  {'Expr Hash':<13}: {expr_hash}")
            meta_digest = revision.get("meta_digest")
            if meta_digest:
                print(f"  {'Meta Digest':<13}: {meta_digest}")
        # Resolve build directory path (handle relative paths)
        bp = revision.get("build", {}).get("path")
        build_dir = Path(bp) if bp else None
        if build_dir and not build_dir.is_absolute():
            build_dir = config_dir / build_dir
        expr = None
        schema = None
        if build_dir and (
            args.full or args.plan or args.schema or args.profiles or args.hashes
        ):
            from xorq.ibis_yaml.compiler import load_expr

            try:
                expr = load_expr(build_dir)
                schema = expr.schema()
            except Exception as e:
                print(f"Error loading expression for DAG: {e}")
        if args.full or args.plan:
            print("\nPlan:")
            if expr is not None:
                print(expr)
            else:
                print("  No plan available.")
        if args.full or args.schema:
            print("\nSchema:")
            if schema:
                for name, dtype in schema.items():
                    print(f"  {name}: {dtype}")
            else:
                print("  No schema available.")
        if args.full or args.profiles:
            # Load profiles from build directory
            profiles_file = build_dir / "profiles.yaml" if build_dir else None
            print("\nProfiles:")
            if profiles_file and profiles_file.exists():
                try:
                    text = profiles_file.read_text()
                    data = yaml.safe_load(text) or {}
                    for name, profile in data.items():
                        print(f"  {name}: {profile}")
                except Exception as e:
                    print(f"  Error loading profiles: {e}")
            else:
                print("  No profiles available.")
        if args.full or args.hashes:
            print("\nNode hashes:")
            # Dynamically compute hashes for CachedNode instances
            if expr is not None:
                try:
                    import dask

                    from xorq.common.utils.graph_utils import walk_nodes
                    from xorq.expr.relations import CachedNode

                    nodes = walk_nodes((CachedNode,), expr)
                    if nodes:
                        for node in nodes:
                            h = dask.base.tokenize(node.to_expr())
                            print(f"  {h}")
                    else:
                        print("  No node hashes recorded.")
                except Exception as e:
                    print(f"  Error computing node hashes: {e}")
            else:
                print("  No node hashes recorded.")
        return
    elif args.subcommand == "info":
        # Show top-level catalog info
        # Load catalog from local catalog file
        catalog = load_catalog(path=config_path)
        entries = catalog.get("entries", []) or []
        aliases = catalog.get("aliases", {}) or {}
        print(f"Catalog path: {config_path}")
        print(f"Entries: {len(entries)}")
        print(f"Aliases: {len(aliases)}")
        return
    elif args.subcommand == "rm":
        # Remove an entry or alias from the catalog
        # Load catalog from local catalog file
        catalog = load_catalog(path=config_path)
        token = args.entry
        # Remove alias if present
        aliases = catalog.get("aliases", {}) or {}
        if token in aliases:
            aliases.pop(token, None)
            # If no aliases remain, clean up key
            if not aliases:
                catalog.pop("aliases", None)
            # Save updated catalog
            save_catalog(catalog, path=config_path)
            print(f"Removed alias {token}")
            return
        # Remove entry if present
        entries = catalog.get("entries", [])
        for i, entry in enumerate(entries):
            if entry.get("entry_id") == token:
                # Remove entry and any related aliases
                entries.pop(i)
                # Clean aliases pointing to this entry
                to_remove = [
                    a for a, m in aliases.items() if m.get("entry_id") == token
                ]
                for a in to_remove:
                    aliases.pop(a, None)
                # Clean empty aliases dict
                if not aliases:
                    catalog.pop("aliases", None)
                # Save updated catalog
                save_catalog(catalog, path=config_path)
                print(f"Removed entry {token}")
                return
        # Not found
        print(f"Entry {token} not found in catalog")
        return
    elif args.subcommand == "export":
        # Export catalog.yaml and all builds to a target directory
        export_dir = Path(args.output_path)
        # Ensure export directory exists
        if export_dir.exists() and not export_dir.is_dir():
            print(f"Export path exists and is not a directory: {export_dir}")
            return
        export_dir.mkdir(parents=True, exist_ok=True)
        # Copy catalog file
        if config_path.exists():
            shutil.copy2(config_path, export_dir / config_path.name)
        else:
            print(f"No catalog found at {config_path}")
            return
        # Copy builds directory
        src_builds = config_dir / "catalog-builds"
        if src_builds.exists() and src_builds.is_dir():
            dest_builds = export_dir / src_builds.name
            if dest_builds.exists():
                shutil.rmtree(dest_builds)
            shutil.copytree(src_builds, dest_builds)
        print(f"Exported catalog and builds to {export_dir}")
        return
    elif args.subcommand == "diff-builds":
        # Compare two build artifacts via git diff --no-index
        code = do_diff_builds(args.left, args.right, args.files, args.all)
        sys.exit(code)
    else:
        print(f"Unknown catalog subcommand: {args.subcommand}")


def ps_command(cache_dir: str) -> None:
    """List active xorq servers."""
    record_dir = Path(cache_dir) / "servers"
    headers, rows = format_server_table(filter_running(get_server_records(record_dir)))
    do_print_table(headers, rows)


def cache_command(args):
    """
    Cache a built expression output to Parquet using a CachedNode.
    """
    from xorq.caching import ParquetStorage
    from xorq.common.utils.caching_utils import find_backend

    # Resolve build target
    catalog = load_catalog()
    build_dir = resolve_build_dir(args.target, catalog)
    if build_dir is None or not build_dir.exists() or not build_dir.is_dir():
        print(f"Build target not found: {args.target}")
        sys.exit(2)
    # Load expression
    expr = load_expr(build_dir)
    # Determine backend for caching
    con, _ = find_backend(expr.op(), use_default=True)
    # Setup Parquet storage at given cache directory
    base_path = Path(args.cache_dir)
    storage = ParquetStorage(source=con, relative_path=Path("."), base_path=base_path)
    # Attach cache node
    cached_expr = expr.cache(storage=storage)
    # Execute to materialize cache
    try:
        for _ in cached_expr.to_pyarrow_batches():
            pass
    except Exception as e:
        print(f"Error during caching execution: {e}")
        sys.exit(1)
    # Report cache files
    cache_path = storage.cache.storage.path
    print(f"Cache written to: {cache_path}")
    for pq_file in sorted(cache_path.rglob("*.parquet")):
        print(f"  {pq_file.relative_to(cache_path)}")


def profile_command(args):
    """
    Manage connection profiles: add new profiles.
    """
    sub = args.subcommand
    if sub == "add":
        alias = args.alias
        con_name = args.con_name
        params = {}
        for p in args.param:
            if "=" not in p:
                print(f"Invalid parameter '{p}', expected KEY=VALUE")
                sys.exit(1)
            k, v = p.split("=", 1)
            params[k] = v
        # Create and save profile
        from xorq.vendor.ibis.backends.profiles import Profile

        prof = Profile(con_name=con_name, kwargs_tuple=tuple(params.items()))
        try:
            path = prof.save(alias=alias, clobber=False)
            print(f"Profile '{alias}' saved to {path}")
        except ValueError as e:
            print(f"Error saving profile: {e}")
            sys.exit(1)
    else:
        print(f"Unknown profile subcommand: {sub}")
        sys.exit(2)


# === Server Recording Utilities ===


@frozen
class ServerRecord:
    pid: int
    command: str
    target: str
    port: Optional[int]
    start_time: datetime
    node_hash: Optional[str] = None

    def clone(self, **changes) -> "ServerRecord":
        """Return a new ServerRecord with updated fields."""
        return evolve(self, **changes)


def maybe_make_server_record(data: dict) -> Optional[ServerRecord]:
    try:
        pid = data["pid"]
        command = data["command"]
        target = data.get("target", "")
        port = data.get("port")
        node_hash = data.get("to_unbind_hash")
        start_time = datetime.fromisoformat(data["start_time"])
        return ServerRecord(
            pid=pid,
            command=command,
            target=target,
            port=port,
            start_time=start_time,
            node_hash=node_hash,
        )
    except Exception:
        return None


def get_server_records(record_dir: Path) -> Tuple[ServerRecord, ...]:
    if not record_dir.exists():
        return ()
    records: list[ServerRecord] = []
    for f in record_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        rec = maybe_make_server_record(data)
        if rec is not None:
            records.append(rec)
    return tuple(records)


def filter_running(records: Tuple[ServerRecord, ...]) -> Tuple[ServerRecord, ...]:
    running: list[ServerRecord] = []
    for rec in records:
        try:
            os.kill(rec.pid, 0)
            running.append(rec)
        except Exception:
            continue
    return tuple(running)


def format_server_table(
    records: Tuple[ServerRecord, ...],
) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]:
    headers = ("TARGET", "STATE", "COMMAND", "HASH", "PID", "PORT", "UPTIME")
    rows: list[tuple[str, ...]] = []
    now = datetime.now()
    for rec in records:
        state = "running"
        try:
            delta = now - rec.start_time
            hours, rem = divmod(int(delta.total_seconds()), 3600)
            minutes, _ = divmod(rem, 60)
            uptime = f"{hours}h{minutes}m"
        except Exception:
            uptime = ""
        rows.append(
            (
                rec.target,
                state,
                rec.command,
                rec.node_hash or "",
                str(rec.pid),
                str(rec.port) if rec.port is not None else "",
                uptime,
            )
        )
    return headers, tuple(rows)


def do_print_table(headers: Tuple[str, ...], rows: Tuple[Tuple[str, ...], ...]) -> None:
    if rows:
        widths = [max(len(cell) for cell in col) for col in zip(headers, *rows)]
    else:
        widths = [len(h) for h in headers]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    for row in rows:
        print(fmt.format(*row))


def make_server_record(
    pid: int,
    command: str,
    target: str,
    port: Optional[int] = None,
    start_time: Optional[datetime] = None,
    node_hash: Optional[str] = None,
) -> ServerRecord:
    """Factory: create a ServerRecord with optional start_time."""
    ts = start_time or datetime.now()
    return ServerRecord(
        pid=pid,
        command=command,
        target=target,
        port=port,
        start_time=ts,
        node_hash=node_hash,
    )


def do_save_server_record(record: ServerRecord, record_dir: Path) -> None:
    """Side effect: save a ServerRecord to JSON file in record_dir."""
    record_dir.mkdir(parents=True, exist_ok=True)
    path = record_dir / f"{record.pid}.json"
    data: dict = {
        "pid": record.pid,
        "command": record.command,
        "target": record.target,
        "port": record.port,
        "start_time": record.start_time.isoformat(),
    }
    if record.node_hash is not None:
        data["to_unbind_hash"] = record.node_hash
    path.write_text(json.dumps(data))


def get_diff_file_list(
    left_dir: Path, right_dir: Path, files: list[str] | None, all_flag: bool
) -> tuple[str, ...]:
    default = ("expr.yaml",)
    if files is not None:
        return tuple(files)
    if all_flag:
        # Exclude node_hashes.yaml as node hashes are not used here
        default_files = (
            "expr.yaml",
            "deferred_reads.yaml",
            "profiles.yaml",
            "sql.yaml",
            "metadata.json",
        )
        sqls = {p.relative_to(left_dir).as_posix() for p in left_dir.rglob("*.sql")} | {
            p.relative_to(right_dir).as_posix() for p in right_dir.rglob("*.sql")
        }
        return default_files + tuple(sorted(sqls))
    return default
