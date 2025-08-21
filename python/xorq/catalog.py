import difflib
import functools
import json
import operator
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import toolz
import yaml
from attrs import evolve, field, frozen
from attrs.validators import deep_iterable, instance_of, optional

import xorq as xo
from xorq.ibis_yaml.compiler import (
    BuildManager,
    load_expr,
)


def get_default_catalog_path():
    # dynamically retrieve: tests need to monkeypatch XDG_CONFIG_HOME
    return (
        Path(os.environ.get("XDG_CONFIG_HOME") or Path.home().joinpath(".config"))
        .joinpath("xorq", "catalog.yaml")
        .absolute()
    )


def get_catalog_path(path: Optional[Union[str, Path]] = None) -> Path:
    """Return the catalog file path, using XDG_CONFIG_HOME if set or default to ~/.config."""
    return Path(path) if path else get_default_catalog_path()


get_now_utc = functools.partial(datetime.now, timezone.utc)


@frozen
class CatalogMetadata:
    """Catalog metadata."""

    # FIXME: make uuid
    catalog_id: str = field(validator=instance_of(str), factory=uuid.uuid4)
    # FIXME: make datetime.datetime
    created_at: str = field(validator=instance_of(datetime), factory=get_now_utc)
    updated_at: str = field(validator=instance_of(datetime), factory=get_now_utc)
    tool_version: str = field(validator=instance_of(str), default=xo.__version__)

    def with_updated_timestamp(self) -> "CatalogMetadata":
        """Return new metadata with updated timestamp."""
        return self.evolve(updated_at=get_now_utc().isoformat())

    def evolve(self, **kwargs) -> "CatalogMetadata":
        """Create a copy with specified changes."""
        return evolve(self, **kwargs)

    to_dict = operator.methodcaller("__getstate__")


@frozen
class Build:
    """Build information."""

    build_id: Optional[str] = field(default=None, validator=optional(instance_of(str)))
    # FIXME: make Path
    path: Optional[Path] = field(
        default=None,
        validator=optional(instance_of(Path)),
        converter=toolz.curried.excepts(Exception, Path),
    )

    def evolve(self, **kwargs) -> "Build":
        """Create a copy with specified changes."""
        return evolve(self, **kwargs)

    def to_dict(self):
        dct = self.__getstate__()
        dct = dct | {
            "path": str(self.path) if self.path else None,
        }
        return dct

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)


def convert_datetime(value):
    match value:
        case None:
            return get_now_utc()
        case str():
            return datetime.fromisoformat(value)
        case datetime():
            return value
        case _:
            raise ValueError


def convert_build(value):
    match value:
        case None:
            return None
        case dict():
            return Build(**value)
        case Build():
            return value
        case _:
            raise ValueError


@frozen
class Revision:
    """Revision data."""

    # FIXME: make int
    revision_id: str = field(validator=instance_of(str))
    created_at: str = field(
        validator=instance_of(datetime),
        factory=get_now_utc,
        converter=convert_datetime,
    )
    build: Optional[Build] = field(
        default=None, validator=optional(instance_of(Build)), converter=convert_build
    )
    expr_hashes: Optional[Dict[str, str]] = field(default=None)
    node_hashes: Tuple[str, ...] = field(
        factory=tuple,
        validator=deep_iterable(instance_of(str), instance_of(tuple)),
        converter=tuple,
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

    def to_dict(self):
        dct = self.__getstate__()
        dct = dct | {
            "created_at": dct["created_at"].isoformat(),
            "build": self.build.to_dict() if self.build else None,
        }
        return dct

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)


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

    @property
    def created_at(self):
        return min(
            (revision.created_at for revision in self.history),
            default=None,
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

    def to_dict(self):
        dct = self.__getstate__()
        dct = dct | {
            "history": tuple(revision.to_dict() for revision in self.history),
        }
        return dct

    @classmethod
    def from_dict(cls, dct):
        dct = dct.copy()
        created_at = dct.pop("created_at", None)
        history = tuple(Revision.from_dict(rev) for rev in dct.get("history", ()))
        if created_at is not None:
            assert datetime.fromisoformat(created_at) == min(
                rev.created_at for rev in history
            )
        return cls(**dct | {"history": history})


@frozen
class Alias:
    entry_id: str = field(validator=instance_of(str))
    revision_id: Optional[str] = field(
        default=None, validator=optional(instance_of(str))
    )
    updated_at: str = field(
        validator=optional(instance_of(datetime)),
        default=None,
        converter=convert_datetime,
    )

    def evolve(self, **kwargs) -> "Alias":
        """Create a copy with specified changes."""
        return evolve(self, **kwargs)

    def to_dict(self):
        dct = self.__getstate__()
        dct = dct | {
            "updated_at": self.updated_at.isoformat(),
        }
        return dct

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)


@frozen
class XorqCatalog:
    """Xorq Catalog container."""

    aliases: Mapping[str, Alias] = field(factory=dict)
    entries: Tuple[Entry, ...] = field(
        validator=deep_iterable(instance_of(Entry), instance_of(tuple)),
        factory=tuple,
    )
    api_version: str = field(default="xorq.dev/v1", validator=instance_of(str))
    kind: str = field(default="XorqCatalog", validator=instance_of(str))
    metadata: CatalogMetadata = field(
        validator=optional(instance_of(CatalogMetadata)), default=None
    )

    def with_entry(self, entry: Entry) -> "XorqCatalog":
        """Add or update an entry."""
        existing_entries = tuple(
            e for e in self.entries if e.entry_id != entry.entry_id
        )
        return self.evolve(entries=existing_entries + (entry,))

    def with_alias(self, name: str, alias: Alias) -> "XorqCatalog":
        """Add or update an alias."""
        return self.evolve(aliases=self.aliases | {name: alias})

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

    def resolve_target(self, target: str):
        return Target.from_str(target, self)

    def to_dict(self):
        dct = self.__getstate__()
        dct = dct | {
            "aliases": {name: alias.to_dict() for name, alias in self.aliases.items()},
            "entries": tuple(entry.to_dict() for entry in self.entries),
            "metadata": self.metadata.to_dict() if self.metadata else self.metadata,
        }
        return dct

    def to_yaml(self, path):
        dct = self.to_dict()
        with path.open("wt") as fh:
            yaml.safe_dump(dct, fh, sort_keys=False)

    def save(self, path):
        catalog_path = get_catalog_path(path)
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_yaml(catalog_path)

    @classmethod
    def from_dict(cls, dct):
        aliases = dct.get("aliases", {})
        entries = dct.get("entries", ())
        return cls(
            **dct
            | {
                "aliases": {k: Alias.from_dict(v) for k, v in aliases.items()},
                "entries": tuple(Entry.from_dict(el) for el in entries),
            }
        )

    @classmethod
    def from_path(cls, path):
        with Path(path).open() as fh:
            dct = yaml.safe_load(fh)
        return cls.from_dict(dct) if dct else cls()

    @classmethod
    def from_default(cls):
        pass


@frozen
class Target:
    entry_id: str = field(validator=instance_of(str))
    rev: Optional[str] = field(default=None, validator=optional(instance_of(str)))
    alias: bool = field(default=False, validator=instance_of(bool))

    @classmethod
    def from_str(cls, target: str, catalog: XorqCatalog = None):
        catalog = catalog or load_catalog()
        (base, rev) = target.split("@", 1) if "@" in target else (target, None)
        if base == "entry" and catalog.entries:
            entry, *_ = catalog.entries
            return Target(
                entry_id=entry.entry_id, rev=rev or entry.current_revision, alias=False
            )
        if alias := catalog.aliases.get(base):
            return Target(
                entry_id=alias.entry_id, rev=rev or alias.revision_id, alias=True
            )
        if entry := next(
            (entry for entry in catalog.entries if entry.entry_id == base), None
        ):
            return Target(entry_id=base, rev=rev or entry.current_revision, alias=False)
        return None


def load_catalog(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load catalog as raw dict; return minimal structure if not found."""
    catalog_path = get_catalog_path(path)
    catalog = (
        XorqCatalog.from_path(catalog_path) if catalog_path.exists() else XorqCatalog()
    )
    return catalog


def resolve_build_dir(
    token: str, catalog: Optional[XorqCatalog] = None
) -> Optional[Path]:
    """Resolve build directory from raw catalog dict."""

    def get_next_absolute_path(paths):
        path = next(filter(None, paths), None)
        if path:
            path = Path(path)
            # If stored path is relative, interpret relative to catalog config directory
            if not path.is_absolute():
                path = get_default_catalog_path().parent.joinpath(path)
            return path
        else:
            return None

    def get_explicit_build_entry(token, catalog):
        # Match on explicit build_id entries
        paths = tuple(
            rev.build.path
            for entry in catalog.entries
            for rev in entry.history
            if rev.build.build_id == token
        )
        return get_next_absolute_path(paths)

    def get_entry_revision_target(t, catalog):
        gen = (
            rev.build.path
            for entry in catalog.entries
            for rev in entry.history
            if entry.entry_id == t.entry_id and rev.revision_id == t.rev
        )
        return get_next_absolute_path(gen)

    path = Path(token)
    if path.exists() and path.is_dir():
        return path
    path = get_explicit_build_entry(token, catalog)
    if path:
        return path

    # Match on entry@revision targets
    t = catalog.resolve_target(token)
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
        now = get_now_utc().isoformat()
        # If alias exists, append a new revision to that entry
        if alias and (mapping := catalog.aliases.get(alias)):
            entry = next(
                (
                    entry
                    for entry in catalog.entries
                    if entry.entry_id == mapping.entry_id
                ),
                None,
            )
            # Determine next revision number
            # FIXME: use XorqCatalog
            existing = [r.get("revision_id", "r0") for r in entry.get("history", [])]
            nums = [int(r[1:]) for r in existing if r.startswith("r")]
            next_num = max(nums, default=0) + 1
            revision_id = f"r{next_num}"
            revision = {
                "revision_id": revision_id,
                "created_at": now,
                "build": Build.from_dict(
                    {"build_id": build_id, "path": build_path_str}
                ),
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
            entry = Entry.from_dict(
                {
                    "entry_id": entry_id,
                    "created_at": now,
                    "current_revision": revision_id,
                    "history": [revision],
                }
            )
            catalog = catalog.with_entry(entry)
            if alias:
                catalog = catalog.with_alias(
                    alias,
                    Alias.from_dict(
                        {
                            "entry_id": entry_id,
                            "revision_id": revision_id,
                            "updated_at": now,
                        }
                    ),
                )
        # Save updated catalog to local catalog file
        catalog.save(path=config_path)
        print(f"Added build {build_id} as entry {entry_id} revision {revision_id}")

    elif args.subcommand == "ls":
        # Load catalog from local catalog file
        catalog = load_catalog(path=config_path)
        if aliases := catalog.aliases:
            print("Aliases:")
            print(
                "\n".join(
                    f"{al}\t{mapping.entry_id}\t{mapping.revision_id}"
                    for al, mapping in aliases.items()
                )
            )
        print("Entries:")
        for entry in catalog.entries:
            curr_rev = entry.current_revision
            build_id = None
            for rev in entry.history:
                if rev.revision_id == curr_rev:
                    build_id = rev.build.build_id
                    break
            print(f"{entry.entry_id}\t{curr_rev}\t{build_id}")

    elif args.subcommand == "inspect":
        # Load catalog from local catalog file
        catalog = load_catalog(path=config_path)

        target = catalog.resolve_target(args.entry)
        if target is None:
            print(f"Entry {args.entry} not found in catalog")
            return
        entry_id = target.entry_id
        revision_id = args.revision or target.rev
        # Find entry
        if (
            entry := next(
                (entry for entry in catalog.entries if entry.entry_id == entry_id), None
            )
        ) is None:
            print(f"Entry {entry_id} not found in catalog")
            return
        # Determine revision
        if not revision_id:
            revision_id = entry.current_revision
        if (
            revision := next(
                (
                    revision
                    for revision in entry.history
                    if revision.revision_id == revision_id
                ),
                None,
            )
        ) is None:
            print(f"Revision {revision_id} not found for entry {entry_id}")
            return
        # Only show summary when not focusing on specific sections
        if args.full or not (args.plan or args.profiles or args.hashes):
            print("Summary:")
            print(f"  {'Entry ID':<13}: {entry_id}")
            if entry_created := entry.created_at:
                print(f"  Entry Created: {entry_created}")
            print(f"  {'Revision ID':<13}: {revision_id}")
            revision_created = revision.created_at
            if revision_created:
                print(f"  Revision Created: {revision_created}")
            expr_hash = (revision.expr_hashes or {}).get(
                "expr"
            ) or revision.build.build_id
            print(f"  {'Expr Hash':<13}: {expr_hash}")
            meta_digest = revision.meta_digest
            if meta_digest:
                print(f"  {'Meta Digest':<13}: {meta_digest}")
        # Resolve build directory path (handle relative paths)
        bp = revision.build.path
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
        print(f"Catalog path: {config_path}")
        print(f"Entries: {len(catalog.entries)}")
        print(f"Aliases: {len(catalog.aliases)}")
        return
    elif args.subcommand == "rm":
        # Remove an entry or alias from the catalog
        # Load catalog from local catalog file
        catalog = load_catalog(path=config_path)
        token = args.entry
        # Remove entry if present
        entry = next(
            (entry for entry in catalog.entries if entry.entry_id == token),
            None,
        )
        if entry:
            entries = tuple(
                entry for entry in catalog.entries if entry.entry_id != token
            )
            # Remove entry and any related aliases
            others = tuple(
                name
                for name, value in catalog.aliases.items()
                if value.entry_id == token
            )
            aliases = toolz.dissoc(catalog.aliases, *others)
            # Save updated catalog
            catalog = catalog.evolve(entries=entries, aliases=aliases)
            catalog.save(path=config_path)
            print(f"Removed entry {token}")
            return
        elif catalog.aliases.pop(token, None):
            # Remove alias if present
            # Save updated catalog
            catalog.save(path=config_path)
            print(f"Removed alias {token}")
            return
        else:
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

    def do_print_table(
        headers: Tuple[str, ...], rows: Tuple[Tuple[str, ...], ...]
    ) -> None:
        if rows:
            widths = [max(len(cell) for cell in col) for col in zip(headers, *rows)]
        else:
            widths = [len(h) for h in headers]
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*headers))
        for row in rows:
            print(fmt.format(*row))

    def format_server_table(
        records: Tuple[ServerRecord, ...],
    ) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]:
        headers = ("TARGET", "STATE", "COMMAND", "HASH", "PID", "PORT", "UPTIME")
        rows: list[tuple[str, ...]] = []
        now = get_now_utc()
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

    headers, rows = format_server_table(
        ServerRecord.load_records(Path(cache_dir) / "servers")
    )
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


@frozen
class ServerRecord:
    pid: int = field(validator=instance_of(int))
    command: str = field(validator=instance_of(str))
    target: str = field(validator=instance_of(str))
    port: Optional[int] = field(validator=optional(instance_of(int)), default=None)
    start_time: datetime = field(
        validator=instance_of(datetime),
        factory=get_now_utc,
        converter=convert_datetime,
    )
    node_hash: Optional[str] = field(validator=optional(instance_of(str)), default=None)

    def clone(self, **changes) -> "ServerRecord":
        """Return a new ServerRecord with updated fields."""
        return evolve(self, **changes)

    def to_dict(self):
        return self.__getstate__()

    def to_json_dict(self):
        data = toolz.dissoc(
            self.to_dict(),
            "node_hash",
        )
        data = (
            data
            | {
                "start_time": self.start_time.isoformat(),
            }
            | toolz.valfilter(
                bool,
                {
                    "to_unbind_hash": self.node_hash,
                },
            )
        )
        return data

    def save(self, record_dir: Path) -> Path:
        """Side effect: save a ServerRecord to JSON file in record_dir."""
        record_dir.mkdir(parents=True, exist_ok=True)
        path = record_dir / f"{self.pid}.json"
        path.write_text(json.dumps(self.to_json_dict()))
        return path

    @property
    def running(self):
        try:
            os.kill(self.pid, 0)
            return True
        except Exception:
            return False

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)

    @classmethod
    def from_path(cls, path):
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def load_records(cls, record_dir: Path, only_running=True):
        if not record_dir.exists():
            return tuple()
        # FIXME: remove json files of records that aren't running
        records = tuple(
            map(toolz.excepts(Exception, cls.from_path), record_dir.glob("*.json"))
        )
        if only_running:
            records = tuple(record for record in records if record.running)
        return records


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
