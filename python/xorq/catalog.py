import functools
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import toolz
import yaml
from attrs import evolve, field, frozen
from attrs.validators import deep_iterable, instance_of, optional
from toolz import curry

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


def to_dict(self):
    return self.__getstate__()


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

    to_dict = to_dict


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

    to_dict = to_dict

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

    def maybe_get_revision(self, build_id: str) -> Optional[Revision]:
        """Get revision by build_id if it exists."""
        gen = (
            rev
            for entry in self.entries
            for rev in entry.history
            if rev.build.build_id == build_id
        )
        first = next(gen, None)
        rest = tuple(gen)
        if rest:
            # unclear what to do if multiple revisions share the same build_id
            raise ValueError
        return first

    def maybe_get_revision_by_token(self, token: str) -> Optional[Revision]:
        """Get revision by token if it exists."""
        t = self.resolve_target(token)
        if t is None:
            return None
        else:
            gen = (
                rev
                for entry in self.entries
                for rev in entry.history
                if rev.build.build_id == t.build_id and rev.revision_id == t.rev
            )
            return next(gen, None)

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
        if path.exists():
            with Path(path).open() as fh:
                dct = yaml.safe_load(fh)
            return cls.from_dict(dct) if dct else cls()
        else:
            return cls()

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
        if alias := catalog.aliases.get(base):
            return Target(
                entry_id=alias.entry_id, rev=rev or alias.revision_id, alias=True
            )
        if entry := catalog.maybe_get_entry(base):
            return Target(entry_id=base, rev=rev or entry.current_revision, alias=False)
        return None


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
        data = data | toolz.valfilter(
            bool,
            {
                "to_unbind_hash": self.node_hash,
            },
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


load_catalog = toolz.compose(XorqCatalog.from_path, get_catalog_path)


def resolve_build_dir(
    token: str, catalog: Optional[XorqCatalog] = None
) -> Optional[Path]:
    """Resolve build directory from raw catalog dict."""

    def absolutify(path):
        if not path.is_absolute():
            path = get_default_catalog_path().parent.joinpath(path)
        return path

    path = Path(token)
    if path.exists() and path.is_dir():
        return path
    for revision in (
        catalog.maybe_get_revision(build_id=token),
        catalog.maybe_get_revision_by_token(token),
    ):
        if revision.build and revision.build.path:
            return absolutify(revision.build.path)
    return None


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
    if left_dir is None:
        print(f"Build target not found: {left}")
        return None
    if right_dir is None:
        print(f"Build target not found: {right}")
        return None
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


def _get_catalog_paths() -> tuple[Path, Path]:
    """Return (config_path, config_dir) for the catalog."""
    config_path = get_catalog_path()
    return config_path, config_path.parent


def _do_unknown(args):
    print(f"Unknown catalog subcommand: {args.subcommand}")


def do_catalog_add(args):
    """Side effect: add a build into the local catalog."""
    config_path, config_dir = _get_catalog_paths()
    build_path = Path(args.build_path).resolve()
    alias = args.alias
    build_id, meta_digest, metadata_preview = BuildManager.validate_build(build_path)
    config_dir.mkdir(parents=True, exist_ok=True)
    builds_dir = config_dir / "catalog-builds"
    builds_dir.mkdir(parents=True, exist_ok=True)
    target_dir = builds_dir / build_id
    temp_dir = builds_dir / f".{build_id}.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    try:
        shutil.copytree(build_path, temp_dir)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        os.replace(str(temp_dir), str(target_dir))
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    build_path_str = str(Path("catalog-builds") / build_id)
    catalog = load_catalog(path=config_path)
    now = get_now_utc().isoformat()
    if alias and (mapping := catalog.aliases.get(alias)):
        entry = next(
            (e for e in catalog.entries if e.entry_id == mapping.entry_id),
            None,
        )
        existing = [r.get("revision_id", "r0") for r in entry.get("history", [])]
        nums = [int(r[1:]) for r in existing if r.startswith("r")]
        next_num = max(nums, default=0) + 1
        revision_id = f"r{next_num}"
        revision = {
            "revision_id": revision_id,
            "created_at": now,
            "build": Build.from_dict({"build_id": build_id, "path": build_path_str}),
            "meta_digest": meta_digest,
        }
        if metadata_preview:
            revision["metadata"] = metadata_preview
        entry.setdefault("history", []).append(revision)
        entry["current_revision"] = revision_id
        mapping["revision_id"] = revision_id
        mapping["updated_at"] = now
    else:
        entry_id = str(uuid.uuid4())
        revision_id = "r1"
        revision = {
            "revision_id": revision_id,
            "created_at": now,
            "build": {"build_id": build_id, "path": build_path_str},
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
    catalog.save(path=config_path)
    print(f"Added build {build_id} as entry {entry_id} revision {revision_id}")


def do_catalog_ls(args):
    """List entries and aliases in the catalog."""
    config_path, _ = _get_catalog_paths()
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


@curry
def maybe_resolve_target(entry_name: str, catalog) -> Optional[Any]:
    return catalog.resolve_target(entry_name)


@curry
def maybe_get_entry(entry_id: str, entries) -> Optional[Any]:
    return next((e for e in entries if e.entry_id == entry_id), None)


@curry
def maybe_get_revision(revision_id: str, history) -> Optional[Any]:
    return next((r for r in history if r.revision_id == revision_id), None)


def compute_build_dir(
    build_path: Optional[Union[str, Path]], config_dir: Path
) -> Optional[Path]:
    if not build_path:
        return None
    build_dir = Path(build_path)
    if not build_dir.is_absolute():
        build_dir = config_dir / build_dir
    return build_dir


def do_catalog_info(args):
    """Show top-level catalog info."""
    config_path, _ = _get_catalog_paths()
    catalog = load_catalog(path=config_path)
    print(f"Catalog path: {config_path}")
    print(f"Entries: {len(catalog.entries)}")
    print(f"Aliases: {len(catalog.aliases)}")


def do_catalog_rm(args):
    """Remove an entry or alias from the catalog."""
    config_path, _ = _get_catalog_paths()
    catalog = load_catalog(path=config_path)
    token = args.entry
    entry = next((e for e in catalog.entries if e.entry_id == token), None)
    if entry:
        entries = tuple(e for e in catalog.entries if e.entry_id != token)
        others = tuple(
            name for name, value in catalog.aliases.items() if value.entry_id == token
        )
        aliases = toolz.dissoc(catalog.aliases, *others)
        catalog = catalog.evolve(entries=entries, aliases=aliases)
        catalog.save(path=config_path)
        print(f"Removed entry {token}")
        return
    elif catalog.aliases.pop(token, None):
        catalog.save(path=config_path)
        print(f"Removed alias {token}")
        return
    else:
        print(f"Entry {token} not found in catalog")


def do_catalog_export(args):
    """Export catalog and builds to a target directory."""
    config_path, config_dir = _get_catalog_paths()
    export_dir = Path(args.output_path)
    if export_dir.exists() and not export_dir.is_dir():
        print(f"Export path exists and is not a directory: {export_dir}")
        return
    export_dir.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        shutil.copy2(config_path, export_dir / config_path.name)
    else:
        print(f"No catalog found at {config_path}")
        return
    src_builds = config_dir / "catalog-builds"
    if src_builds.exists() and src_builds.is_dir():
        dest_builds = export_dir / src_builds.name
        if dest_builds.exists():
            shutil.rmtree(dest_builds)
        shutil.copytree(src_builds, dest_builds)
    print(f"Exported catalog and builds to {export_dir}")


def do_catalog_diff_builds(args):
    """Compare two build artifacts via git diff --no-index."""
    code = do_diff_builds(args.left, args.right, args.files, args.all)
    sys.exit(code)


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


_CATALOG_HANDLER_MAP = {
    "add": do_catalog_add,
    "ls": do_catalog_ls,
    "info": do_catalog_info,
    "rm": do_catalog_rm,
    "export": do_catalog_export,
    "diff-builds": do_catalog_diff_builds,
}


def _catalog_command_dispatch(args):
    return _CATALOG_HANDLER_MAP.get(args.subcommand, _do_unknown)(args)


catalog_command = _catalog_command_dispatch


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
    from xorq.caching import ParquetStorage
    from xorq.common.utils.caching_utils import find_backend

    catalog = load_catalog()
    build_dir = resolve_build_dir(args.target, catalog)
    if build_dir is None or not build_dir.exists() or not build_dir.is_dir():
        print(f"Build target not found: {args.target}")
        sys.exit(2)
    expr = load_expr(build_dir)
    con, _ = find_backend(expr.op(), use_default=True)
    base_path = Path(args.cache_dir)
    storage = ParquetStorage(source=con, relative_path=Path("."), base_path=base_path)
    cached_expr = expr.cache(storage=storage)
    try:
        for _ in cached_expr.to_pyarrow_batches():
            pass
    except Exception as e:
        print(f"Error during caching execution: {e}")
        sys.exit(1)
    cache_path = storage.cache.storage.path
    print(f"Cache written to: {cache_path}")
    for pq_file in sorted(cache_path.rglob("*.parquet")):
        print(f"  {pq_file.relative_to(cache_path)}")


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
