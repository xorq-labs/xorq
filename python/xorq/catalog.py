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

import xorq as xo
from xorq.common.utils.dask_normalize.dask_normalize_utils import (
    file_digest,
)
from xorq.common.utils.func_utils import (
    if_not_none,
)
from xorq.ibis_yaml.compiler import (
    DumpFiles,
    load_expr,
)


CATALOG_YAML_FILENAME = "catalog.yaml"


def get_default_catalog_path():
    # dynamically retrieve: tests need to monkeypatch XDG_CONFIG_HOME
    return (
        Path(os.environ.get("XDG_CONFIG_HOME") or Path.home().joinpath(".config"))
        .joinpath("xorq", CATALOG_YAML_FILENAME)
        .absolute()
    )


def get_project_catalog_path() -> Optional[Path]:
    """Return path to project-level catalog if it exists (.xorq/catalog.yaml)."""
    project_catalog = Path.cwd() / ".xorq" / CATALOG_YAML_FILENAME
    return project_catalog if project_catalog.exists() else None


def get_catalog_path(path: Optional[Union[str, Path]] = None) -> Path:
    """Return the catalog file path. Priority: explicit path > project catalog > default user catalog."""
    if path:
        p = Path(path)
        # If path is a directory, append catalog.yaml
        if p.is_dir():
            return p / CATALOG_YAML_FILENAME
        # If path doesn't have .yaml extension, assume it's a directory and append catalog.yaml
        if not p.name.endswith(".yaml"):
            return p / CATALOG_YAML_FILENAME
        return p
    if project_catalog := get_project_catalog_path():
        return project_catalog
    return get_default_catalog_path()


def get_builds_dir_for_catalog(catalog_path: Path) -> Path:
    """Return builds directory for a given catalog path.

    Project catalogs (.xorq/catalog.yaml) use .xorq/builds/
    User catalogs (~/.config/xorq/catalog.yaml) use catalog-builds/
    """
    catalog_dir = catalog_path.parent
    # Check if this is a project catalog (.xorq directory)
    if catalog_dir.name == ".xorq":
        return catalog_dir / "builds"
    # Otherwise use the legacy catalog-builds name
    return catalog_dir / "catalog-builds"


get_now_utc = functools.partial(datetime.now, timezone.utc)


@toolz.curry
def to_dict(self, **kwargs):
    dct = self.__getstate__()
    dct = dct | {k: v(dct[k]) for k, v in kwargs.items()}
    return dct


@toolz.curry
def from_dict(cls, dct: dict, **kwargs):
    modifications = {k: v(dct.get(k)) for k, v in kwargs.items()}
    return cls(**dct | modifications)


def convert_uuid(value):
    match value:
        case None:
            return uuid.uuid4()
        case str():
            return uuid.UUID(value)
        case uuid.UUID():
            return value
        case _:
            raise ValueError


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
            return Build.from_dict(value)
        case Build():
            return value
        case _:
            raise ValueError


@frozen
class CatalogMetadata:
    """Catalog metadata."""

    # FIXME: make uuid
    catalog_id: str = field(
        validator=instance_of(uuid.UUID), factory=uuid.uuid4, converter=convert_uuid
    )
    created_at: str = field(validator=instance_of(datetime), factory=get_now_utc)
    updated_at: str = field(validator=instance_of(datetime), factory=get_now_utc)
    tool_version: str = field(validator=instance_of(str), default=xo.__version__)

    def with_updated_timestamp(self) -> "CatalogMetadata":
        """Return new metadata with updated timestamp."""
        return self.evolve(updated_at=get_now_utc())

    evolve = evolve

    to_dict = to_dict(catalog_id=str)

    from_dict = classmethod(
        from_dict(created_at=convert_datetime, updated_at=convert_datetime)
    )


@frozen
class Build:
    """Build information."""

    build_id: Optional[str] = field(default=None, validator=optional(instance_of(str)))
    path: Optional[Path] = field(
        default=None,
        validator=optional(instance_of(Path)),
        converter=toolz.curried.excepts(Exception, Path),
    )

    evolve = evolve

    to_dict = to_dict(path=if_not_none(str))

    from_dict = classmethod(from_dict)


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

    evolve = evolve

    to_dict = to_dict(build=if_not_none(Build.to_dict))

    from_dict = classmethod(from_dict)


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

    evolve = evolve

    to_dict = to_dict(
        history=toolz.compose(tuple, functools.partial(map, Revision.to_dict))
    )

    @classmethod
    def from_dict(cls, dct):
        dct = dct.copy()
        created_at = dct.pop("created_at", None)
        obj = from_dict(
            cls,
            dct,
            history=toolz.compose(tuple, functools.partial(map, Revision.from_dict)),
        )
        if created_at is not None:
            assert datetime.fromisoformat(created_at) == min(
                rev.created_at for rev in obj.history
            )
        return obj


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

    evolve = evolve

    to_dict = to_dict

    from_dict = classmethod(from_dict)


@frozen
class XorqCatalog:
    """Xorq Catalog container."""

    aliases: Mapping[str, Alias] = field(factory=dict)
    entries: Tuple[Entry, ...] = field(
        validator=deep_iterable(instance_of(Entry), instance_of(tuple)),
        factory=tuple,
        converter=tuple,
    )
    api_version: str = field(default="xorq.dev/v1", validator=instance_of(str))
    kind: str = field(default="XorqCatalog", validator=instance_of(str))
    metadata: CatalogMetadata = field(
        factory=CatalogMetadata,
        validator=optional(instance_of(CatalogMetadata)),
    )

    def __attrs_post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, "metadata", CatalogMetadata())

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

    def maybe_get_revision_by_token(self, token: str) -> Optional[Revision]:
        """Get revision by entry@revision token if it exists."""
        t = self.resolve_target(token)
        if t is None:
            return None
        gen = (
            rev
            for entry in self.entries
            for rev in entry.history
            if entry.entry_id == t.entry_id and rev.revision_id == t.rev
        )
        return next(gen, None)

    def with_updated_metadata(self) -> "XorqCatalog":
        """Return catalog with updated timestamp."""
        return self.evolve(metadata=self.metadata.with_updated_timestamp())

    def get_entry_ids(self) -> Tuple[str, ...]:
        """Get all entry IDs."""
        return tuple(e.entry_id for e in self.entries)

    evolve = evolve

    def resolve_target(self, target: str):
        return Target.from_str(target, self)

    to_dict = to_dict(
        aliases=toolz.curried.valmap(Alias.to_dict),
        entries=toolz.compose(tuple, functools.partial(map, Entry.to_dict)),
        metadata=CatalogMetadata.to_dict,
    )

    def to_yaml(self, path):
        dct = self.to_dict()
        with path.open("wt") as fh:
            yaml.safe_dump(dct, fh, sort_keys=False)

    def save(self, path):
        catalog_path = get_catalog_path(path)
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_yaml(catalog_path)

    from_dict = classmethod(
        from_dict(
            aliases=toolz.curried.valmap(Alias.from_dict),
            entries=toolz.compose(tuple, functools.partial(map, Entry.from_dict)),
            metadata=if_not_none(CatalogMetadata.from_dict),
        )
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

    to_dict = to_dict

    def to_json_dict(self):
        data = toolz.dissoc(
            self.to_dict(),
            "node_hash",
        )
        data = data | toolz.valfilter(
            bool,
            {
                "to_unbind_hash": self.node_hash,
                "start_time": self.start_time.isoformat(),
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

    from_dict = classmethod(from_dict)

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
    revision = catalog.maybe_get_revision_by_token(token)
    if revision and revision.build and revision.build.path:
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


def get_diff_file_list(
    left_dir: Path, right_dir: Path, files: list[str] | None, all_flag: bool
) -> tuple[str, ...]:
    default = (DumpFiles.expr,)
    if files is not None:
        return tuple(files)
    if all_flag:
        default_files = (
            DumpFiles.expr,
            DumpFiles.deferred_reads,
            DumpFiles.profiles,
            DumpFiles.sql,
            DumpFiles.metadata,
        )
        sqls = {p.relative_to(left_dir).as_posix() for p in left_dir.rglob("*.sql")} | {
            p.relative_to(right_dir).as_posix() for p in right_dir.rglob("*.sql")
        }
        return default_files + tuple(sorted(sqls))
    return default


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


@frozen
class AddBuildRequest:
    build_path: Path = field(converter=Path)
    alias: Optional[str] = field(default=None)

    def __attrs_post_init__(self):
        if not self.build_path.exists():
            raise ValueError(f"Build path does not exist: {self.build_path}")


@frozen
class BuildInfo:
    """Validated build information."""

    build_id: str = field()
    meta_digest: str = field()
    source_path: Path = field()


@frozen
class CatalogPaths:
    config_path: Path = field()
    config_dir: Path = field()
    builds_dir: Path = field()

    @classmethod
    def create(cls, catalog_path: Optional[Path] = None) -> "CatalogPaths":
        config_path = get_catalog_path(catalog_path)
        config_dir = config_path.parent
        builds_dir = get_builds_dir_for_catalog(config_path)
        return cls(
            config_path=config_path, config_dir=config_dir, builds_dir=builds_dir
        )


@frozen
class AddBuildResult:
    entry_id: str = field()
    revision_id: str = field()
    build_id: str = field()


def validate_build(request: AddBuildRequest) -> BuildInfo:
    def get_meta_file(path):
        build_path = Path(path)
        meta_file = build_path / DumpFiles.metadata
        if not build_path.exists() or not build_path.is_dir():
            raise ValueError(f"Build path not found: {build_path}")
        if not meta_file.exists():
            raise ValueError(
                f"{DumpFiles.metadata} not found in build path: {build_path}"
            )
        return meta_file

    source_path = request.build_path.resolve()
    meta_file = get_meta_file(source_path)
    # The build_id is the directory name
    build_id = meta_file.parent.name
    # Compute meta_digest from metadata.json
    meta_digest = f"sha1:{file_digest(meta_file)}"

    return BuildInfo(
        build_id=build_id,
        meta_digest=meta_digest,
        source_path=source_path,
    )


def make_build_object(
    build_info: BuildInfo, builds_dir_name: str = "catalog-builds"
) -> Build:
    """Create Build object with relative path to build directory.

    Args:
        build_info: Build information
        builds_dir_name: Name of builds directory (e.g., "builds" for .xorq or "catalog-builds" for ~/.config)
    """
    build_path_str = str(Path(builds_dir_name) / build_info.build_id)
    return Build.from_dict({"build_id": build_info.build_id, "path": build_path_str})


def make_revision(
    build_info: BuildInfo,
    revision_id: str,
    timestamp: str,
    builds_dir_name: str = "catalog-builds",
) -> Revision:
    build_obj = make_build_object(build_info, builds_dir_name)
    revision_data = {
        "revision_id": revision_id,
        "created_at": timestamp,
        "build": build_obj,
        "meta_digest": build_info.meta_digest,
    }

    return Revision.from_dict(revision_data)


def maybe_find_existing_entry(catalog: XorqCatalog, alias: str) -> Optional[Entry]:
    if not alias:
        return None
    alias_obj = catalog.maybe_get_alias(alias)
    if not alias_obj:
        return None
    return catalog.maybe_get_entry(alias_obj.entry_id)


def compute_next_revision_id(entry: Entry) -> str:
    existing = [rev.revision_id for rev in entry.history]
    nums = [int(r[1:]) for r in existing if r.startswith("r") and r[1:].isdigit()]
    next_num = max(nums, default=0) + 1
    return f"r{next_num}"


def create_new_entry(
    build_info: BuildInfo, timestamp: str, builds_dir_name: str = "catalog-builds"
) -> Tuple[Entry, str]:
    entry_id = str(uuid.uuid4())
    revision_id = "r1"
    revision = make_revision(build_info, revision_id, timestamp, builds_dir_name)

    entry_data = {
        "entry_id": entry_id,
        "current_revision": revision_id,
        "history": (revision.to_dict(),),
    }
    entry = Entry.from_dict(entry_data)
    return entry, revision_id


def update_existing_entry(
    entry: Entry,
    build_info: BuildInfo,
    timestamp: str,
    builds_dir_name: str = "catalog-builds",
) -> Tuple[Entry, str]:
    revision_id = compute_next_revision_id(entry)
    revision = make_revision(build_info, revision_id, timestamp, builds_dir_name)
    updated_entry = entry.with_revision(revision)
    return updated_entry, revision_id


def update_catalog_with_entry(
    catalog: XorqCatalog,
    entry: Entry,
    revision_id: str,
    alias: Optional[str],
    timestamp: str,
) -> XorqCatalog:
    updated_catalog = catalog.with_entry(entry)

    if alias:
        alias_obj = Alias.from_dict(
            {
                "entry_id": entry.entry_id,
                "revision_id": revision_id,
                "updated_at": timestamp,
            }
        )
        updated_catalog = updated_catalog.with_alias(alias, alias_obj)

    return updated_catalog.with_updated_metadata()


def process_catalog_update(
    catalog: XorqCatalog,
    build_info: BuildInfo,
    alias: Optional[str],
    timestamp: str,
    builds_dir_name: str = "catalog-builds",
) -> Tuple[XorqCatalog, str, str]:
    existing_entry = maybe_find_existing_entry(catalog, alias)

    if existing_entry:
        entry, revision_id = update_existing_entry(
            existing_entry, build_info, timestamp, builds_dir_name
        )
    else:
        entry, revision_id = create_new_entry(build_info, timestamp, builds_dir_name)

    updated_catalog = update_catalog_with_entry(
        catalog, entry, revision_id, alias, timestamp
    )

    return updated_catalog, entry.entry_id, revision_id


def do_ensure_directories(paths: CatalogPaths) -> None:
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.builds_dir.mkdir(parents=True, exist_ok=True)


def do_copy_build_safely(build_info: BuildInfo, paths: CatalogPaths) -> None:
    target_dir = paths.builds_dir / build_info.build_id
    temp_dir = paths.builds_dir / f".{build_info.build_id}.tmp"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    try:
        shutil.copytree(build_info.source_path, temp_dir)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        os.replace(str(temp_dir), str(target_dir))
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def do_save_catalog(catalog: XorqCatalog, config_path: Path) -> None:
    catalog.save(path=config_path)


def do_print_result(result: AddBuildResult) -> None:
    print(
        f"Added build {result.build_id} as entry {result.entry_id} revision {result.revision_id}"
    )


def do_catalog_add(args) -> None:
    """Add a build to the catalog."""
    request = AddBuildRequest(build_path=args.build_path, alias=args.alias)
    namespace = getattr(args, "namespace", None)
    paths = CatalogPaths.create(catalog_path=namespace)
    timestamp = get_now_utc().isoformat()

    build_info = validate_build(request)

    do_ensure_directories(paths)
    do_copy_build_safely(build_info, paths)

    catalog = load_catalog(path=paths.config_path)
    builds_dir_name = paths.builds_dir.name
    updated_catalog, entry_id, revision_id = process_catalog_update(
        catalog, build_info, request.alias, timestamp, builds_dir_name
    )

    do_save_catalog(updated_catalog, paths.config_path)
    result = AddBuildResult(
        entry_id=entry_id, revision_id=revision_id, build_id=build_info.build_id
    )
    do_print_result(result)


def do_catalog_ls(args):
    """List entries and aliases in the catalog."""
    namespace = getattr(args, "namespace", None)
    config_path = get_catalog_path(namespace)
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
    namespace = getattr(args, "namespace", None)
    config_path = get_catalog_path(namespace)
    catalog = load_catalog(path=config_path)
    catalog_type = "project" if config_path.parent.name == ".xorq" else "user"
    print(f"Catalog path: {config_path}")
    print(f"Catalog type: {catalog_type}")
    print(f"Entries: {len(catalog.entries)}")
    print(f"Aliases: {len(catalog.aliases)}")


def do_catalog_rm(args):
    namespace = getattr(args, "namespace", None)
    config_path = get_catalog_path(namespace)
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
    namespace = getattr(args, "namespace", None)
    config_path = get_catalog_path(namespace)
    builds_dir = get_builds_dir_for_catalog(config_path)

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
    if builds_dir.exists() and builds_dir.is_dir():
        dest_builds = export_dir / builds_dir.name
        if dest_builds.exists():
            shutil.rmtree(dest_builds)
        shutil.copytree(builds_dir, dest_builds)
    print(f"Exported catalog and builds to {export_dir}")


def do_catalog_diff_builds(args):
    code = do_diff_builds(args.left, args.right, args.files, args.all)
    sys.exit(code)


def do_catalog_init(args):
    """Initialize a catalog namespace."""
    namespace = getattr(args, "namespace", None)

    if namespace:
        # Custom namespace path
        catalog_path = Path(namespace)
        if not catalog_path.name.endswith(".yaml"):
            # If namespace is a directory, create catalog.yaml inside it
            catalog_path = catalog_path / CATALOG_YAML_FILENAME
    else:
        # Default to .xorq/catalog.yaml in current directory
        catalog_path = Path.cwd() / ".xorq" / CATALOG_YAML_FILENAME

    # Check if catalog already exists
    if catalog_path.exists():
        print(f"Catalog already exists at {catalog_path}")
        return

    # Create catalog directory and builds directory
    catalog_dir = catalog_path.parent
    builds_dir = get_builds_dir_for_catalog(catalog_path)

    catalog_dir.mkdir(parents=True, exist_ok=True)
    builds_dir.mkdir(parents=True, exist_ok=True)

    # Create empty catalog
    catalog = XorqCatalog()
    catalog.save(path=catalog_path)

    catalog_type = "project" if catalog_dir.name == ".xorq" else "custom"
    print(f"Initialized {catalog_type} catalog at {catalog_path}")
    print(f"Builds directory: {builds_dir}")


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
    "init": do_catalog_init,
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
