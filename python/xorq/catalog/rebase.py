"""Re-derive a catalog entry over its live sources, as a new entry (#2323).

The old entry is never edited or removed. Every refusal comes before the first
write: the pre-flight reads only the archive, the sweep connects through
``drift``'s no-create guard, and nothing loads until the sweep has compared
every source.
"""

from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path

from attr import field, frozen
from attr.validators import deep_iterable, in_, instance_of

from xorq.catalog.catalog import Catalog, CatalogAlias, CatalogEntry
from xorq.catalog.drift import (
    LeafReport,
    duckdb_no_create,
    format_error,
    iter_leaf_reports,
    make_profile,
    read_record,
    sqlite_no_create,
)
from xorq.catalog.enums import RebaseStatus, Verdict
from xorq.catalog.exceptions import RebaseError
from xorq.catalog.inspection import BuildRecord
from xorq.catalog.refresh import check_refreshable, live_schemas, refresh_schemas
from xorq.catalog.zip_utils import BuildZip
from xorq.common.exceptions import SchemaRefreshError
from xorq.ibis_yaml.enums import DumpFiles


# The backends that create a missing database file on connect.
NO_CREATE = {"sqlite": sqlite_no_create, "duckdb": duckdb_no_create}

WHEEL_SUFFIX = ".whl"


@frozen
class RebaseResult:
    """``new_entry`` is ``old_entry`` for a no-op."""

    status = field(validator=in_(tuple(RebaseStatus)))
    old_entry = field(validator=instance_of(CatalogEntry))
    new_entry = field(validator=instance_of(CatalogEntry))
    reports = field(
        converter=tuple,
        validator=deep_iterable(instance_of(LeafReport), instance_of(tuple)),
    )
    moved_aliases = field(
        default=(),
        converter=tuple,
        validator=deep_iterable(instance_of(str), instance_of(tuple)),
    )
    # Aliases the pull moved off the old entry, or removed: left as it has them.
    skipped_aliases = field(
        default=(),
        converter=tuple,
        validator=deep_iterable(instance_of(str), instance_of(tuple)),
    )


def recorded_python_minor(catalog_entry: CatalogEntry) -> tuple[int, int] | None:
    """The ``(major, minor)`` the entry was built under, if its metadata says."""
    metadata = BuildZip(catalog_entry.catalog_path).read_dump_file(
        DumpFiles.build_metadata, json.loads
    )
    info = metadata.get("sys-version_info") if isinstance(metadata, dict) else None
    return tuple(info[:2]) if info else None


def bundle_members(catalog_entry: CatalogEntry) -> tuple[str, ...]:
    """The archive's wheels and ``requirements.txt``."""
    with zipfile.ZipFile(catalog_entry.catalog_path) as zf:
        return tuple(
            member
            for member in zf.namelist()
            if Path(member).name.endswith(WHEEL_SUFFIX)
            or Path(member).name == DumpFiles.requirements
        )


def stage_bundle(catalog_entry: CatalogEntry, build_path: Path) -> None:
    """Copy the entry's wheels and requirements into ``build_path``.

    ``catalog.add`` then packages nothing from the caller's cwd.
    """
    with zipfile.ZipFile(catalog_entry.catalog_path) as zf:
        for member in bundle_members(catalog_entry):
            (build_path / Path(member).name).write_bytes(zf.read(member))


def missing_databases(record: BuildRecord) -> tuple[str, ...]:
    """Local database files ``record``'s profiles name that no longer exist.

    The load connects every profile, not only the probed ones, and sqlite and
    duckdb create a missing file on connect.
    """
    paths = (
        NO_CREATE[profile_dict["con_name"]](make_profile(profile_dict))[0]
        for profile_dict in record.profiles.values()
        if isinstance(profile_dict, dict) and profile_dict.get("con_name") in NO_CREATE
    )
    return tuple(path for path in paths if path is not None and not Path(path).exists())


def preflight(
    catalog_entry: CatalogEntry,
    move_aliases: Iterable[str] | None,
    ignore_mismatch: bool,
) -> tuple[BuildRecord, tuple[str, ...]]:
    """``catalog_entry``'s record and the aliases to move; reads only the archive."""
    name = catalog_entry.name
    record = read_record(catalog_entry)
    if isinstance(record, Exception):
        raise RebaseError(f"{name} is unreadable: {format_error(record)}", 2)
    if any(leaf.pinned for leaf in record.source_leaves):
        raise RebaseError(
            f"{name} is pinned, and a pin is drift-exempt; run "
            f"`xorq catalog unpin {name}` and rebase the result",
            1,
        )
    # Refused rather than rebuilt: the rewrite moves only what the sweep
    # compared, so an unprobed source would come back unchanged, as a no-op.
    try:
        check_refreshable(record)
    except SchemaRefreshError as e:
        raise RebaseError(f"{name} cannot be rebased: {e.cause}", 1) from e
    try:
        recorded = recorded_python_minor(catalog_entry)
        members = bundle_members(catalog_entry)
    except Exception as e:
        raise RebaseError(f"{name} is unreadable: {format_error(e)}", 2) from e
    running = tuple(sys.version_info[:2])
    if recorded not in (None, running) and not ignore_mismatch:
        raise RebaseError(
            f"{name} was built on Python {'.'.join(map(str, recorded))}, this is "
            f"{'.'.join(map(str, running))}; pass --ignore-venv-mismatch to rebase "
            "anyway",
            1,
        )
    aliases = tuple(catalog_alias.alias for catalog_alias in catalog_entry.aliases)
    if move_aliases is None:
        moving = aliases
    elif unknown := sorted(set(move_aliases) - set(aliases)):
        raise RebaseError(f"{name} has no alias {', '.join(unknown)}", 1)
    else:
        moving = tuple(dict.fromkeys(move_aliases))
    if not any(Path(m).name.endswith(WHEEL_SUFFIX) for m in members):
        raise RebaseError(f"{name} carries no wheel for the rebased entry", 2)
    # Without it, `catalog.add` would package the cwd's project requirements.
    if not any(Path(m).name == DumpFiles.requirements for m in members):
        raise RebaseError(
            f"{name} carries no {DumpFiles.requirements} for the rebased entry", 2
        )
    return record, moving


def sweep_proves_noop(reports: Iterable[LeafReport]) -> bool:
    """Whether the sweep alone settles it: every source equal.

    That every source was probed is ``preflight``'s ``check_refreshable``.
    """
    return all(report.verdict == Verdict.EQUAL for report in reports)


def alias_targets(catalog: Catalog, aliases: Iterable[str]) -> dict[str, str | None]:
    """The entry each of ``aliases`` points at, ``None`` for an unregistered one."""
    registered = set(catalog.list_aliases())
    return {
        alias: CatalogAlias.from_name(alias, catalog).catalog_entry.name
        if alias in registered
        else None
        for alias in aliases
    }


def roll_back(
    new_entry: CatalogEntry, prior: Mapping[str, str | None], added: bool
) -> None:
    """Remove ``new_entry`` if ``added``, and point each alias back at ``prior``.

    An alias ``prior`` maps to ``None`` did not exist, and is removed. Logged,
    not raised, so the error that caused it is the one that surfaces.
    """
    catalog = new_entry.catalog
    try:
        # First: removing the entry takes the aliases on it along.
        if added:
            catalog.remove(new_entry.name, sync=False)
        for alias, target in prior.items():
            if target is not None:
                if target != new_entry.name:
                    catalog.add_alias(target, alias, sync=False)
            elif alias in catalog.list_aliases():
                catalog.remove_alias(alias, sync=False)
    except Exception:
        from xorq.common.utils.logging_utils import get_logger  # noqa: PLC0415

        get_logger(__name__).exception("rebase rollback failed for %s", new_entry.name)


def add_rebased(
    old_entry: CatalogEntry,
    build_path: Path,
    alias: str | None,
    moving: tuple[str, ...],
    sync: bool,
) -> tuple[CatalogEntry, tuple[str, ...], tuple[str, ...]]:
    """Catalog ``build_path`` and move ``moving`` onto it, all or nothing.

    Returns the new entry, the aliases moved, and those skipped: an alias the
    pull no longer has on ``old_entry`` is not taken from where it went.
    """
    catalog = old_entry.catalog
    added_aliases = (alias,) if alias else ()
    moved = []
    with catalog.maybe_synchronizing(sync):
        # Read after the pull, which can bring in the entry or retarget an
        # alias. A rebase can land on an entry that already exists (an earlier
        # rebase of the same entry); a rollback must not remove that one.
        added = not catalog.contains(build_path.name)
        # `catalog.add` and `add_alias` overwrite an alias, so each prior
        # target is kept to restore.
        prior = alias_targets(catalog, (*added_aliases, *moving))
        skipped = tuple(name for name in moving if prior[name] != old_entry.name)
        moving = tuple(name for name in moving if name not in skipped)
        new_entry = catalog.add(
            build_path,
            sync=False,
            aliases=added_aliases,
            exist_ok=True,
        )
        try:
            for name in moving:
                catalog.add_alias(new_entry.name, name, sync=False)
                moved.append(name)
        except Exception:
            touched = (*added_aliases, *moved)
            roll_back(new_entry, {name: prior[name] for name in touched}, added)
            raise
    return new_entry, tuple(moved), skipped


def rebase_entry(
    catalog_entry: CatalogEntry,
    *,
    alias: str | None = None,
    move_aliases: Iterable[str] | None = None,
    sync: bool = True,
    ignore_mismatch: bool = False,
    cache_dir: str | Path | None = None,
) -> RebaseResult:
    """``catalog_entry`` re-derived over its live sources, as a new entry.

    Aliases on the old entry move to the new one, all of them unless
    ``move_aliases`` names some; ``alias`` registers another. A no-op returns
    ``catalog_entry`` and commits nothing. Raises ``RebaseError`` before any
    write.
    """
    from xorq.ibis_yaml.compiler import ExprDumper  # noqa: PLC0415

    # `ExprDumper` validates `cache_dir` as a `Path`.
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    record, moving = preflight(catalog_entry, move_aliases, ignore_mismatch)
    reports = tuple(iter_leaf_reports(record))
    if sweep_proves_noop(reports):
        return RebaseResult(RebaseStatus.NOOP, catalog_entry, catalog_entry, reports)
    try:
        live = live_schemas(record, reports)
    except SchemaRefreshError as e:
        raise RebaseError(f"{catalog_entry.name}: {e.cause}", 2) from e
    try:
        missing = missing_databases(record)
    except Exception as e:
        raise RebaseError(
            f"{catalog_entry.name} is unreadable: {format_error(e)}", 2
        ) from e
    if missing:
        raise RebaseError(
            f"{catalog_entry.name}: database {', '.join(missing)} does not exist", 2
        )
    with tempfile.TemporaryDirectory() as td:
        # Held until the add: the extract dir that bundled reads point into
        # lives as long as the loaded expression.
        loaded = catalog_entry.load_expr(cache_dir=cache_dir)
        try:
            expr = refresh_schemas(loaded, live)
        except SchemaRefreshError as e:
            raise RebaseError(f"{catalog_entry.name}: {e}", 2) from e
        # `relocate_reads=False` keeps each read's recorded posture: a bundled
        # read stays bundled, an external one external.
        dumper = ExprDumper(
            expr, builds_dir=td, cache_dir=cache_dir, relocate_reads=False
        )
        if dumper.expr_hash == catalog_entry.name:
            return RebaseResult(
                RebaseStatus.NOOP, catalog_entry, catalog_entry, reports
            )
        build_path = dumper.dump_expr()
        stage_bundle(catalog_entry, build_path)
        new_entry, moved, skipped = add_rebased(
            catalog_entry, build_path, alias, moving, sync
        )
    return RebaseResult(
        RebaseStatus.REBASED, catalog_entry, new_entry, reports, moved, skipped
    )
