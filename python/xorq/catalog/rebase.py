"""Re-derive a catalog entry over its live sources, as a new entry (#2323).

The old entry is never edited or removed. Every refusal comes before the first
write: the checks read only the archive, the sweep connects through ``drift``'s
no-create guard, and nothing loads until the sweep has compared every source it
can probe.
"""

from __future__ import annotations

import sys
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from attr import field, frozen
from attr.validators import deep_iterable, in_, instance_of

from xorq.catalog.catalog import Catalog, CatalogAlias, CatalogEntry
from xorq.catalog.drift import (
    LeafReport,
    format_error,
    iter_leaf_reports,
    make_profile,
    missing_database,
    read_record,
    roll_up,
    unchecked_leaves,
)
from xorq.catalog.enums import RebaseExit, RebaseStatus, Verdict
from xorq.catalog.exceptions import RebaseError
from xorq.catalog.inspection import BuildRecord
from xorq.catalog.refresh import check_refreshable, live_schemas, refresh_schemas
from xorq.catalog.zip_utils import BuildZip, harvest_entry_from_zip
from xorq.common.exceptions import SchemaRefreshError
from xorq.common.utils.logging_utils import get_logger
from xorq.ibis_yaml.enums import DumpFiles
from xorq.ibis_yaml.packager import python_minor_from_metadata_text


logger = get_logger(__name__)


def str_tuple() -> Any:
    """An attrs field holding a tuple of ``str``, empty by default."""
    return field(
        default=(),
        converter=tuple,
        validator=deep_iterable(instance_of(str), instance_of(tuple)),
    )


@frozen
class RebaseResult:
    """``new_entry`` is ``old_entry`` unless ``status`` is ``REBASED``."""

    status = field(validator=in_(tuple(RebaseStatus)))
    old_entry = field(validator=instance_of(CatalogEntry))
    new_entry = field(validator=instance_of(CatalogEntry))
    reports = field(
        converter=tuple,
        validator=deep_iterable(instance_of(LeafReport), instance_of(tuple)),
    )
    moved_aliases = str_tuple()
    # Aliases the pull moved off the old entry, or removed: left as it has them.
    skipped_aliases = str_tuple()
    # The sources no probe could compare, when none could.
    unprobed = str_tuple()


def read_entry_record(catalog_entry: CatalogEntry) -> BuildRecord:
    record = read_record(catalog_entry)
    if isinstance(record, Exception):
        raise RebaseError(
            f"{catalog_entry.name} is unreadable: {format_error(record)}",
            RebaseExit.UNREACHABLE,
        )
    return record


def check_rebasable(
    catalog_entry: CatalogEntry, record: BuildRecord
) -> tuple[str, ...]:
    """The unprobed sources, when no source can be probed; refuses otherwise.

    An entry whose sources are all unprobed is re-derived and let the hash
    decide. One with only some unprobed is refused: those would come back
    unchanged beside the refreshed ones.
    """
    name = catalog_entry.name
    if any(leaf.pinned for leaf in record.source_leaves):
        raise RebaseError(
            f"{name} is pinned, and a pin is drift-exempt; run "
            f"`xorq catalog unpin {name}` and rebase the result",
            RebaseExit.REFUSED,
        )
    unchecked = unchecked_leaves(record)
    if unchecked and len(unchecked) == len(record.external_leaves):
        return tuple(leaf.name for leaf in unchecked)
    try:
        check_refreshable(record)
    except SchemaRefreshError as e:
        raise RebaseError(
            f"{name} cannot be rebased: {e.cause}", RebaseExit.REFUSED
        ) from e
    return ()


def recorded_python_minor(catalog_entry: CatalogEntry) -> tuple[int, int] | None:
    """The ``(major, minor)`` the entry was built under, if its metadata says."""
    return BuildZip(catalog_entry.catalog_path).read_dump_file(
        DumpFiles.build_metadata, python_minor_from_metadata_text
    )


def check_python_minor(catalog_entry: CatalogEntry, ignore_mismatch: bool) -> None:
    """Refuse an entry built on another Python minor: its UDFs are cloudpickled."""
    name = catalog_entry.name
    try:
        recorded = recorded_python_minor(catalog_entry)
    except Exception as e:
        raise RebaseError(
            f"{name} is unreadable: {format_error(e)}", RebaseExit.UNREACHABLE
        ) from e
    running = tuple(sys.version_info[:2])
    if recorded not in (None, running) and not ignore_mismatch:
        raise RebaseError(
            f"{name} was built on Python {'.'.join(map(str, recorded))}, this is "
            f"{'.'.join(map(str, running))}; pass --ignore-venv-mismatch to rebase "
            "anyway",
            RebaseExit.REFUSED,
        )


def aliases_to_move(
    catalog_entry: CatalogEntry, alias: str | None, move_aliases: Iterable[str] | None
) -> tuple[str, ...]:
    """``move_aliases``, or every alias for ``None``, checked against the entry."""
    name = catalog_entry.name
    aliases = tuple(catalog_alias.alias for catalog_alias in catalog_entry.aliases)
    if move_aliases is None:
        moving = aliases
    elif unknown := sorted(set(move_aliases) - set(aliases)):
        raise RebaseError(
            f"{name} has no alias {', '.join(unknown)}", RebaseExit.REFUSED
        )
    else:
        moving = tuple(dict.fromkeys(move_aliases))
    # `catalog.add` overwrites an alias, so this one would move all the same.
    if alias in aliases and alias not in moving:
        raise RebaseError(
            f"{name} already has alias {alias}, which would move to the new "
            "entry; include it among the aliases to move, or don't request it as "
            "an extra alias",
            RebaseExit.REFUSED,
        )
    return moving


def check_databases_exist(catalog_entry: CatalogEntry, record: BuildRecord) -> None:
    """Refuse if a local database file ``record``'s profiles name is gone.

    The load connects every profile, not only the probed ones, and sqlite and
    duckdb create a missing file on connect.
    """
    try:
        paths = tuple(
            missing_database(make_profile(profile_dict))
            for profile_dict in record.profiles.values()
            if isinstance(profile_dict, dict)
        )
    except Exception as e:
        raise RebaseError(
            f"{catalog_entry.name} is unreadable: {format_error(e)}",
            RebaseExit.UNREACHABLE,
        ) from e
    if missing := [path for path in paths if path is not None]:
        raise RebaseError(
            f"{catalog_entry.name}: database {', '.join(missing)} does not exist",
            RebaseExit.UNREACHABLE,
        )


def live_or_refuse(
    catalog_entry: CatalogEntry, record: BuildRecord, reports: tuple[LeafReport, ...]
) -> dict:
    try:
        return live_schemas(record, reports)
    except SchemaRefreshError as e:
        # Worst verdict wins, ranked as `check-sources` ranks it. Below
        # `table-missing`, what refused is an unreachable or unreadable source,
        # or two reads that disagree on a live schema.
        worst = roll_up(report.verdict for report in reports)
        raise RebaseError(
            f"{catalog_entry.name}: {e.cause}",
            RebaseExit.CONFLICT
            if worst == Verdict.TABLE_MISSING
            else RebaseExit.UNREACHABLE,
        ) from e


def stage_bundle(catalog_entry: CatalogEntry, build_path: Path) -> None:
    """Copy the entry's wheels and requirements into ``build_path``.

    ``catalog.add`` then packages nothing from the caller's cwd.
    """
    name = catalog_entry.name
    try:
        with zipfile.ZipFile(catalog_entry.catalog_path) as zf:
            wheels, requirements, _ = harvest_entry_from_zip(zf, build_path)
    except Exception as e:
        raise RebaseError(
            f"{name} is unreadable: {format_error(e)}", RebaseExit.UNREACHABLE
        ) from e
    if not wheels:
        raise RebaseError(
            f"{name} carries no wheel for the rebased entry", RebaseExit.UNREACHABLE
        )
    # Without it, `catalog.add` would package the cwd's project requirements.
    if requirements is None:
        raise RebaseError(
            f"{name} carries no {DumpFiles.requirements} for the rebased entry",
            RebaseExit.UNREACHABLE,
        )
    (build_path / DumpFiles.requirements).write_bytes(requirements)


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
        logger.exception("rebase rollback failed for %s", new_entry.name)


def add_rebased(
    old_entry: CatalogEntry,
    build_path: Path,
    alias: str | None,
    moving: tuple[str, ...],
    sync: bool,
) -> tuple[CatalogEntry, tuple[str, ...], tuple[str, ...]]:
    """Catalog ``build_path`` and move ``moving`` onto it, all or nothing.

    Returns the new entry, the aliases moved, and those skipped: an alias the
    pull no longer has on ``old_entry`` is not taken from where it went, unless
    it is ``alias``.
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
        # `alias` lands on the new entry wherever the pull put it.
        skipped = tuple(
            name for name in moving if name != alias and prior[name] != old_entry.name
        )
        moving = tuple(name for name in moving if name not in skipped)
        new_entry = catalog.add(
            build_path,
            sync=False,
            aliases=added_aliases,
            exist_ok=True,
        )
        attempted = []
        try:
            for name in moving:
                # Recorded before the call: a move that fails after writing
                # the symlink must be restored too.
                attempted.append(name)
                catalog.add_alias(new_entry.name, name, sync=False)
                moved.append(name)
        except Exception:
            touched = (*added_aliases, *attempted)
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
    record = read_entry_record(catalog_entry)
    unprobed = check_rebasable(catalog_entry, record)
    check_python_minor(catalog_entry, ignore_mismatch)
    moving = aliases_to_move(catalog_entry, alias, move_aliases)
    reports = tuple(iter_leaf_reports(record))
    # Only a sweep that probed every source can prove a no-op on its own.
    if not unprobed and all(report.verdict == Verdict.EQUAL for report in reports):
        return RebaseResult(RebaseStatus.NOOP, catalog_entry, catalog_entry, reports)
    live = live_or_refuse(catalog_entry, record, reports)
    check_databases_exist(catalog_entry, record)
    with tempfile.TemporaryDirectory() as td:
        # Loaded inside the block: `load_expr` keeps the archive's extract dir,
        # which bundled reads point into, alive only as long as the expression.
        loaded = catalog_entry.load_expr(cache_dir=cache_dir)
        try:
            expr = refresh_schemas(loaded, live)
        except SchemaRefreshError as e:
            raise RebaseError(f"{catalog_entry.name}: {e}", RebaseExit.CONFLICT) from e
        # `relocate_reads=False` keeps each read's recorded posture: a bundled
        # read stays bundled, an external one external.
        dumper = ExprDumper(
            expr, builds_dir=td, cache_dir=cache_dir, relocate_reads=False
        )
        if dumper.expr_hash == catalog_entry.name:
            status = RebaseStatus.ATTEMPTED if unprobed else RebaseStatus.NOOP
            return RebaseResult(
                status, catalog_entry, catalog_entry, reports, unprobed=unprobed
            )
        build_path = dumper.dump_expr()
        stage_bundle(catalog_entry, build_path)
        new_entry, moved, skipped = add_rebased(
            catalog_entry, build_path, alias, moving, sync
        )
    return RebaseResult(
        RebaseStatus.REBASED,
        catalog_entry,
        new_entry,
        reports,
        moved,
        skipped,
        unprobed,
    )
