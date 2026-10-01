"""Re-derive a catalog entry over its live sources, as a new entry (#2323).

The old entry is never edited or removed. Every refusal comes before the first
write: the checks read only the archive, the sweep connects through ``drift``'s
no-create guard, and nothing loads until the sweep has compared every source it
can probe. With a sync, the alias checks run again after its pull, so a refusal
there leaves what the pull merged in place, unpushed.
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
from xorq.catalog.zip_utils import BuildZip, bundle_members, harvest_entry_from_zip
from xorq.common.exceptions import SchemaRefreshError
from xorq.common.utils.logging_utils import get_logger
from xorq.ibis_yaml.enums import DumpFiles
from xorq.ibis_yaml.packager import parse_python_minor


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
    # Registered by the rebase: an `alias` that was on no entry.
    added_aliases = str_tuple()
    # The sources no probe could compare, when none could; never on REBASED.
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

    An entry whose sources are all unprobed is re-derived, and the hash
    decides: the same hash is ``ATTEMPTED``, a new one is refused, since no
    schema was refreshed to follow it. One with only some unprobed is refused:
    those would come back unchanged beside the refreshed ones.
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
    """The ``(major, minor)`` the entry was built under, if its metadata says.

    Raises on malformed metadata, which is refused rather than read as none.
    """
    return BuildZip(catalog_entry.catalog_path).read_dump_file(
        DumpFiles.build_metadata, parse_python_minor
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


def check_bundle(catalog_entry: CatalogEntry) -> None:
    """Refuse an archive that lacks the wheels or requirements to re-add."""
    name = catalog_entry.name
    try:
        with zipfile.ZipFile(catalog_entry.catalog_path) as zf:
            wheels, requirements, _ = bundle_members(zf)
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


def plan_aliases(
    old_entry: CatalogEntry,
    alias: str | None,
    move_aliases: bool,
    only_aliases: tuple[str, ...],
    new_name: str | None = None,
) -> tuple[tuple[str, ...], dict[str, str | None]]:
    """The aliases to move onto the new entry; refuses any the request can't have.

    ``move_aliases`` moves every alias ``old_entry`` has now, ``only_aliases``
    just these, each of which must be on it. ``alias`` must be unregistered or
    on ``old_entry``: one on another entry is refused, not taken. An alias
    already on ``new_name`` is where it was asked to be. Run after the sync's
    pull, which can move an alias, or with no ``new_name`` when there is
    nothing to rebase.

    Returned with them: the entry each of them and ``alias`` points at now
    (``None`` for an unregistered one), for a rollback to restore.
    """
    here = {old_entry.name, new_name} - {None}
    added = (alias,) if alias else ()
    targets = alias_targets(old_entry.catalog, (*only_aliases, *added))
    if lacking := [name for name in only_aliases if targets[name] not in here]:
        raise RebaseError(
            f"{old_entry.name} has no alias {', '.join(sorted(lacking))}",
            RebaseExit.REFUSED,
        )
    if alias and targets[alias] not in {None, *here}:
        # `add-alias` moves it in one commit; removing it first would leave
        # it resolving to nothing until a later rebase lands.
        raise RebaseError(
            f"alias {alias!r} points at {targets[alias]}; pick another name, "
            "or rebase without it and then run "
            f"`xorq catalog add-alias {new_name} {alias}`",
            RebaseExit.REFUSED,
        )
    if move_aliases:
        moving = tuple(catalog_alias.alias for catalog_alias in old_entry.aliases)
        # Each was found on `old_entry`; no need to read it again.
        return moving, {**targets, **dict.fromkeys(moving, old_entry.name)}
    return only_aliases, targets


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
    ``check_bundle`` has made sure the archive carries both.
    """
    name = catalog_entry.name
    try:
        with zipfile.ZipFile(catalog_entry.catalog_path) as zf:
            _, requirements, _ = harvest_entry_from_zip(zf, build_path)
    except Exception as e:
        raise RebaseError(
            f"{name} is unreadable: {format_error(e)}", RebaseExit.UNREACHABLE
        ) from e
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


def check_entry_alias(old_entry: CatalogEntry, entry_alias: str | None) -> None:
    """Refuse if the pull took ``entry_alias``, which named ``old_entry``, off it.

    The rebase would otherwise re-derive an entry the name no longer means.
    """
    if not entry_alias:
        return
    old = old_entry.name
    now = alias_targets(old_entry.catalog, (entry_alias,))[entry_alias]
    if now != old:
        raise RebaseError(
            f"alias {entry_alias!r} points at {now or 'nothing'} after the pull, "
            f"not {old}; rerun with {old} to rebase {old}",
            RebaseExit.REFUSED,
        )


def add_rebased(
    old_entry: CatalogEntry,
    build_path: Path,
    alias: str | None,
    move_aliases: bool,
    only_aliases: tuple[str, ...],
    sync: bool,
    entry_alias: str | None = None,
) -> tuple[CatalogEntry, tuple[str, ...], tuple[str, ...]]:
    """Catalog ``build_path`` and move the requested aliases onto it, all or nothing.

    Returns the new entry, the aliases moved (an ``alias`` already on
    ``old_entry`` counts as moved) and the ones added (an ``alias`` on no
    entry).
    """
    catalog = old_entry.catalog
    added_aliases = (alias,) if alias else ()
    with catalog.maybe_synchronizing(sync):
        # Planned again after the pull, which can move an alias: a refusal
        # here still comes before the rebase's first write, though after
        # whatever the pull merged. `catalog.add` and `add_alias` overwrite
        # an alias, so each prior target is kept to restore.
        check_entry_alias(old_entry, entry_alias)
        moving, prior = plan_aliases(
            old_entry, alias, move_aliases, only_aliases, build_path.name
        )
        # Read after the pull, which can bring in the entry. A rebase can land
        # on an entry that already exists (an earlier rebase of the same
        # entry); a rollback must not remove that one.
        added = not catalog.contains(build_path.name)
        # `catalog.add` moves `alias` itself; one already on the new entry stays.
        moving = tuple(
            name for name in moving if name != alias and prior[name] != build_path.name
        )
        new_entry = catalog.add(
            build_path,
            sync=False,
            aliases=added_aliases,
            exist_ok=True,
        )
        moved = [alias] if alias and prior[alias] == old_entry.name else []
        added_alias = (alias,) if alias and prior[alias] is None else ()
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
    return new_entry, tuple(moved), added_alias


def rebase_entry(
    catalog_entry: CatalogEntry,
    *,
    alias: str | None = None,
    move_aliases: bool = False,
    only_aliases: Iterable[str] = (),
    sync: bool = True,
    ignore_mismatch: bool = False,
    cache_dir: str | Path | None = None,
    entry_alias: str | None = None,
) -> RebaseResult:
    """``catalog_entry`` re-derived over its live sources, as a new entry.

    No alias moves unless asked: ``move_aliases`` moves all of the old
    entry's, ``only_aliases`` just these; ``alias`` registers another. A no-op
    returns ``catalog_entry`` and commits nothing. ``entry_alias`` is the
    alias ``catalog_entry`` was named by, if any; a pull that moves or removes
    it refuses the rebase. Raises ``RebaseError``
    before the rebase's first write; with ``sync``, an alias refusal can come
    after the pull.
    """
    from xorq.ibis_yaml.compiler import ExprDumper  # noqa: PLC0415

    # 0.4.5 took the names to move as `move_aliases`; any such call is truthy
    # and would now move every alias.
    if not isinstance(move_aliases, bool):
        raise TypeError("move_aliases is a bool; pass alias names as only_aliases")
    # Materialized first: an empty iterator is truthy.
    only_aliases = tuple(dict.fromkeys(only_aliases))
    if move_aliases and only_aliases:
        raise ValueError("move_aliases and only_aliases are mutually exclusive")
    # `ExprDumper` validates `cache_dir` as a `Path`.
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    # First: reading the record opens a `BuildZip`, which refuses a
    # bundle-less archive with a bare assertion.
    check_bundle(catalog_entry)
    record = read_entry_record(catalog_entry)
    unprobed = check_rebasable(catalog_entry, record)
    check_python_minor(catalog_entry, ignore_mismatch)
    # Only an unregistered name is sure to be refused this early: an alias
    # elsewhere may be on the new entry (a rerun after a failed push), whose
    # name only the dump knows.
    if unknown := sorted(
        name
        for name, target in alias_targets(catalog_entry.catalog, only_aliases).items()
        if target is None
    ):
        raise RebaseError(
            f"{catalog_entry.name} has no alias {', '.join(unknown)}",
            RebaseExit.REFUSED,
        )
    reports = tuple(iter_leaf_reports(record))
    # Only a sweep that probed every source can prove a no-op on its own.
    if not unprobed and all(report.verdict == Verdict.EQUAL for report in reports):
        # No new entry, so each of `only_aliases` must be on this one.
        plan_aliases(catalog_entry, None, False, only_aliases)
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
            plan_aliases(catalog_entry, None, False, only_aliases)
            status = RebaseStatus.ATTEMPTED if unprobed else RebaseStatus.NOOP
            return RebaseResult(
                status, catalog_entry, catalog_entry, reports, unprobed=unprobed
            )
        if unprobed:
            # Nothing was refreshed, so the new hash follows no drift (the
            # build hash reads a source's path, not its contents); a new entry
            # would carry the old schemas under a rebased name.
            raise RebaseError(
                f"{catalog_entry.name}: re-derived to a new hash, but no source "
                f"could be probed ({', '.join(unprobed)}), so nothing was "
                "refreshed",
                RebaseExit.REFUSED,
            )
        build_path = dumper.dump_expr()
        stage_bundle(catalog_entry, build_path)
        new_entry, moved, added = add_rebased(
            catalog_entry,
            build_path,
            alias,
            move_aliases,
            only_aliases,
            sync,
            entry_alias,
        )
    return RebaseResult(
        RebaseStatus.REBASED,
        catalog_entry,
        new_entry,
        reports,
        moved,
        added_aliases=added,
    )
