"""Re-derive a catalog entry over its live sources, as a new entry (#2323).

The old entry is never edited or removed. Every refusal comes before the first
write: the checks read only the archive, the sweep connects through ``drift``'s
no-create guard, and nothing loads until the sweep has compared every source it
can probe. With a sync, the alias checks run again after its pull, so a refusal
there leaves what the pull merged in place, unpushed.

The aliases go through ``xorq.catalog.aliases``, shared with ``catalog pin``
and ``unpin``, which derive a new entry from an old one the same way.
"""

from __future__ import annotations

import sys
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path

from attr import field, frozen
from attr.validators import deep_iterable, in_, instance_of, optional

from xorq.catalog.aliases import AliasOutcome, AliasPlan, add_with_aliases, str_tuple
from xorq.catalog.annex import AnnexError
from xorq.catalog.catalog import CatalogEntry
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
from xorq.catalog.exceptions import (
    ContentIntegrityError,
    ContentStoreError,
    RebaseError,
)
from xorq.catalog.inspection import BuildRecord
from xorq.catalog.refresh import check_refreshable, live_schemas, refresh_schemas
from xorq.catalog.zip_utils import BuildZip, bundle_members, harvest_entry_from_zip
from xorq.common.exceptions import SchemaRefreshError, UnmatchedSourceError
from xorq.ibis_yaml.enums import DumpFiles
from xorq.ibis_yaml.packager import parse_python_minor


@frozen
class RebaseConflict:
    """What a conflicted rebase could not follow.

    ``op_name`` is the op that could not be rebuilt, ``None`` when a source is
    gone. ``sources`` are the sweep's reports for every changed source in the
    entry on an op failure, or for every source the sweep did not find equal
    (gone, changed, unreachable) on a missing-table conflict. ``detail`` is
    the cause's text: the op's error, or what refused the live schemas.
    """

    detail = field(validator=instance_of(str))
    sources = field(
        converter=tuple,
        validator=deep_iterable(instance_of(LeafReport), instance_of(tuple)),
    )
    op_name = field(default=None, validator=optional(instance_of(str)))


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
    # Empty unless REBASED: nothing moves onto an entry that already was.
    aliases = field(factory=AliasOutcome, validator=instance_of(AliasOutcome))
    # The sources no probe could compare, when none could; never on REBASED.
    unprobed = str_tuple()

    # Set exactly when ``status`` is ``CONFLICT``.
    conflict = field(default=None, validator=optional(instance_of(RebaseConflict)))


def fetch_archive(catalog_entry: CatalogEntry) -> None:
    """Make the entry's archive local, refusing with the rebase exit codes.

    A transport failure is retryable; content that arrives but does not match
    its pointer is not.
    """
    if catalog_entry.is_content_local:
        return
    try:
        catalog_entry.fetch()
    except ContentIntegrityError as e:
        raise RebaseError(
            f"{catalog_entry.name} archive is corrupt: {format_error(e)}",
            RebaseExit.REFUSED,
        ) from e
    except (ContentStoreError, AnnexError, OSError) as e:
        raise RebaseError(
            f"{catalog_entry.name} archive could not be fetched: {format_error(e)}",
            RebaseExit.UNREACHABLE,
        ) from e


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


def live_or_conflict(
    catalog_entry: CatalogEntry, record: BuildRecord, reports: tuple[LeafReport, ...]
) -> dict | RebaseConflict:
    """The live schemas to refresh over, or the conflict of a gone table."""
    try:
        return live_schemas(record, reports)
    except SchemaRefreshError as e:
        # Worst verdict wins, ranked as `check-sources` ranks it. Below
        # `table-missing`, what refused is an unreachable or unreadable source,
        # or two reads that disagree on a live schema.
        if roll_up(report.verdict for report in reports) == Verdict.TABLE_MISSING:
            # Every source the sweep flagged, so a changed or unreachable
            # neighbour of the gone table is reported in the same breath.
            involved = [r for r in reports if r.verdict != Verdict.EQUAL]
            return RebaseConflict(str(e.cause), involved)
        raise RebaseError(
            f"{catalog_entry.name}: {e.cause}", RebaseExit.UNREACHABLE
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
    or a conflict returns ``catalog_entry`` and commits nothing.
    ``entry_alias`` is the alias ``catalog_entry`` was named by, if any; a pull
    that moves or removes it refuses the rebase. Raises ``RebaseError`` before
    the rebase's first write; before that it may fetch the entry's archive into
    the working tree and the content cache. With ``sync``, an alias refusal can
    come after the pull.
    """
    from xorq.ibis_yaml.compiler import ExprDumper  # noqa: PLC0415

    # 0.4.5 took the names to move as `move_aliases`; any such call is truthy
    # and would now move every alias.
    if not isinstance(move_aliases, bool):
        raise TypeError("move_aliases is a bool; pass alias names as only_aliases")
    # A bare string would be read as one alias per character.
    if isinstance(only_aliases, str):
        raise TypeError("only_aliases is an iterable of alias names, not one name")
    # Materialized first: an empty iterator is truthy.
    only_aliases = tuple(dict.fromkeys(only_aliases))
    if move_aliases and only_aliases:
        raise ValueError("move_aliases and only_aliases are mutually exclusive")
    # `ExprDumper` validates `cache_dir` as a `Path`.
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    # Every check below opens the archive, which a pointer or annex catalog
    # may not have locally yet (a fresh clone has only the pointer).
    fetch_archive(catalog_entry)
    # Then: reading the record opens a `BuildZip`, which refuses a
    # bundle-less archive with a bare assertion.
    check_bundle(catalog_entry)
    record = read_entry_record(catalog_entry)
    unprobed = check_rebasable(catalog_entry, record)
    check_python_minor(catalog_entry, ignore_mismatch)
    # With no new name yet, only an unregistered alias is refused.
    plan = AliasPlan.read(
        catalog_entry,
        alias=alias,
        move_aliases=move_aliases,
        only_aliases=only_aliases,
        entry_alias=entry_alias,
    )
    reports = tuple(iter_leaf_reports(record))
    # Only a sweep that probed every source can prove a no-op on its own.
    if not unprobed and all(report.verdict == Verdict.EQUAL for report in reports):
        # The new entry is this one, so each of `only_aliases` must be on it;
        # `alias` is not added.
        plan.evolved(new=catalog_entry.name, alias=None)
        return RebaseResult(RebaseStatus.NOOP, catalog_entry, catalog_entry, reports)

    def conflicted(conflict: RebaseConflict) -> RebaseResult:
        return RebaseResult(
            RebaseStatus.CONFLICT,
            catalog_entry,
            catalog_entry,
            reports,
            conflict=conflict,
        )

    live = live_or_conflict(catalog_entry, record, reports)
    if isinstance(live, RebaseConflict):
        return conflicted(live)
    check_databases_exist(catalog_entry, record)
    with tempfile.TemporaryDirectory() as td:
        # Loaded inside the block: `load_expr` keeps the archive's extract dir,
        # which bundled reads point into, alive only as long as the expression.
        loaded = catalog_entry.load_expr(cache_dir=cache_dir)
        try:
            expr = refresh_schemas(loaded, live)
        except UnmatchedSourceError as e:
            raise RebaseError(
                f"{catalog_entry.name}: {e.cause}", RebaseExit.REFUSED
            ) from e
        except SchemaRefreshError as e:
            changed = [r for r in reports if r.verdict == Verdict.CHANGED]
            # The cause alone: the CLI's headline already names the op.
            return conflicted(RebaseConflict(format_error(e.cause), changed, e.op_name))
        # `relocate_reads=False` keeps each read's recorded posture: a bundled
        # read stays bundled, an external one external.
        dumper = ExprDumper(
            expr, builds_dir=td, cache_dir=cache_dir, relocate_reads=False
        )
        if dumper.expr_hash == catalog_entry.name:
            plan.evolved(new=catalog_entry.name, alias=None)
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
        # The new name is known now: an alias on another entry is refused
        # before the dump and the sync's pull, not after them.
        plan = plan.evolved(new=dumper.expr_hash)
        build_path = dumper.dump_expr()
        stage_bundle(catalog_entry, build_path)
        new_entry, aliases = add_with_aliases(plan, build_path, sync)
    return RebaseResult(
        RebaseStatus.REBASED, catalog_entry, new_entry, reports, aliases=aliases
    )
