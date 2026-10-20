"""Re-derive a catalog entry over its live sources, as a new entry (#2323).

The old entry is never edited or removed. Every refusal comes before the first
write: the checks read only the archive, the sweep connects through ``drift``'s
no-create guard, and nothing loads until the sweep has compared every source it
can probe. With a sync, the alias checks run again after its pull, so a refusal
there leaves what the pull merged in place, unpushed.

``AliasPlan`` and ``add_with_aliases`` are shared with ``catalog pin`` and
``unpin``, which derive a new entry from an old one the same way.
"""

from __future__ import annotations

import sys
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from attr import evolve, field, frozen
from attr.validators import deep_iterable, in_, instance_of, optional

from xorq.catalog.catalog import Catalog, CatalogEntry
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
    # Asked for, or seen on the old entry, and on the new entry already (a
    # pull or an earlier run put them there).
    already_aliases = str_tuple()
    # On the old entry before the sync's pull, on neither entry after it; only
    # with `move_aliases`, which takes what the old entry has once the pull is in.
    unmoved_aliases = str_tuple()
    # Still on the old entry, not asked for.
    left_aliases = str_tuple()
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


def alias_targets(catalog: Catalog) -> dict[str, str]:
    """Every registered alias and the entry it points at, in one read."""
    return {
        catalog_alias.alias: catalog_alias.catalog_entry.name
        for catalog_alias in catalog.catalog_aliases
    }


@frozen
class AliasPlan:
    """The aliases a new entry derived from ``old_entry`` takes, over one read of them.

    ``move_aliases`` takes every alias ``old_entry`` has, ``only_aliases`` just
    these, each of which must be on it; ``alias`` is registered besides, and
    must be unregistered or on ``old_entry``: one on another entry is refused,
    not taken. One already on ``new`` is where it was asked to be. Until the
    derivation knows its name, ``new`` is ``None`` and only an unregistered
    name is refused: an alias elsewhere may be on the new entry (a rerun
    after a failed push). ``entry_alias`` is the alias ``old_entry`` was named
    by, if any; the sync's pull moving it off ``old_entry`` is refused, since
    the name no longer means this entry.

    ``targets`` is the read: each registered alias and the entry it points at.
    """

    old_entry = field(validator=instance_of(CatalogEntry))
    targets = field(converter=dict, validator=instance_of(dict))
    new = field(default=None, validator=optional(instance_of(str)))
    alias = field(default=None, validator=optional(instance_of(str)))
    move_aliases = field(default=False, validator=instance_of(bool))
    only_aliases = str_tuple()
    entry_alias = field(default=None, validator=optional(instance_of(str)))

    @classmethod
    def read(cls, old_entry: CatalogEntry, **kwargs: Any) -> AliasPlan:
        """The plan over the aliases as they are now, checked."""
        return cls(old_entry, alias_targets(old_entry.catalog), **kwargs).checked()

    def evolved(self, **changes: Any) -> AliasPlan:
        """``self`` with ``changes``, checked; ``targets`` to re-read after a pull."""
        return evolve(self, **changes).checked()

    @property
    def old(self) -> str:
        return self.old_entry.name

    @property
    def here(self) -> frozenset[str]:
        return frozenset({self.old, self.new} - {None})

    def target(self, name: str) -> str | None:
        """The entry ``name`` points at, ``None`` for an unregistered one."""
        return self.targets.get(name)

    @property
    def on_old(self) -> tuple[str, ...]:
        return tuple(
            sorted(name for name, target in self.targets.items() if target == self.old)
        )

    @property
    def asked(self) -> tuple[str, ...]:
        """The aliases to end up on ``new``, ``alias`` first."""
        named = self.on_old if self.move_aliases else self.only_aliases
        return tuple(dict.fromkeys((*((self.alias,) if self.alias else ()), *named)))

    def checked(self) -> AliasPlan:
        """``self``; raises ``RebaseError`` for an alias the request can't have."""
        if unknown := tuple(n for n in self.only_aliases if self.target(n) is None):
            raise RebaseError(
                f"{self.old} has no alias {', '.join(sorted(unknown))}",
                RebaseExit.REFUSED,
            )
        if self.new is None:
            return self
        here = self.here
        if elsewhere := tuple(
            n for n in self.only_aliases if self.target(n) not in here
        ):
            # Says where it went: the pull can move one.
            raise RebaseError(
                "; ".join(
                    f"alias {n!r} points at {self.target(n)}, not {self.old}"
                    for n in elsewhere
                ),
                RebaseExit.REFUSED,
            )
        if self.alias and self.target(self.alias) not in {None, *here}:
            # `add-alias` moves it in one commit; removing it first would leave
            # it resolving to nothing until the new entry lands.
            raise RebaseError(
                f"alias {self.alias!r} points at {self.target(self.alias)}; pick "
                "another name, or leave it off and then run "
                f"`xorq catalog add-alias {self.new} {self.alias}`",
                RebaseExit.REFUSED,
            )
        if self.entry_alias and self.target(self.entry_alias) not in here:
            now = self.target(self.entry_alias)
            raise RebaseError(
                f"alias {self.entry_alias!r} points at {now or 'nothing'} after "
                f"the pull, not {self.old}; rerun with {self.old}",
                RebaseExit.REFUSED,
            )
        return self

    @property
    def moving(self) -> tuple[str, ...]:
        """To point at ``new`` after the add, which registers ``alias`` itself."""
        return tuple(
            n for n in self.asked if n != self.alias and self.target(n) != self.new
        )

    @property
    def moved(self) -> tuple[str, ...]:
        return tuple(n for n in self.asked if self.target(n) == self.old)

    @property
    def added(self) -> tuple[str, ...]:
        return (self.alias,) if self.alias and self.target(self.alias) is None else ()

    @property
    def already(self) -> tuple[str, ...]:
        return tuple(n for n in self.asked if self.target(n) == self.new)

    @property
    def left(self) -> tuple[str, ...]:
        return tuple(n for n in self.on_old if n not in self.asked)

    def pulled_away(self, before: AliasPlan) -> tuple[str, ...]:
        """Of the aliases ``before`` saw on the old entry, the ones not asked now.

        With ``move_aliases``, the pull between the two plans took them off it.
        """
        return tuple(n for n in before.on_old if n not in self.asked)


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


def roll_back(
    catalog: Catalog, name: str, prior: Mapping[str, str | None], added: bool
) -> None:
    """Remove entry ``name`` if ``added``, and point each alias back at ``prior``.

    An alias ``prior`` maps to ``None`` did not exist, and is removed. Logged,
    not raised, so the error that caused it is the one that surfaces.
    """
    try:
        # First: removing the entry takes the aliases on it along. An add that
        # failed part-way left some of the entry's pieces; `remove` takes
        # whichever are there.
        pieces = CatalogEntry(name, catalog, require_exists=False)._exists_components  # noqa: SLF001  # xorq-style: disable=protected-access
        if added and any(pieces.values()):
            catalog.remove(name, sync=False)
        for alias, target in prior.items():
            if target is not None:
                if target != name:
                    catalog.add_alias(target, alias, sync=False)
            elif alias in catalog.list_aliases():
                catalog.remove_alias(alias, sync=False)
    except Exception:
        logger.exception("rollback failed for %s", name)


def add_with_aliases(
    plan: AliasPlan, build_path: Path, sync: bool
) -> tuple[CatalogEntry, AliasPlan]:
    """Catalog ``build_path`` and point ``plan``'s aliases at it, all or nothing.

    Returns the new entry and the plan as it stood after the sync's pull, which
    can move an alias: what it moved, added and left is the plan's.
    """
    catalog = plan.old_entry.catalog
    name = build_path.name
    with catalog.maybe_synchronizing(sync):
        # Planned again after the pull: a refusal here still comes before the
        # first write of our own, though after whatever the pull merged.
        plan = plan.evolved(new=name, targets=alias_targets(catalog))
        # Read after the pull, which can bring in the entry (an earlier
        # derivation of the same one); a rollback must not remove that one.
        added = not catalog.contains(name)
        # `catalog.add` and `add_alias` overwrite an alias, so each is recorded
        # before the call that writes it, for its prior target to be restored.
        registered = (plan.alias,) if plan.alias else ()
        attempted = [*registered]
        try:
            new_entry = catalog.add(
                build_path, sync=False, aliases=registered, exist_ok=True
            )
            for alias in plan.moving:
                attempted.append(alias)
                catalog.add_alias(name, alias, sync=False)
        except Exception:
            roll_back(catalog, name, {n: plan.target(n) for n in attempted}, added)
            raise
    return new_entry, plan


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
    # A bare string would be read as one alias per character.
    if isinstance(only_aliases, str):
        raise TypeError("only_aliases is an iterable of alias names, not one name")
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
        new_entry, landed = add_with_aliases(plan, build_path, sync)
    # `move_aliases` takes what the old entry has after the pull; what the user
    # saw on it before is accounted for too.
    pulled = landed.pulled_away(plan) if move_aliases else ()
    return RebaseResult(
        RebaseStatus.REBASED,
        catalog_entry,
        new_entry,
        reports,
        moved_aliases=landed.moved,
        added_aliases=landed.added,
        already_aliases=(
            *landed.already,
            *(n for n in pulled if landed.target(n) == landed.new),
        ),
        unmoved_aliases=tuple(n for n in pulled if landed.target(n) != landed.new),
        left_aliases=landed.left,
    )
