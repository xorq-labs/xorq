"""Re-derive a catalog entry over its live sources, as a new entry (#2323).

The old entry is never edited or removed. Every refusal comes before the first
write: the checks read only the archive, the sweep connects through ``drift``'s
no-create guard, and nothing loads until the sweep has compared every source it
can probe.

The JSON document (``rebase_document``, behind ``rebase --json``, #2326) covers
every way a rebase of a resolved entry can end, buffered and emitted once::

    {
      "state": "conflict",         # a RebaseStatus or a RebaseFailure
      "exit_code": 4,              # what the command exits with
      "old_entry": "b0e5f7bd7d78",
      "new_entry": null,           # old_entry on a no-op or an attempted entry
      "build_path": null,          # always null: the build is not kept
      "failing_op": "Field",       # null unless an op could not be rebuilt
      "drift": {...},              # this entry's `check-sources --json` document
      "aliases": {"moved": []},
      "error": "XorqTypeError: ..."
    }

``state`` and ``exit_code`` come from one enum member, so the two cannot
disagree. ``drift`` is the document ``check-sources --json`` publishes under
``entries`` for this entry, built by the same code over the rebase's own sweep;
it is ``null`` on a failure, which returns no sweep. ``aliases`` is what the
rebase did to them, so it is empty unless ``state`` is ``rebased``. ``error``
is present only when there is one: a failure's message, or what a conflict's
op or source raised.
"""

from __future__ import annotations

import sys
import tempfile
import warnings
import zipfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from attr import field, frozen
from attr.validators import (
    and_,
    deep_iterable,
    in_,
    instance_of,
    max_len,
    min_len,
    optional,
)

from xorq.catalog.annex import AnnexError
from xorq.catalog.catalog import Catalog, CatalogAlias, CatalogEntry
from xorq.catalog.drift import (
    LeafReport,
    format_error,
    iter_leaf_reports,
    make_profile,
    missing_database,
    read_record,
    reports_document,
    roll_up,
    unchecked_leaves,
)
from xorq.catalog.enums import RebaseExit, RebaseFailure, RebaseStatus, Verdict
from xorq.catalog.exceptions import (
    ContentIntegrityError,
    ContentStoreError,
    RebaseError,
)
from xorq.catalog.inspection import BuildRecord
from xorq.catalog.refresh import check_refreshable, live_schemas, refresh_schemas
from xorq.catalog.zip_utils import BuildZip, bundle_members, harvest_entry_from_zip
from xorq.common.exceptions import SchemaRefreshError, UnmatchedSourceError
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
class RebaseConflict:
    """What a conflicted rebase could not follow.

    ``op_name`` is the op that could not be rebuilt, ``None`` when a source is
    gone. ``sources`` are the sweep's reports for all changed sources in the
    entry on an op failure, or the gone sources on a missing-table conflict.
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
    # The record ``reports`` were probed from.
    record = field(validator=instance_of(BuildRecord))
    moved_aliases = str_tuple()
    # `(alias, entry)`: the ``alias`` that ``-a`` took off another entry.
    taken_aliases = field(
        default=(),
        converter=tuple,
        validator=deep_iterable(
            and_(
                deep_iterable(instance_of(str), instance_of(tuple)),
                min_len(2),
                max_len(2),
            ),
            instance_of(tuple),
        ),
    )
    # Aliases the pull moved off the old entry, or removed: left as it has them.
    skipped_aliases = str_tuple()
    # The sources no probe could compare, when none could; never on REBASED.
    unprobed = str_tuple()

    # Set exactly when ``status`` is ``CONFLICT``.
    conflict = field(default=None, validator=optional(instance_of(RebaseConflict)))

    def __attrs_post_init__(self) -> None:
        if (self.status == RebaseStatus.CONFLICT) != (self.conflict is not None):
            raise ValueError("conflict must be set exactly when status is CONFLICT")

    def to_dict(self) -> dict:
        """The ``rebase --json`` document; see the module docstring."""
        conflict = self.conflict
        _, drift = reports_document(self.record, self.reports)
        document = make_rebase_document(
            self.status,
            self.old_entry.name,
            new_entry=None if conflict is not None else self.new_entry.name,
            failing_op=None if conflict is None else conflict.op_name,
            drift=drift,
            aliases={"moved": list(self.moved_aliases)},
        )
        if conflict is not None:
            document["error"] = conflict.detail
        return document


def make_rebase_document(
    state: RebaseStatus | RebaseFailure,
    old_entry: str,
    *,
    new_entry: str | None = None,
    failing_op: str | None = None,
    drift: dict | None = None,
    aliases: dict | None = None,
) -> dict:
    """The keys every ``rebase --json`` document carries, built in one place.

    Every outcome comes through here, so the key set cannot grow on one and
    not another, and ``exit_code`` is read off ``state``, the value the CLI
    exits with.
    """
    return {
        "state": str(state),
        "exit_code": int(state.exit_code),
        "old_entry": old_entry,
        "new_entry": new_entry,
        # Set by `--no-add` (#2325); until then no rebase keeps its build.
        "build_path": None,
        "failing_op": failing_op,
        "drift": drift,
        "aliases": aliases or {"moved": []},
    }


def rebase_failure(e: Exception) -> RebaseFailure:
    """The failure ``e`` ends a rebase with: a ``RebaseError`` by its code."""
    if not isinstance(e, RebaseError):
        return RebaseFailure.FAILED
    match e.exit_code:
        case RebaseExit.REFUSED:
            return RebaseFailure.REFUSED
        case RebaseExit.UNREACHABLE:
            return RebaseFailure.UNREACHABLE
        case _:
            raise ValueError(f"no failure for exit code {e.exit_code}")


def rebase_state(outcome: RebaseResult | Exception) -> RebaseStatus | RebaseFailure:
    """``outcome``'s ``state``, whose ``exit_code`` the CLI exits with."""
    if isinstance(outcome, Exception):
        return rebase_failure(outcome)
    return outcome.status


def rebase_document(
    catalog_entry: CatalogEntry, outcome: RebaseResult | Exception
) -> dict:
    """``outcome`` of rebasing ``catalog_entry``, as ``rebase --json`` prints it."""
    if not isinstance(outcome, Exception):
        return outcome.to_dict()
    document = make_rebase_document(rebase_state(outcome), catalog_entry.name)
    document["error"] = (
        str(outcome) if isinstance(outcome, RebaseError) else format_error(outcome)
    )
    return document


@contextmanager
def keeping_read_posture() -> Iterator[None]:
    """Silence the local-path warning a dump with ``relocate_reads=False`` raises.

    The warning asks for ``--relocate-reads``, which rebase deliberately does
    not do: an external read stays external so the new entry matches the old
    one's posture. Under ``--json`` it would also break the empty-stderr
    contract.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The Read op path is using a local filesystem path"
        )
        yield


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


def aliases_to_move(
    catalog_entry: CatalogEntry, move_aliases: Iterable[str] | None
) -> tuple[str, ...]:
    """``move_aliases``, or every alias for ``None``, checked against the entry."""
    aliases = tuple(catalog_alias.alias for catalog_alias in catalog_entry.aliases)
    if move_aliases is None:
        return aliases
    if unknown := sorted(set(move_aliases) - set(aliases)):
        raise RebaseError(
            f"{catalog_entry.name} has no alias {', '.join(unknown)}",
            RebaseExit.REFUSED,
        )
    return tuple(dict.fromkeys(move_aliases))


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
            gone = [r for r in reports if r.verdict == Verdict.TABLE_MISSING]
            return RebaseConflict(str(e.cause), gone)
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
) -> tuple[CatalogEntry, tuple[str, ...], tuple[tuple[str, str], ...], tuple[str, ...]]:
    """Catalog ``build_path`` and move ``moving`` onto it, all or nothing.

    Returns the new entry, the aliases moved, the ``(alias, entry)`` taken off
    another entry, and those skipped: an alias the pull no longer has on
    ``old_entry`` is not taken from where it went, unless it is ``alias``. An
    ``alias`` already on ``old_entry`` counts as moved, one on any other entry
    as taken.
    """
    catalog = old_entry.catalog
    added_aliases = (alias,) if alias else ()
    moved = []
    taken = ()
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
        # `catalog.add` moves `alias` itself.
        moving = tuple(name for name in moving if name not in (*skipped, alias))
        new_entry = catalog.add(
            build_path,
            sync=False,
            aliases=added_aliases,
            exist_ok=True,
        )
        if alias and prior[alias] == old_entry.name:
            moved.append(alias)
        elif alias and prior[alias] not in (None, new_entry.name):
            taken = ((alias, prior[alias]),)
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
    return new_entry, tuple(moved), taken, skipped


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
    ``move_aliases`` names some; ``alias`` registers another. A no-op or a
    conflict returns ``catalog_entry`` and commits nothing. Raises
    ``RebaseError`` before any catalog write; before that it may fetch the
    entry's archive into the working tree and the content cache.
    """
    from xorq.ibis_yaml.compiler import ExprDumper  # noqa: PLC0415

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
    moving = aliases_to_move(catalog_entry, move_aliases)
    reports = tuple(iter_leaf_reports(record))
    # Only a sweep that probed every source can prove a no-op on its own.
    if not unprobed and all(report.verdict == Verdict.EQUAL for report in reports):
        return RebaseResult(
            RebaseStatus.NOOP, catalog_entry, catalog_entry, reports, record
        )

    def conflicted(conflict: RebaseConflict) -> RebaseResult:
        return RebaseResult(
            RebaseStatus.CONFLICT,
            catalog_entry,
            catalog_entry,
            reports,
            record,
            conflict=conflict,
        )

    live = live_or_conflict(catalog_entry, record, reports)
    if isinstance(live, RebaseConflict):
        return conflicted(live)
    check_databases_exist(catalog_entry, record)
    with tempfile.TemporaryDirectory() as td, keeping_read_posture():
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
            status = RebaseStatus.ATTEMPTED if unprobed else RebaseStatus.NOOP
            return RebaseResult(
                status, catalog_entry, catalog_entry, reports, record, unprobed=unprobed
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
        new_entry, moved, taken, skipped = add_rebased(
            catalog_entry, build_path, alias, moving, sync
        )
    return RebaseResult(
        RebaseStatus.REBASED,
        catalog_entry,
        new_entry,
        reports,
        record,
        moved,
        taken,
        skipped,
    )
