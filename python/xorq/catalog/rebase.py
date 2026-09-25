"""Re-derive a catalog entry over its live sources, as a new entry (#2323).

The old entry is never edited or removed. Every refusal comes before the first
write: the checks read only the archive, the sweep connects through ``drift``'s
no-create guard, and nothing loads until the sweep has compared every source it
can probe. The new entry keeps the old one's wheels, requirements, and each
read's recorded posture (bundled or external); only schemas change. With a
sync, the alias checks run again after its pull, so a refusal there leaves
what the pull merged in place, unpushed.
"""

from __future__ import annotations

import sys
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from attr import evolve, field, frozen
from attr.validators import deep_iterable, in_, instance_of, optional

from xorq.catalog.catalog import CatalogEntry
from xorq.catalog.derive import Derived, add_derived, alias_targets, plan_aliases
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
    AliasRefusedError,
    PullError,
    PushError,
    RebaseError,
    RebasePushError,
    RollbackError,
)
from xorq.catalog.inspection import BuildRecord, SourceLeaf
from xorq.catalog.refresh import (
    check_refreshable,
    leaf_key,
    live_schemas,
    refresh_schemas,
)
from xorq.catalog.zip_utils import BuildZip, bundle_members, harvest_entry_from_zip
from xorq.common.exceptions import SchemaRefreshError
from xorq.ibis_yaml.enums import DumpFiles
from xorq.ibis_yaml.packager import parse_python_minor
from xorq.vendor.ibis.expr.schema import Schema


def str_tuple() -> Any:
    """An attrs field holding a tuple of ``str``, empty by default."""
    return field(
        default=(),
        converter=tuple,
        validator=deep_iterable(instance_of(str), instance_of(tuple)),
    )


@frozen
class ColumnChange:
    """One output column the rebase changed; ``None`` on the side it's absent."""

    name = field(validator=instance_of(str))
    before = field(validator=optional(instance_of(str)))
    after = field(validator=optional(instance_of(str)))

    def __str__(self) -> str:
        if self.before is None:
            return f"{self.name} {self.after} added"
        if self.after is None:
            return f"{self.name} removed"
        return f"{self.name} {self.before} -> {self.after}"


def output_changes(before: Schema, after: Schema) -> tuple[ColumnChange, ...]:
    """The columns whose type ``after`` changed, added or removed, in order.

    Types are spelled as ``format_schema`` spells them.
    """
    (old, new) = (dict(before.items()), dict(after.items()))
    return tuple(
        ColumnChange(
            name,
            None if name not in old else str(old[name]),
            None if name not in new else str(new[name]),
        )
        for name in dict.fromkeys((*old, *new))
        if old.get(name) != new.get(name)
    )


@frozen
class Rename:
    """``--rename SOURCE OLD NEW``: the expression keeps seeing ``old``, now read
    from the live column ``new``."""

    source = field(validator=instance_of(str))
    old = field(validator=instance_of(str))
    new = field(validator=instance_of(str))
    # ``new`` was itself a recorded column: it now feeds ``old``, and the
    # expression no longer sees it under its own name. Set by ``rebase_entry``.
    consumes = field(default=False, validator=instance_of(bool))

    def __str__(self) -> str:
        return f"{self.source}: {self.old} <- {self.new}"

    @property
    def warning(self) -> str | None:
        if not self.consumes:
            return None
        return (
            f"WARNING: {self.source}: {self.new} was already a recorded column; "
            f"{self.old} now reads {self.new}'s data, and the expression no "
            f"longer has a column {self.new}. Check that {self.old} was really "
            f"renamed to {self.new}"
        )


@frozen
class RebaseConflict:
    """What a conflicted rebase could not follow.

    ``op_name`` is the op that could not be rebuilt, ``None`` when a source is
    gone. ``sources`` are the sweep's own reports for the sources involved: the
    changed ones an op failed over, or the gone ones.
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
    moved_aliases = str_tuple()
    # Registered by the rebase: an `alias` that was on no entry.
    added_aliases = str_tuple()
    # The sources no probe can compare, on UNPROBED.
    unprobed = str_tuple()
    # REBASED only: false when the new entry was already cataloged, and its
    # archive kept.
    created = field(default=False, validator=instance_of(bool))
    # REBASED only: false without a sync or a remote.
    pushed = field(default=False, validator=instance_of(bool))
    # REBASED only: how the entry's output schema changed; empty if it didn't.
    output_changes = field(
        default=(),
        converter=tuple,
        validator=deep_iterable(instance_of(ColumnChange), instance_of(tuple)),
    )
    # The Python-minor check `ignore_mismatch` overrode, if any.
    venv_warning = field(default=None, validator=optional(instance_of(str)))
    # REBASED only: the renames put above their refreshed sources.
    renames = field(
        default=(),
        converter=tuple,
        validator=deep_iterable(instance_of(Rename), instance_of(tuple)),
    )
    # Set exactly when ``status`` is ``CONFLICT``.
    conflict = field(default=None, validator=optional(instance_of(RebaseConflict)))


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

    An entry whose sources are all unprobed is ``UNPROBED``: nothing can be
    refreshed, so it isn't loaded. One with only some unprobed is refused:
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


def check_python_minor(
    catalog_entry: CatalogEntry, ignore_mismatch: bool
) -> str | None:
    """Refuse an entry built on another Python minor, or on one it doesn't
    record: its UDFs are cloudpickled. Returns the warning ``ignore_mismatch``
    overrode, if any.
    """
    name = catalog_entry.name
    try:
        recorded = recorded_python_minor(catalog_entry)
    except Exception as e:
        raise RebaseError(
            f"{name} is unreadable: {format_error(e)}", RebaseExit.UNREACHABLE
        ) from e
    running = ".".join(map(str, sys.version_info[:2]))
    if recorded == tuple(sys.version_info[:2]):
        return None
    problem = (
        f"{name} records no Python minor"
        if recorded is None
        else f"{name} was built on Python {'.'.join(map(str, recorded))}"
    )
    if not ignore_mismatch:
        raise RebaseError(
            f"{problem}, this is {running}; its UDFs may not load; pass "
            "--ignore-venv-mismatch to rebase anyway",
            RebaseExit.REFUSED,
        )
    return (
        f"WARNING: {problem}; rebasing under {running}, but cloudpickled UDFs "
        "may SIGSEGV if built on a different minor"
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


def refuse_rename(catalog_entry: CatalogEntry, detail: str) -> RebaseError:
    return RebaseError(f"{catalog_entry.name}: --rename {detail}", RebaseExit.REFUSED)


def refuse_renames(catalog_entry: CatalogEntry, details: Iterable[str]) -> None:
    """Refuse every failed ``--rename`` at once, so one run shows them all."""
    match tuple(dict.fromkeys(details)):
        case ():
            return
        case (detail,):
            raise refuse_rename(catalog_entry, detail)
        case many:
            raise RebaseError(
                f"{catalog_entry.name}: --rename refused:\n"
                + "\n".join(f"  {detail}" for detail in many),
                RebaseExit.REFUSED,
            )


def quoted(names: Iterable[str]) -> str:
    """Each name quoted: a multi-path read's own name holds ``", "``."""
    return ", ".join(map(repr, names)) or "-"


def renamed_source(
    catalog_entry: CatalogEntry, record: BuildRecord, name: str
) -> dict[tuple, SourceLeaf] | str:
    """``leaf_key`` -> leaf for the one external source ``name`` is, or why
    there is none.

    One source can hold several keys: a table bound before and after it grew
    is recorded at two schemas. One name on two connections is two sources,
    and refused, as is no source at all.
    """
    try:
        by_key = {
            leaf_key(leaf, record): leaf
            for leaf in record.external_leaves
            if leaf.name == name
        }
    except Exception as e:
        # The profile the sweep would rank `unreadable`: the same exit.
        raise RebaseError(
            f"{catalog_entry.name}: {name} is unreadable: {format_error(e)}",
            RebaseExit.UNREACHABLE,
        ) from e
    if not by_key:
        sources = sorted({leaf.name for leaf in record.external_leaves})
        return f"names no source {name!r}; its sources: {quoted(sources)}"
    # `leaf_key` less its recorded schema: kind, profile, name.
    if (count := len({key[:3] for key in by_key})) > 1:
        return f"{name!r} names {count} sources (one name on different connections)"
    return by_key


def plan_renames(
    catalog_entry: CatalogEntry,
    record: BuildRecord,
    renames: tuple[Rename, ...],
    unprobed: tuple[str, ...],
) -> tuple[dict[tuple, dict[str, str]], tuple[Rename, ...]]:
    """``leaf_key`` -> recorded name -> live name, checked against the record,
    and ``renames`` with ``consumes`` set. Every failure is refused at once.

    A source recorded at several schemas is renamed under each that records
    ``old``. The live side is checked after the sweep, by
    ``check_live_renames``. A ``new`` the source also recorded is allowed and
    warned about: from the schemas alone, a column that looks unchanged can't
    be told apart from one that was dropped and replaced by the renamed one.
    """
    if renames and unprobed:
        raise refuse_rename(
            catalog_entry,
            f"needs a live schema, and no source can be probed ({quoted(unprobed)})",
        )
    planned: dict[tuple, dict[str, str]] = {}
    checked = []
    details = []
    # Per source: old -> new, and new -> old, as declared so far.
    by_old: dict[tuple[str, str], str] = {}
    by_new: dict[tuple[str, str], str] = {}
    for rename in renames:
        (source, old, new) = (rename.source, rename.old, rename.new)
        if isinstance(by_key := renamed_source(catalog_entry, record, source), str):
            details.append(by_key)
            continue
        if not (keys := [key for key, leaf in by_key.items() if old in leaf.recorded]):
            recorded = dict.fromkeys(
                name for leaf in by_key.values() for name in leaf.recorded
            )
            details.append(
                f"{source}: {old!r} is not a recorded column "
                f"(recorded: {quoted(recorded)})"
            )
            continue
        # Refused, not collapsed, even when identical: as with `git mv a a d/`
        # ("multiple sources for the same target"), a repeat is most likely a
        # copied `--rename` left unedited, and collapsing it would silently
        # drop the mapping that was meant.
        if (earlier := by_old.get((source, old))) is not None:
            details.append(
                f"{source} {old} {new} is given twice"
                if earlier == new
                else f"{source}: {old!r} is mapped to both {earlier!r} and {new!r}"
            )
            continue
        if (earlier := by_new.get((source, new))) is not None:
            details.append(
                f"{source}: {earlier!r} and {old!r} are both mapped to {new!r}"
            )
            continue
        (by_old[source, old], by_new[source, new]) = (new, old)
        for key in keys:
            planned.setdefault(key, {})[old] = new
        consumes = any(new in leaf.recorded for leaf in by_key.values())
        checked.append(evolve(rename, consumes=consumes))
    refuse_renames(catalog_entry, details)
    return planned, tuple(checked)


def check_live_renames(
    catalog_entry: CatalogEntry,
    record: BuildRecord,
    reports: tuple[LeafReport, ...],
    planned: dict[tuple, dict[str, str]],
) -> None:
    """Refuse, all at once, each rename whose new column isn't live, or whose
    old one still is and no other rename of that source moves it away (a
    chain, ``a <- x, b <- a``, or a swap).

    An unreachable or unreadable source is left to the sweep's own refusal,
    and never keyed: its profile may be what made it unreadable.
    """
    if not planned:
        return
    details = []
    for report in reports:
        if report.verdict not in (Verdict.EQUAL, Verdict.CHANGED):
            continue
        if not (by_old := planned.get(leaf_key(report.leaf, record))):
            continue
        (name, live) = (report.leaf.name, report.live)
        claimed = set(by_old.values())
        for old, new in by_old.items():
            if new not in live:
                details.append(
                    f"{name}: {new!r} is not a live column (live: {quoted(live)})"
                )
            if old in live and old not in claimed:
                details.append(
                    f"{name}: {old!r} is still a live column, so nothing was renamed"
                )
    refuse_renames(catalog_entry, details)


def rename_hint(
    record: BuildRecord,
    reports: tuple[LeafReport, ...],
    planned: dict[tuple, dict[str, str]],
) -> str:
    """The columns each changed source lost and gained, less those a
    ``--rename`` already maps; a mapping is not guessed, nor a fix promised."""
    lines = []
    for report in reports:
        if report.verdict != Verdict.CHANGED:
            continue
        by_old = planned.get(leaf_key(report.leaf, record), {})
        (recorded, live) = (report.leaf.recorded, report.live)
        if gone := [
            name for name in recorded if name not in live and name not in by_old
        ]:
            new = [
                name
                for name in live
                if name not in recorded and name not in by_old.values()
            ]
            lines.append(
                f"{report.leaf.name}: recorded columns gone: {', '.join(gone)}; "
                f"live columns new: {', '.join(new) or '-'}"
            )
    if not lines:
        return ""
    return "\n" + "\n".join(
        (
            *lines,
            "if a gone column was renamed to a new one, "
            "--rename <source> <old> <new> maps it",
        )
    )


def refuse_aliases(e: AliasRefusedError) -> RebaseError:
    return RebaseError(str(e), RebaseExit.REFUSED)


def check_only_aliases(
    catalog_entry: CatalogEntry, only_aliases: tuple[str, ...]
) -> None:
    """With no new entry, each of ``only_aliases`` must be on ``catalog_entry``."""
    try:
        plan_aliases(catalog_entry, None, False, only_aliases)
    except AliasRefusedError as e:
        raise refuse_aliases(e) from e


def add_rebased(
    catalog_entry: CatalogEntry, build_path: Path, **kwargs: object
) -> Derived:
    """``add_derived``, its failures mapped to ``RebaseError``.

    A failed push is ``RebasePushError``, raised by the caller once the result
    is known.
    """
    try:
        return add_derived(catalog_entry, build_path, **kwargs)
    except AliasRefusedError as e:
        raise refuse_aliases(e) from e
    except PullError as e:
        raise RebaseError(
            f"{catalog_entry.name}: {e}; nothing written", RebaseExit.UNREACHABLE
        ) from e
    except RollbackError as e:
        raise RebaseError(f"{catalog_entry.name}: {e}", RebaseExit.REFUSED) from e


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
    renames: Iterable[Rename | tuple[str, str, str]] = (),
) -> RebaseResult:
    """``catalog_entry`` re-derived over its live sources, as a new entry.

    No alias moves unless asked: ``move_aliases`` moves all of the old
    entry's, ``only_aliases`` just these; ``alias`` registers another.
    ``renames`` (``Rename``s, or ``(source, old, new)``) reads each recorded
    column ``old`` from the live column ``new``, keeping the recorded name. A
    no-op, a conflict, or an entry none of whose sources can be probed,
    returns ``catalog_entry`` and commits nothing; with ``renames``, a no-op
    and an unprobed entry are refused. ``entry_alias`` is the alias
    ``catalog_entry`` was named by, if any; a pull that moves or removes it
    refuses the rebase. Raises ``RebaseError`` before the rebase's first write
    (with ``sync``, an alias refusal can come after the pull), except
    ``RebasePushError`` (committed locally, push failed) and a failed
    rollback, whose message names what it left.
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
    renames = tuple(
        rename if isinstance(rename, Rename) else Rename(*rename) for rename in renames
    )
    # `ExprDumper` validates `cache_dir` as a `Path`.
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    # First: reading the record opens a `BuildZip`, which refuses a
    # bundle-less archive with a bare assertion.
    check_bundle(catalog_entry)
    record = read_entry_record(catalog_entry)
    unprobed = check_rebasable(catalog_entry, record)
    venv_warning = check_python_minor(catalog_entry, ignore_mismatch)
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
    (planned, renames) = plan_renames(catalog_entry, record, renames, unprobed)
    if unprobed:
        check_only_aliases(catalog_entry, only_aliases)
        # Nothing can be refreshed, so a load and dump could only return the
        # entry's own hash, or a new one following no drift.
        return RebaseResult(
            RebaseStatus.UNPROBED,
            catalog_entry,
            catalog_entry,
            (),
            unprobed=unprobed,
            venv_warning=venv_warning,
        )
    reports = tuple(iter_leaf_reports(record))
    # Before the no-op: a rename of an unchanged source is refused, not ignored.
    check_live_renames(catalog_entry, record, reports, planned)
    if all(report.verdict == Verdict.EQUAL for report in reports):
        check_only_aliases(catalog_entry, only_aliases)
        return RebaseResult(
            RebaseStatus.NOOP,
            catalog_entry,
            catalog_entry,
            reports,
            venv_warning=venv_warning,
        )

    def conflicted(conflict: RebaseConflict) -> RebaseResult:
        return RebaseResult(
            RebaseStatus.CONFLICT,
            catalog_entry,
            catalog_entry,
            reports,
            conflict=conflict,
            venv_warning=venv_warning,
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
            expr = refresh_schemas(loaded, live, planned)
        except SchemaRefreshError as e:
            changed = [r for r in reports if r.verdict == Verdict.CHANGED]
            detail = f"{e}{rename_hint(record, reports, planned)}"
            return conflicted(RebaseConflict(detail, changed, e.op_name))
        # `relocate_reads=False` keeps each read's recorded posture: a bundled
        # read stays bundled, an external one external.
        dumper = ExprDumper(
            expr, builds_dir=td, cache_dir=cache_dir, relocate_reads=False
        )
        if dumper.expr_hash == catalog_entry.name:
            check_only_aliases(catalog_entry, only_aliases)
            return RebaseResult(
                RebaseStatus.NOOP,
                catalog_entry,
                catalog_entry,
                reports,
                venv_warning=venv_warning,
            )
        changes = output_changes(loaded.schema(), expr.schema())
        build_path = dumper.dump_expr()
        stage_bundle(catalog_entry, build_path)

        def result(derived: Derived) -> RebaseResult:
            return RebaseResult(
                RebaseStatus.REBASED,
                catalog_entry,
                derived.new_entry,
                reports,
                derived.moved_aliases,
                added_aliases=derived.added_aliases,
                created=derived.created,
                pushed=derived.pushed,
                output_changes=changes,
                venv_warning=venv_warning,
                renames=renames,
            )

        try:
            derived = add_rebased(
                catalog_entry,
                build_path,
                alias=alias,
                move_aliases=move_aliases,
                only_aliases=only_aliases,
                sync=sync,
                entry_alias=entry_alias,
            )
        except PushError as e:
            raise RebasePushError(
                f"{catalog_entry.name}: rebased to {e.derived.new_entry.name} "
                f"locally; {e}; run `xorq catalog push`",
                RebaseExit.PUSH_FAILED,
                result(e.derived),
            ) from e
    return result(derived)
