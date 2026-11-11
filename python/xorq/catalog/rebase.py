"""Re-derive a catalog entry over its live sources, as a new entry (#2323).

The old entry is never edited or removed. Every refusal comes before the first
write: the checks read only the archive, the sweep connects through ``drift``'s
no-create guard, and nothing loads until the sweep has compared every source it
can probe. With a sync, the alias checks run again after its pull, so a refusal
there leaves what the pull merged in place; whatever stops a rebase says if the
local catalog then has commits its remote lacks. A pull that fails is
retryable and leaves nothing written; a push that fails after the write is not
an error but its own outcome, ``PUSH_FAILED``: the rebase is committed locally.

The aliases go through ``xorq.catalog.aliases``, shared with ``catalog pin``
and ``unpin``, which derive a new entry from an old one the same way.

The JSON document (``rebase_document``, behind ``rebase --json``) covers every
way a rebase of a resolved entry can end, buffered and emitted once::

    {
      "state": "conflict",         # a RebaseStatus or a RebaseFailure
      "exit_code": 4,              # what the command exits with
      "old_entry": "b0e5f7bd7d78",
      "new_entry": null,           # old_entry on a no-op; null unless cataloged
      "new_hash": null,            # the rebased build's hash, cataloged or not
      "created": null,             # on a cataloged rebase: false if it already was
      "unpushed": null,            # on a failure: whether there is a commit to push
      "output_changes": null,      # once derived: [{"name", "before", "after"}]
      "renames": null,             # once derived: [{"source", "old", "new"}]
      "swaps": null,               # once derived: [{"old": {"name", "profile"}, "new": {...}}]
      "build_path": null,          # where `--no-add` wrote the build
      "failing_op": "Field",       # null unless an op could not be rebuilt
      "drift": {...},              # this entry's `check-sources --json` document
      "aliases": {"moved": [], "added": [], "already": [], "unmoved": [], "left": []},
      "warnings": [],              # what the rebase warned, instead of stderr
      "error": "XorqTypeError: ..."
    }

``state`` and ``exit_code`` come from one enum member, so the two cannot
disagree. ``drift`` is the document ``check-sources --json`` publishes under
``entries`` for this entry, built by the same code over the rebase's own sweep;
a refusal after the sweep carries it too, so it is ``null`` only when nothing
was swept or the rebase failed rather than refused. ``aliases`` is what the
rebase did to them, as stderr reports it, so it is empty unless a new entry was
written (``rebased`` or ``push-failed``), or on ``preview``, where it is what
the rebase would do (before the sync's pull, which a preview does not run).
``new_hash`` is the hash the rebase derived: ``new_entry`` once cataloged, the
hash ``--dry-run`` (``preview``) or ``--no-add`` (``built``) did not catalog,
or the one a refusal came after once the rebase derived it, and ``null``
otherwise. ``created`` is set only once an entry is cataloged (``rebased``,
``push-failed``): ``false`` when it already was, from an earlier rebase, and
its archive was kept. ``unpushed`` is set only on a failure (any
``RebaseFailure``) of a catalog with a remote: ``true`` when the local catalog
has a commit the remote lacks, as last fetched, such as a merge the sync's pull
left before a refusal; `xorq catalog sync` pushes it. ``output_changes`` is set
once a rebase is derived (``rebased``, ``push-failed``, ``preview``,
``built``): each output column it retyped, added or removed, ``before`` or
``after`` ``null`` where the column is absent, and empty when the output schema
did not change. ``renames`` is set when ``output_changes`` is: each
``--rename`` the rebase put above its source, empty when none was given.
``swaps`` is set when ``output_changes`` is: each source a ``--swap``
re-pointed, by name and profile as recorded (``old``) and as now read
(``new``), empty when none was given; the new entry's
``build_metadata.json`` records the same list as ``swapped_sources``.
``build_path`` is set only on ``built``. ``error`` is present only when there
is one, always as ``Type: message``, then the stderr hint, if any, on a line
of its own: what refused or failed the rebase, what a
conflict's op or source raised, or what a failed push raised. ``warnings``
holds each warning the rebase raised, as ``Category: message``, so stderr stays
empty; a ``--rename`` that reads from a column its source recorded too raises a
``RenameWarning``. On ``push-failed`` the new entry and its aliases are
committed locally, and `xorq catalog sync` publishes them.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
import tempfile
import warnings
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from attr import Factory, asdict, converters, evolve, field, frozen
from attr.validators import deep_iterable, in_, instance_of, optional
from git.exc import GitCommandError

from xorq.catalog.aliases import AliasOutcome, AliasPlan, add_with_aliases, str_tuple
from xorq.catalog.annex import AnnexError
from xorq.catalog.catalog import (
    Catalog,
    CatalogEntry,
    CatalogMergeConflict,
    CatalogPullError,
    SourceAt,
    SwappedSource,
)
from xorq.catalog.constants import REBASED_FROM, SWAPPED_SOURCES
from xorq.catalog.drift import (
    LeafReport,
    checkable_leaves,
    format_error,
    iter_leaf_reports,
    make_profile,
    missing_database,
    read_record,
    reports_document,
    roll_up,
    sqlite_no_create,
    state_and_code,
    unchecked_leaves,
)
from xorq.catalog.enums import LeafKind, RebaseFailure, RebaseStatus, Verdict
from xorq.catalog.exceptions import (
    CatalogPushError,
    ContentIntegrityError,
    ContentStoreError,
    PythonMinorWarning,
    RebaseError,
    RebaseOptionsError,
    RenameWarning,
    RollbackError,
)
from xorq.catalog.inspection import BuildRecord, SourceLeaf, table_name
from xorq.catalog.refresh import (
    LeafKey,
    ambiguous_renames,
    check_refreshable,
    leaf_key,
    live_schemas,
    refresh_schemas,
    repoint_tables,
)
from xorq.catalog.zip_utils import BuildZip, bundle_members, harvest_entry_from_zip
from xorq.common.exceptions import (
    LocalReadPathWarning,
    SchemaRefreshError,
    UnmatchedSourceError,
)
from xorq.common.utils.env_utils import maybe_substitute_env_vars
from xorq.common.utils.graph_utils import find_all_sources
from xorq.common.utils.node_utils import update_read_kwargs
from xorq.ibis_yaml.enums import DumpFiles, ReadKwarg
from xorq.ibis_yaml.packager import parse_python_minor
from xorq.vendor.ibis.backends.profiles import Profile, Profiles


if TYPE_CHECKING:
    from xorq.vendor.ibis.backends import BaseBackend
    from xorq.vendor.ibis.expr.schema import Schema
    from xorq.vendor.ibis.expr.types import Expr


def alias_names(only_aliases: Iterable[str]) -> tuple[str, ...]:
    # A bare string would be read as one alias per character.
    if isinstance(only_aliases, str):
        raise TypeError("only_aliases is an iterable of alias names, not one name")
    # Materialized first: an empty iterator is truthy.
    return tuple(dict.fromkeys(only_aliases))


def check_move_aliases(instance: object, attribute: object, value: object) -> None:
    # 0.4.5 took the names to move as `move_aliases`; any such call is truthy
    # and would now move every alias.
    if not isinstance(value, bool):
        raise TypeError("move_aliases is a bool; pass alias names as only_aliases")


@frozen
class Rename:
    """``--rename SOURCE OLD NEW``: the expression keeps seeing ``old``, now read
    from the live column ``new``."""

    source = field(validator=instance_of(str))
    old = field(validator=instance_of(str))
    new = field(validator=instance_of(str))
    # ``new`` was itself a recorded column, and no other rename of the source
    # (a chain, ``a <- x, b <- a``, or a swap) maps it onward: it now feeds
    # ``old``, and the expression no longer sees it under its own name. Set
    # by ``rebase_entry``, and derived, so not part of a rename's identity.
    consumes = field(default=False, eq=False, validator=instance_of(bool))

    def __str__(self) -> str:
        return f"{self.source}: {self.old} <- {self.new}"

    @property
    def warning(self) -> str | None:
        if not self.consumes:
            return None
        return (
            f"{self.source}: {self.new} was already a recorded column; "
            f"{self.old} now reads {self.new}'s data, and the expression no "
            f"longer has a column {self.new}. "
            f"Check that {self.old} was really renamed to {self.new}"
        )

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "old": self.old, "new": self.new}


def rename_tuple(renames: Iterable[Rename | tuple[str, str, str]]) -> tuple:
    # A bare string would be read as one rename per character.
    if isinstance(renames, str):
        raise TypeError("renames is an iterable of (source, old, new), not a string")
    return tuple(
        rename if isinstance(rename, Rename) else Rename(*rename) for rename in renames
    )


def repeated_renames(renames: tuple[Rename, ...]) -> Iterator[str]:
    """Why each of ``renames`` that repeats a column, or renames one to itself,
    is refused.

    Refused, not collapsed, even when identical: as with `git mv a a d/`
    ("multiple sources for the same target"), a repeat is most likely a copied
    `--rename` left unedited, and collapsing it would silently drop the
    mapping that was meant.
    """
    # Per source: old -> new, and new -> old, as declared so far.
    by_old: dict[tuple[str, str], str] = {}
    by_new: dict[tuple[str, str], str] = {}
    for rename in renames:
        (source, old, new) = (rename.source, rename.old, rename.new)
        if old == new:
            yield f"{source} {old} {new} renames {old!r} to itself"
        elif (earlier := by_old.get((source, old))) is not None:
            yield (
                f"{source} {old} {new} is given twice"
                if earlier == new
                else f"{source}: {old!r} is mapped to both {earlier!r} and {new!r}"
            )
        elif (earlier := by_new.get((source, new))) is not None:
            yield f"{source}: {earlier!r} and {old!r} are both mapped to {new!r}"
        else:
            (by_old[source, old], by_new[source, new]) = (new, old)


def rename_refusal(details: Iterable[str]) -> str | None:
    """One message for every refused ``--rename``, so one run shows them all."""
    match tuple(dict.fromkeys(details)):
        case ():
            return None
        case (detail,):
            return f"--rename {detail}"
        case many:
            return "--rename refused:\n" + "\n".join(f"  {d}" for d in many)


@frozen
class Swap:
    """``--swap OLD NEW``: the source recorded as ``old`` now reads ``new``
    on the same connection: a table beside it in its namespace, or, for an
    external read, the one path ``new`` by the same method and kwargs."""

    old = field(validator=instance_of(str))
    new = field(validator=instance_of(str))

    def __str__(self) -> str:
        return f"{self.old} {self.new}"


def swap_tuple(swaps: Iterable[Swap | tuple[str, str]]) -> tuple:
    # A bare string would be read as one swap per character.
    if isinstance(swaps, str):
        raise TypeError("swaps is an iterable of (old, new), not a string")
    return tuple(swap if isinstance(swap, Swap) else Swap(*swap) for swap in swaps)


def repeated_swaps(swaps: tuple[Swap, ...], moved: bool = False) -> Iterator[str]:
    """Why each of ``swaps`` that swaps a source to itself, or twice, or to no
    table, is refused. ``moved``: onto another connection, its own name is
    a swap."""
    yield from (f"--swap {swap.old} names no table" for swap in swaps if not swap.new)
    yield from (
        f"--swap {swap} swaps nothing"
        for swap in swaps
        if swap.old == swap.new and not moved
    )
    counts = Counter(swap.old for swap in swaps)
    yield from (f"--swap {old} is given twice" for old, n in counts.items() if n > 1)


def stored_profile(name: str) -> Profile:
    """The stored profile ``name``, refused before any write if there is none."""
    try:
        return Profiles().get(name)
    except FileNotFoundError as e:
        raise RebaseOptionsError(f"--profile {name} names no stored profile") from e


def builds_path(builds_dir: str | Path) -> Path:
    # `Path("")` is `.`, which would write the build beside the caller's files.
    if not str(builds_dir):
        raise RebaseOptionsError("--builds-dir names no directory")
    return Path(builds_dir)


@frozen
class RebaseOptions:
    """What a rebase is asked to do, refused if it contradicts itself.

    The one place the option rules live: the CLI builds it before resolving
    the entry and reports ``RebaseOptionsError`` as a usage error, and
    ``rebase_entry`` builds it from its own arguments.
    """

    alias = field(default=None, validator=optional(instance_of(str)))
    move_aliases = field(default=False, validator=check_move_aliases)
    only_aliases = field(default=(), converter=alias_names)
    dry_run = field(default=False, validator=instance_of(bool))
    add = field(default=True, validator=instance_of(bool))
    # Only with `add=False`, which writes there, `builds` by default.
    builds_dir = field(default=None, converter=converters.optional(builds_path))
    renames = field(
        default=(),
        converter=rename_tuple,
        validator=deep_iterable(instance_of(Rename), instance_of(tuple)),
    )
    swaps = field(
        default=(),
        converter=swap_tuple,
        validator=deep_iterable(instance_of(Swap), instance_of(tuple)),
    )
    # Every swap's new connection.
    profile = field(default=None, converter=converters.optional(stored_profile))

    def __attrs_post_init__(self) -> None:
        if self.move_aliases and self.only_aliases:
            raise RebaseOptionsError(
                "--move-aliases and --only-alias are mutually exclusive"
            )
        if self.dry_run and not self.add:
            raise RebaseOptionsError("--dry-run and --no-add are mutually exclusive")
        if not self.add and (self.alias or self.move_aliases or self.only_aliases):
            raise RebaseOptionsError(
                "--no-add takes no --alias, --move-aliases or --only-alias: there "
                "is no entry to name"
            )
        if self.add and self.builds_dir is not None:
            raise RebaseOptionsError(
                "--builds-dir applies only with --no-add, which writes the build there"
            )
        if message := rename_refusal(repeated_renames(self.renames)):
            raise RebaseOptionsError(message)
        if self.profile is not None and not self.swaps:
            raise RebaseOptionsError("--profile needs at least one --swap")
        if message := "; ".join(repeated_swaps(self.swaps, self.profile is not None)):
            raise RebaseOptionsError(message)

    @property
    def build_target_dir(self) -> Path:
        return self.builds_dir if self.builds_dir is not None else Path("builds")


@frozen
class ColumnChange:
    """One output column a rebase changed; ``None`` on the side it's absent."""

    name = field(validator=instance_of(str))
    before = field(validator=optional(instance_of(str)))
    after = field(validator=optional(instance_of(str)))

    def __attrs_post_init__(self) -> None:
        # Absent on both sides, or the same type on both, is no change.
        if self.before == self.after:
            raise ValueError(
                f"{self.name}: {self.before} -> {self.after} changes nothing"
            )

    def __str__(self) -> str:
        if self.before is None:
            return f"{self.name} {self.after} added"
        if self.after is None:
            return f"{self.name} removed"
        return f"{self.name} {self.before} -> {self.after}"


def column_changes(before: Schema, after: Schema) -> tuple[ColumnChange, ...]:
    """The output columns ``after`` retyped, added or removed, in order.

    A source column's new type can retype the columns derived from it too
    (an int64 that becomes float64 makes ``a + 1`` float64), which the sweep,
    comparing sources only, does not show.
    """
    old, new = dict(before.items()), dict(after.items())
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
class RebaseConflict:
    """What a conflicted rebase could not follow.

    ``op_name`` is the op that could not be rebuilt, ``None`` when a source is
    gone. ``sources`` are the sweep's reports for every changed source in the
    entry on an op failure, or for every source the sweep did not find equal
    (gone, changed, unreachable) on a missing-table conflict. ``detail`` is
    the cause as ``Type: message``: the op's error, or what refused the live
    schemas.
    """

    detail = field(validator=instance_of(str))
    sources = field(
        converter=tuple,
        validator=deep_iterable(instance_of(LeafReport), instance_of(tuple)),
    )
    op_name = field(default=None, validator=optional(instance_of(str)))
    # Lines that may help: on an op failure, what each changed source lost and
    # gained, less what a `--rename` already maps.
    hints = str_tuple()


@frozen
class RebaseResult:
    """``new_entry`` is ``old_entry`` unless ``status`` is ``REBASED`` or
    ``PUSH_FAILED``.

    ``new_hash`` is the rebased build's: ``new_entry``'s, or on ``PREVIEW``
    and ``BUILT`` the one a rebase would catalog.
    """

    status = field(validator=in_(tuple(RebaseStatus)))
    old_entry = field(validator=instance_of(CatalogEntry))
    new_entry = field(validator=instance_of(CatalogEntry))
    reports = field(
        converter=tuple,
        validator=deep_iterable(instance_of(LeafReport), instance_of(tuple)),
    )
    # The record ``reports`` were probed from.
    record = field(validator=instance_of(BuildRecord))
    # Empty unless REBASED or PUSH_FAILED, or PREVIEW, where it is what the
    # rebase would do: nothing moves onto an entry that already was.
    aliases = field(factory=AliasOutcome, validator=instance_of(AliasOutcome))
    # The sources no probe can compare, on UNPROBED.
    unprobed = str_tuple()

    # Set exactly when ``status`` is ``CONFLICT``.
    conflict = field(default=None, validator=optional(instance_of(RebaseConflict)))
    # Set exactly when ``status`` is ``PUSH_FAILED``: what the push raised.
    push_error = field(default=None, validator=optional(instance_of(Exception)))
    new_hash = field(
        default=Factory(lambda self: self.new_entry.name, takes_self=True),
        validator=instance_of(str),
    )
    # Set exactly when ``status`` is ``BUILT``: the build directory written.
    build_path = field(default=None, validator=optional(instance_of(Path)))
    # Set exactly when an entry is cataloged (REBASED, PUSH_FAILED): false when
    # it already was, from an earlier rebase, and its archive was kept.
    created = field(default=None, validator=optional(instance_of(bool)))
    # Set exactly when a rebase was derived (REBASED, PUSH_FAILED, PREVIEW,
    # BUILT): how the entry's output schema changed, empty if it didn't.
    output_changes = field(
        default=None,
        converter=converters.optional(tuple),
        validator=optional(
            deep_iterable(instance_of(ColumnChange), instance_of(tuple))
        ),
    )
    # Set exactly when ``output_changes`` is: the renames put above their
    # refreshed sources, empty if none was asked for.
    renames = field(
        default=None,
        converter=converters.optional(tuple),
        validator=optional(deep_iterable(instance_of(Rename), instance_of(tuple))),
    )
    # Set exactly when ``output_changes`` is: the sources swapped, empty if
    # none was asked for.
    swaps = field(
        default=None,
        converter=converters.optional(tuple),
        validator=optional(
            deep_iterable(instance_of(SwappedSource), instance_of(tuple))
        ),
    )

    def __attrs_post_init__(self) -> None:
        conflicted = self.status == RebaseStatus.CONFLICT
        if conflicted != (self.conflict is not None):
            raise ValueError(f"a {self.status} rebase has conflict={self.conflict!r}")
        if (self.status == RebaseStatus.PUSH_FAILED) != (self.push_error is not None):
            raise ValueError(
                f"a {self.status} rebase has push_error={self.push_error!r}"
            )
        cataloged = self.status in (RebaseStatus.REBASED, RebaseStatus.PUSH_FAILED)
        if cataloged != (self.created is not None):
            raise ValueError(f"a {self.status} rebase has created={self.created!r}")
        derived = cataloged or self.status in (RebaseStatus.PREVIEW, RebaseStatus.BUILT)
        if derived != (self.output_changes is not None):
            raise ValueError(
                f"a {self.status} rebase has output_changes={self.output_changes!r}"
            )
        if derived != (self.renames is not None):
            raise ValueError(f"a {self.status} rebase has renames={self.renames!r}")
        if derived != (self.swaps is not None):
            raise ValueError(f"a {self.status} rebase has swaps={self.swaps!r}")
        if (self.status == RebaseStatus.UNPROBED) != bool(self.unprobed):
            raise ValueError(f"a {self.status} rebase has unprobed={self.unprobed!r}")
        if (self.status == RebaseStatus.BUILT) != (self.build_path is not None):
            raise ValueError(
                f"a {self.status} rebase has build_path={self.build_path!r}"
            )
        # A preview or a build catalogs nothing, under a hash of its own; any
        # other outcome's hash is its entry's.
        if self.status in (RebaseStatus.PREVIEW, RebaseStatus.BUILT):
            uncataloged = self.new_entry is self.old_entry
            if not uncataloged or self.new_hash == self.old_entry.name:
                raise ValueError(f"a {self.status} rebase has new_hash={self.new_hash}")
            if self.build_path is not None and self.build_path.name != self.new_hash:
                raise ValueError(
                    f"build_path {self.build_path} is not named {self.new_hash}"
                )
        elif self.new_hash != self.new_entry.name:
            raise ValueError(
                f"a {self.status} rebase has new_hash={self.new_hash}, "
                f"not {self.new_entry.name}"
            )

    @property
    def push_message(self) -> str | None:
        return None if self.push_error is None else format_error(self.push_error)

    @property
    def sync_hint(self) -> str | None:
        """What finishes a failed push: ``xorq catalog sync``, once it can push.

        Sync pulls and pushes again. That alone finishes a push the remote
        could not be reached for. A ``CatalogPushError`` is any rejection: a
        remote that moved on, which the pull fixes, or a hook that refused the
        entry, which it does not. Anything else (a failed consistency check, a
        misconfigured remote) would fail the sync's push the same way.
        """
        sync = "`xorq catalog sync`"
        match self.push_error:
            case None:
                return None
            case GitCommandError():
                return f"run {sync}"
            case CatalogPushError():
                return (
                    f"if the remote moved on, run {sync}; if it rejected the "
                    "entry, fix that first"
                )
            case _:
                return f"fix that, then run {sync}"

    def to_dict(self) -> dict:
        """The ``rebase --json`` document; see the module docstring."""
        conflict = self.conflict
        conflicted = self.status == RebaseStatus.CONFLICT
        _, drift = reports_document(self.record, self.reports)
        # Nothing new is cataloged on a conflict, a preview or a build.
        uncataloged = (RebaseStatus.PREVIEW, RebaseStatus.BUILT)
        document = make_rebase_document(
            self.status,
            self.old_entry.name,
            new_entry=None
            if conflicted or self.status in uncataloged
            else self.new_entry.name,
            new_hash=None if conflicted else self.new_hash,
            build_path=self.build_path,
            created=self.created,
            output_changes=self.output_changes,
            renames=self.renames,
            swaps=self.swaps,
            failing_op=conflict.op_name if conflicted else None,
            drift=drift,
            aliases=self.aliases,
        )
        error = conflict.detail if conflicted else self.push_message
        if error is not None:
            document["error"] = error
        return document


def make_rebase_document(
    state: RebaseStatus | RebaseFailure,
    old_entry: str,
    *,
    new_entry: str | None = None,
    new_hash: str | None = None,
    build_path: Path | None = None,
    created: bool | None = None,
    unpushed: bool | None = None,
    output_changes: Iterable[ColumnChange] | None = None,
    renames: Iterable[Rename] | None = None,
    swaps: Iterable[SwappedSource] | None = None,
    failing_op: str | None = None,
    drift: dict | None = None,
    aliases: AliasOutcome | None = None,
) -> dict:
    """The keys every ``rebase --json`` document carries, built in one place.

    Every outcome comes through here, so the key set cannot grow on one and
    not another, and ``exit_code`` is read off ``state``, the value the CLI
    exits with.
    """
    return {
        **state_and_code(state),
        "old_entry": old_entry,
        "new_entry": new_entry,
        "new_hash": new_hash,
        "build_path": None if build_path is None else str(build_path),
        "created": created,
        "unpushed": unpushed,
        "output_changes": None
        if output_changes is None
        else [asdict(change) for change in output_changes],
        "renames": None
        if renames is None
        else [rename.to_dict() for rename in renames],
        "swaps": None if swaps is None else [swap.to_dict() for swap in swaps],
        "failing_op": failing_op,
        "drift": drift,
        "aliases": {
            name: list(names)
            for name, names in asdict(aliases or AliasOutcome()).items()
        },
        # The run's, not the outcome's: `rebase_document` fills it in.
        "warnings": [],
    }


def rebase_failure(e: Exception) -> RebaseFailure:
    """The failure ``e`` ends a rebase with: a ``RebaseError``'s own, or a
    failed rollback's, which may have left part of the write."""
    match e:
        case RebaseError():
            return e.failure
        case RollbackError():
            return RebaseFailure.ROLLBACK_FAILED
        case _:
            return RebaseFailure.FAILED


def rebase_document(
    catalog_entry: CatalogEntry,
    outcome: RebaseResult | Exception,
    caught: Iterable[warnings.WarningMessage] = (),
) -> dict:
    """``outcome`` of rebasing ``catalog_entry``, as ``rebase --json`` prints it.

    ``caught`` is what the rebase warned, which the document carries instead of
    stderr.
    """
    if isinstance(outcome, Exception):
        refusal = outcome if isinstance(outcome, RebaseError) else None
        document = make_rebase_document(
            rebase_failure(outcome),
            catalog_entry.name,
            new_hash=refusal and refusal.new_hash,
            drift=reports_document(refusal.record, refusal.reports)[1]
            if refusal is not None and refusal.record is not None
            else None,
            unpushed=catalog_entry.catalog.unpushed(),
        )
        document["error"] = format_error(outcome)
    else:
        document = outcome.to_dict()
    document["warnings"] = [f"{w.category.__name__}: {w.message}" for w in caught]
    return document


@contextmanager
def keeping_read_posture() -> Iterator[None]:
    """Silence the local-path warning a dump with ``relocate_reads=False`` raises.

    The warning asks for ``--relocate-reads``, which rebase deliberately does
    not do: an external read stays external so the new entry matches the old
    one's posture.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LocalReadPathWarning)
        yield


@contextmanager
def refusing_with(**fields: object) -> Iterator[None]:
    """Attach ``fields`` to a ``RebaseError`` raised inside, for its document."""
    try:
        yield
    except RebaseError as e:
        raise e.carrying(**fields) from e


def pull_before(catalog: Catalog) -> None:
    """The sync's pull, refused as retryable: nothing of the rebase is written."""
    try:
        catalog.pull()
    except (CatalogPullError, CatalogMergeConflict, GitCommandError) as e:
        raise RebaseError(
            f"the sync's pull failed: {format_error(e)}", RebaseFailure.UNREACHABLE
        ) from e


def push_after(catalog: Catalog) -> Exception | None:
    """The sync's push, after the write: what it raised, or ``None`` once pushed.

    Not raised: the rebase is committed locally either way, so whatever the
    push raises, a consistency check before pushing included, leaves a
    ``PUSH_FAILED`` rebase rather than a failure that reads as nothing written.
    """
    try:
        catalog.push()
    except Exception as e:
        return e
    return None


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
            RebaseFailure.REFUSED,
        ) from e
    except (ContentStoreError, AnnexError, OSError) as e:
        raise RebaseError(
            f"{catalog_entry.name} archive could not be fetched: {format_error(e)}",
            RebaseFailure.UNREACHABLE,
        ) from e


def read_entry_record(catalog_entry: CatalogEntry) -> BuildRecord:
    record = read_record(catalog_entry)
    if isinstance(record, Exception):
        raise RebaseError(
            f"{catalog_entry.name} is unreadable: {format_error(record)}",
            RebaseFailure.UNREACHABLE,
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
            RebaseFailure.REFUSED,
        )
    unchecked = unchecked_leaves(record)
    if unchecked and len(unchecked) == len(record.external_leaves):
        return tuple(leaf.name for leaf in unchecked)
    try:
        check_refreshable(record)
    except SchemaRefreshError as e:
        raise RebaseError(
            f"{name} cannot be rebased: {e.cause}", RebaseFailure.REFUSED
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
    """Refuse an entry built on another Python minor, or on one it doesn't
    record: its UDFs are cloudpickled. ``ignore_mismatch`` turns the refusal
    into a ``PythonMinorWarning``."""
    name = catalog_entry.name
    try:
        recorded = recorded_python_minor(catalog_entry)
    except Exception as e:
        raise RebaseError(
            f"{name} is unreadable: {format_error(e)}", RebaseFailure.UNREACHABLE
        ) from e
    if recorded == tuple(sys.version_info[:2]):
        return
    running = ".".join(map(str, sys.version_info[:2]))
    problem = (
        f"{name} records no Python minor"
        if recorded is None
        else f"{name} was built on Python {'.'.join(map(str, recorded))}"
    )
    if not ignore_mismatch:
        raise RebaseError(
            f"{problem}, this is {running}; pass --ignore-venv-mismatch to rebase "
            "anyway",
            RebaseFailure.REFUSED,
        )
    warnings.warn(
        f"{problem}; rebasing under {running}, but its cloudpickled UDFs may not load",
        PythonMinorWarning,
        stacklevel=2,
    )


def check_bundle(catalog_entry: CatalogEntry) -> None:
    """Refuse an archive that lacks the wheels or requirements to re-add."""
    name = catalog_entry.name
    try:
        with zipfile.ZipFile(catalog_entry.catalog_path) as zf:
            wheels, requirements, _ = bundle_members(zf)
    except Exception as e:
        raise RebaseError(
            f"{name} is unreadable: {format_error(e)}", RebaseFailure.UNREACHABLE
        ) from e
    if not wheels:
        raise RebaseError(
            f"{name} carries no wheel for the rebased entry", RebaseFailure.UNREACHABLE
        )
    # Without it, `catalog.add` would package the cwd's project requirements.
    if requirements is None:
        raise RebaseError(
            f"{name} carries no {DumpFiles.requirements} for the rebased entry",
            RebaseFailure.UNREACHABLE,
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
            RebaseFailure.UNREACHABLE,
        ) from e
    if missing := [path for path in paths if path is not None]:
        raise RebaseError(
            f"{catalog_entry.name}: database {', '.join(missing)} does not exist",
            RebaseFailure.UNREACHABLE,
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
            return RebaseConflict(format_error(e.cause), involved)
        raise RebaseError(
            f"{catalog_entry.name}: {e.cause}", RebaseFailure.UNREACHABLE
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
            f"{name} is unreadable: {format_error(e)}", RebaseFailure.UNREACHABLE
        ) from e
    (build_path / DumpFiles.requirements).write_bytes(requirements)


def stamp_rebased_from(
    build_path: Path, ancestor: str, swaps: tuple[SwappedSource, ...] = ()
) -> None:
    """Record ``ancestor``, and ``swaps`` if any, in ``build_path``'s
    ``build_metadata.json``.

    In the archive rather than only the sidecar: a catalog copy re-derives
    each sidecar from its archive. The build hash reads ``expr.yaml`` only.
    """
    path = build_path / DumpFiles.build_metadata
    metadata = json.loads(path.read_text()) | {REBASED_FROM: ancestor}
    if swaps:
        metadata[SWAPPED_SOURCES] = [swap.to_dict() for swap in swaps]
    path.write_text(json.dumps(metadata, indent=2))


# POSIX `rename(2)` replaces an empty directory, so `publish_build` claims the
# target with `mkdir` first; Windows refuses any existing target, so needs none.
CLAIM_BEFORE_RENAME = os.name != "nt"


def free_build_target(catalog_entry: CatalogEntry, builds_dir: Path, name: str) -> Path:
    """Where ``--no-add`` writes the build named ``name``, refused if taken.

    Whatever is there (an earlier ``xorq build`` of the same expression, which
    shares the ``builds`` default, or a build since edited) is never replaced; a
    dangling link counts as there. A ``builds_dir`` that is not a directory is
    refused here too, before anything is built.
    """
    if builds_dir.exists() and not builds_dir.is_dir():
        raise RebaseError(
            f"{catalog_entry.name}: {builds_dir} is not a directory; --no-add "
            "writes its build there",
            RebaseFailure.REFUSED,
        )
    target = builds_dir / name
    if target.is_symlink() or target.exists():
        raise RebaseError(
            f"{catalog_entry.name}: {target} exists; --no-add does not replace it",
            RebaseFailure.REFUSED,
        )
    return target


def publish_build(
    catalog_entry: CatalogEntry, build_path: Path, builds_dir: Path
) -> Path:
    """Move the complete ``build_path`` to ``builds_dir``, under its hash.

    Moved first into a hidden directory in ``builds_dir``, on the target's
    filesystem, then renamed into place, so the target appears whole or not at
    all: a copy that fails partway leaves nothing behind. The target is claimed
    with an exclusive ``mkdir`` before the rename, since a rename alone would
    replace an empty directory; so a directory that appeared since the check
    (another ``--no-add``, an ``xorq build``) is refused, never nested in or
    replaced, and one written into after the claim fails the rename.

    Windows never renames onto an existing directory, so there the absent
    target is its own claim (a path the catalog CI, on Linux, does not run). A
    process killed mid-copy can leave its stage, ``.<hash>.*`` in
    ``builds_dir``, which nothing reads and is safe to delete.
    """
    target = free_build_target(catalog_entry, builds_dir, build_path.name)
    builds_dir.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{build_path.name}.", dir=builds_dir))
    claimed = published = False
    try:
        staged = Path(shutil.move(build_path, stage / build_path.name))
        if CLAIM_BEFORE_RENAME:
            try:
                target.mkdir()
            except FileExistsError:
                # Taken since the check: refused as the check refuses.
                free_build_target(catalog_entry, builds_dir, build_path.name)
                raise
            claimed = True
        try:
            # Onto the empty claim, or (Windows) onto no directory at all.
            staged.rename(target)
        except OSError:
            if not (claimed and target.is_dir() and not any(target.iterdir())):
                # Taken since: written into the claim, or there on Windows.
                free_build_target(catalog_entry, builds_dir, build_path.name)
            raise
        published = True
    finally:
        if claimed and not published:
            # Given back on any failure, an interrupt included, so a retry is
            # not refused; `rmdir` leaves a claim someone wrote into, or one
            # already gone.
            with contextlib.suppress(OSError):
                target.rmdir()
        shutil.rmtree(stage, ignore_errors=True)
    return target


def refuse_renames(catalog_entry: CatalogEntry, details: Iterable[str]) -> None:
    """Refuse every failed ``--rename`` at once, so one run shows them all."""
    if message := rename_refusal(details):
        raise RebaseError(f"{catalog_entry.name}: {message}", RebaseFailure.REFUSED)


def quoted(names: Iterable[str]) -> str:
    """Each name quoted: a multi-path read's own name holds ``", "``."""
    return ", ".join(map(repr, names)) or "-"


def renamed_source(
    catalog_entry: CatalogEntry,
    record: BuildRecord,
    name: str,
    leaves: tuple[SourceLeaf, ...],
) -> dict[LeafKey, SourceLeaf] | str:
    """``leaf_key`` -> leaf for the one source of ``leaves`` ``name`` is, or
    why there is none.

    One source can hold several keys: a table bound before and after it grew
    is recorded at two schemas. One name on two connections is two sources,
    and refused, as is no source at all.
    """
    try:
        by_key = {leaf_key(leaf, record): leaf for leaf in leaves if leaf.name == name}
    except Exception as e:
        # The profile the sweep would rank `unreadable`: the same exit.
        raise RebaseError(
            f"{catalog_entry.name}: {name} is unreadable: {format_error(e)}",
            RebaseFailure.UNREACHABLE,
        ) from e
    if not by_key:
        sources = dict.fromkeys(leaf.name for leaf in leaves)
        return f"names no source {name!r}; its sources: {quoted(sorted(sources))}"
    # `leaf_key` less its snapshot pin and recorded schema: kind, profile, name.
    if (count := len({key[:3] for key in by_key})) > 1:
        return f"{name!r} names {count} sources (one name on different connections)"
    return by_key


def swap_refusal(
    swap: Swap,
    by_key: dict[LeafKey, SourceLeaf] | str,
    record: BuildRecord,
    profile: str | None = None,
) -> str | None:
    """Why ``swap`` onto ``profile``, if given, is refused, given what
    ``renamed_source`` found, if it is.

    One source is one kind: ``renamed_source`` refuses a name on two connections.
    """
    match by_key:
        case str():
            return f"--swap {by_key}"
        # `old` spells the namespace too: `aux.t t` is `t` back onto itself,
        # as is `t t` onto a profile with `t`'s connection.
        case {**leaves} if any(
            leaf_key(swapped_leaf(leaf, swap.new, profile), record)[:3] == key[:3]
            for key, leaf in leaves.items()
        ):
            return f"--swap {swap} swaps nothing"
        # A snapshot id is one table's: `new` has none of `old`'s.
        case {**leaves} if any(
            leaf.snapshot_id is not None for leaf in leaves.values()
        ):
            return (
                f"--swap {swap.old!r} is pinned to a snapshot, which "
                f"{swap.new!r} does not have"
            )
        case _:
            return None


def swapped_leaf(leaf: SourceLeaf, new: str, profile: str | None = None) -> SourceLeaf:
    """``leaf`` reading ``new`` instead, through ``profile`` if given: a
    read's one path, or a table in ``leaf``'s namespace. A bundled read
    stays bundled: the rebuild copies ``new`` into the archive."""
    profile = profile or leaf.profile
    if leaf.kind == LeafKind.READ:
        return evolve(
            leaf,
            name=new,
            profile=profile,
            paths=(new,),
            read_kwargs=update_read_kwargs(
                leaf.read_kwargs, ((ReadKwarg.hash_path, new),)
            ),
        )
    return evolve(
        leaf, table=new, name=table_name(leaf.namespace, new), profile=profile
    )


def bundled_reads(record: BuildRecord) -> tuple[SourceLeaf, ...]:
    """The reads whose bytes ``record``'s archive holds: no checkable leaf,
    but a swap re-points one to a new file."""
    return tuple(
        leaf
        for leaf in record.source_leaves
        if leaf.bundled and leaf.kind == LeafKind.READ
    )


def swapped_leaves(
    catalog_entry: CatalogEntry,
    record: BuildRecord,
    swaps: tuple[Swap, ...],
    profile: str | None = None,
) -> dict[LeafKey, SourceLeaf]:
    """``leaf_key`` -> the leaf as it reads once swapped, onto ``profile`` if
    given, for each source ``swaps`` names, found as ``--rename`` finds one.
    Refuses, all at once, a swap naming no source, sources on several
    connections, or a swap onto a table another swap, or a source left as
    is, reads: they would share one ``--rename`` plan. A bundled read is
    named by its archive path, as its record holds no other.
    """
    swappable = record.external_leaves + bundled_reads(record)
    resolved = {
        swap: renamed_source(catalog_entry, record, swap.old, swappable)
        for swap in swaps
    }
    refused = {
        swap: detail
        for swap, by_key in resolved.items()
        if (detail := swap_refusal(swap, by_key, record, profile)) is not None
    }
    onto = {
        # Keyed as the `--rename` plans are: kind, profile hash, name.
        swap: {
            leaf_key(swapped_leaf(leaf, swap.new, profile), record)[:3]
            for leaf in by_key.values()
        }
        for swap, by_key in resolved.items()
        if swap not in refused
    }
    details = [
        *refused.values(),
        *(
            " and ".join(f"--swap {swap}" for swap, ids in onto.items() if to in ids)
            + f" both swap to {to[2]}"
            for to, n in Counter(to for ids in onto.values() for to in ids).items()
            if n > 1
        ),
    ]
    moved = {key for swap in onto for key in resolved[swap]}
    details += [
        f"--swap {swap} swaps onto {to[2]}, which the entry already reads"
        for swap, ids in onto.items()
        for to in ids
        if isinstance(
            found := renamed_source(catalog_entry, record, to[2], swappable), dict
        )
        and any(key[:3] == to for key in found.keys() - moved)
    ]
    if details:
        raise RebaseError(
            f"{catalog_entry.name}: {'; '.join(details)}",
            RebaseFailure.REFUSED,
        )
    return {
        key: swapped_leaf(leaf, swap.new, profile)
        for swap, by_key in resolved.items()
        for key, leaf in by_key.items()
    }


def swapped_sources(
    record: BuildRecord, swapped: dict[LeafKey, SourceLeaf]
) -> tuple[SwappedSource, ...]:
    """Each source ``swapped`` (``swapped_leaves``'s) re-points, once: as
    recorded, and as it now reads."""
    names = {key[2] for key in swapped}
    return tuple(
        dict.fromkeys(
            SwappedSource(
                SourceAt(leaf.name, leaf.profile), SourceAt(new.name, new.profile)
            )
            for leaf in record.external_leaves
            if leaf.name in names
            and (new := swapped.get(leaf_key(leaf, record))) is not None
        )
    )


def loaded_con(expr: Expr, con: BaseBackend) -> BaseBackend:
    """The connection ``expr`` already holds to where ``con`` connects, else
    ``con``, so a source moved beside another there shares its backend."""
    content_hash = Profile.from_con(con).content_hash
    return next(
        (
            source
            for source in find_all_sources(expr)
            if Profile.from_con(source).content_hash == content_hash
        ),
        con,
    )


def vacated(
    record: BuildRecord, swapped: Mapping[LeafKey, SourceLeaf]
) -> dict[str, str]:
    """Each swapped source moved off its connection -> that connection's
    profile content hash."""
    return {
        key[2]: key[1]
        for key, leaf in swapped.items()
        if leaf_key(leaf, record)[1] != key[1]
    }


def profile_connection(name: str, profile: Profile, alias: str) -> BaseBackend:
    """``profile``'s connection, refused where opening it would create its
    database.

    sqlite opens never creating, which fails on a missing `file:` URI or
    attached database, yet the connection kept records ``profile`` as stored,
    which source matching keys on. duckdb only gets the path check, since a
    read-only open refuses the read-write one the entry may load on the same
    file.
    """
    if (path := missing_database(profile)) is not None:
        raise RebaseError(
            f"{name}: --profile {alias}: database {path} does not exist",
            RebaseFailure.UNREACHABLE,
        )
    kwargs = sqlite_no_create(profile)[1] if profile.con_name == "sqlite" else {}
    try:
        opened = profile.get_con(**kwargs)
        if not kwargs:
            return opened
        # A backend records the kwargs it is built with: built from the stored
        # ones, it opens with the never-creating ones `opened` used.
        con = type(opened)(**profile.kwargs_dict)
        con.do_connect(**maybe_substitute_env_vars(profile.kwargs_dict | kwargs))
        opened.disconnect()
        return con
    except Exception as e:
        raise RebaseError(
            f"{name}: --profile {alias}: {format_error(e)}", RebaseFailure.UNREACHABLE
        ) from e


def with_profile(record: BuildRecord, profile: Profile | None) -> BuildRecord:
    """``record`` also holding ``profile``, as a build records it, so a leaf
    swapped onto it is keyed and probed there."""
    if profile is None:
        return record
    recorded = profile.as_dict() | {"kwargs_tuple": profile.kwargs_dict}
    return evolve(record, profiles={**record.profiles, profile.hash_name: recorded})


def unswapped_refusal(
    catalog_entry: CatalogEntry,
    reports: Iterable[LeafReport],
    unswapped: tuple[SourceLeaf, ...],
) -> RebaseError | None:
    """Why a swap is refused beside a source it leaves that is not proven
    unchanged, if it is: each new entry carries one kind of change.

    A drifted source is refused as an entry would be, ranked as `check-sources`
    ranks it: a changed or missing table won't come back by retrying, an
    unreachable or unreadable source may.
    """
    name = catalog_entry.name
    blocking = tuple(
        report
        for report in reports
        if report.leaf in unswapped and report.verdict != Verdict.EQUAL
    )
    if not blocking:
        return None
    details = "; ".join(
        f"{report.leaf.name} changed; run `xorq catalog rebase {name}` first"
        if report.verdict == Verdict.CHANGED
        else f"{report.leaf.name} could not be checked "
        f"({report.error or report.verdict}); a swap needs every other source "
        "unchanged"
        for report in blocking
    )
    worst = roll_up(report.verdict for report in blocking)
    return RebaseError(
        f"{name}: {details}",
        RebaseFailure.REFUSED
        if worst in (Verdict.CHANGED, Verdict.TABLE_MISSING)
        else RebaseFailure.UNREACHABLE,
    )


def plan_renames(
    catalog_entry: CatalogEntry,
    record: BuildRecord,
    renames: tuple[Rename, ...],
    unprobed: tuple[str, ...],
    leaves: tuple[SourceLeaf, ...],
) -> tuple[dict[LeafKey, dict[str, str]], tuple[Rename, ...], list[str]]:
    """``leaf_key`` -> recorded name -> live name for each rename the record
    allows, ``renames`` with ``consumes`` set, and why each other one fails
    against the record. A ``new`` the source also recorded is allowed and
    warned about: from the schemas alone, a column that looks unchanged can't
    be told apart from one that was dropped and replaced by the renamed one.
    ``leaves`` are the sources a ``--rename`` can name.

    Not refused here: ``check_live_renames`` refuses these together with what
    fails against the live schema, after the sweep, so one run names every bad
    ``--rename`` and the refusal carries the sweep. ``RebaseOptions`` has
    refused a repeated column already.

    A source recorded at several schemas is renamed under each, whichever
    records ``old``.
    """
    if renames and unprobed:
        return (
            {},
            (),
            [f"needs a live schema, and no source can be probed ({quoted(unprobed)})"],
        )
    # Each source resolved once, however many of its columns are renamed.
    resolved = {
        source: renamed_source(catalog_entry, record, source, leaves)
        for source in dict.fromkeys(rename.source for rename in renames)
    }
    moved_on = {
        source: {rename.old for rename in renames if rename.source == source}
        for source in resolved
    }
    planned: dict[LeafKey, dict[str, str]] = {}
    checked = []
    details = []
    for rename in renames:
        (source, old, new) = (rename.source, rename.old, rename.new)
        if isinstance(by_key := resolved[source], str):
            details.append(by_key)
            continue
        if not any(old in leaf.recorded for leaf in by_key.values()):
            recorded = dict.fromkeys(
                name for leaf in by_key.values() for name in leaf.recorded
            )
            details.append(
                f"{source}: {old!r} is not a recorded column "
                f"(recorded: {quoted(recorded)})"
            )
            continue
        # Under every key, not only those that record `old`: the mapping is
        # the live source's, and a chain (`a <- x, b <- a`) moves `a` away
        # from a binding that never recorded `b` too.
        for key in by_key:
            planned.setdefault(key, {})[old] = new
        # A `new` another rename of this source maps onward (a chain or a
        # swap) is still a column of the expression.
        consumes = any(new in leaf.recorded for leaf in by_key.values()) and (
            new not in moved_on[source]
        )
        checked.append(evolve(rename, consumes=consumes))
    return planned, tuple(checked), details


def check_live_renames(
    catalog_entry: CatalogEntry,
    record: BuildRecord,
    reports: tuple[LeafReport, ...],
    planned: dict[LeafKey, dict[str, str]],
    failed: Iterable[str] = (),
    names: Mapping[LeafKey, str] = MappingProxyType({}),
) -> dict[LeafKey, dict[str, str]]:
    """``planned`` for the changed bindings, the ones ``refresh_schemas``
    rebuilds. Refuses, all at once and with ``failed`` (what failed against
    the record), each rename whose new column isn't live, or whose old one
    still is and no other rename of that source moves it away (a chain,
    ``a <- x, b <- a``, or a swap).

    An unchanged binding is never renamed or checked: a pinned snapshot keeps
    its schema while the current table moves on. A column only unchanged
    bindings record is refused, as still live: even a swap renamed nothing.
    An unreachable or unreadable source is left to the sweep's own refusal,
    and never keyed: its profile may be what made it unreadable. ``names``
    (``leaf_key`` -> name) names a swapped source as it was recorded.
    """
    details = [*failed]
    changed = {}
    # (source, old) that some changed binding renames, or only unchanged ones record.
    renamed = set()
    unchanged = []
    # With nothing planned there is nothing to key: `leaf_key` hashes a profile.
    for report in reports if planned else ():
        if report.verdict not in (Verdict.EQUAL, Verdict.CHANGED):
            continue
        key = leaf_key(report.leaf, record)
        if not (by_old := planned.get(key)):
            continue
        (name, live) = (names.get(key, report.leaf.name), report.live)
        if report.verdict == Verdict.EQUAL:
            unchanged += [(name, old) for old in by_old if old in report.leaf.recorded]
            continue
        changed[key] = by_old
        # Only what this binding records: every binding gets the source's whole
        # mapping, for chains, so the rest is no rename of its own.
        renamed |= {(name, old) for old in by_old if old in report.leaf.recorded}
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
    details += [
        f"{name}: {old!r} is still a live column, so nothing was renamed"
        for (name, old) in dict.fromkeys(unchanged)
        if (name, old) not in renamed
    ]
    refuse_renames(catalog_entry, details)
    return changed


def rename_hints(
    record: BuildRecord,
    reports: Iterable[LeafReport],
    planned: dict[LeafKey, dict[str, str]],
    names: Mapping[LeafKey, str] = MappingProxyType({}),
) -> tuple[str, ...]:
    """The columns each changed source lost and gained, less those a
    ``--rename`` already maps; a mapping is not guessed, nor a fix promised.

    A source bound more than once is one line, over every binding's record:
    a column another binding recorded is not new. ``names`` names a swapped
    source as it was recorded, as ``--rename`` takes it.
    """
    # `leaf_key` less its snapshot pin and recorded schema: kind, profile, name.
    sources: dict[tuple, tuple[str, dict, dict, dict]] = {}
    for report in reports:
        if report.verdict != Verdict.CHANGED:
            continue
        key = leaf_key(report.leaf, record)
        (_, recorded, live, by_old) = sources.setdefault(
            key[:3], (names.get(key, report.leaf.name), {}, {}, {})
        )
        recorded |= dict.fromkeys(report.leaf.recorded)
        live |= dict.fromkeys(report.live)
        by_old |= planned.get(key, {})
    lines = []
    for name, recorded, live, by_old in sources.values():
        if gone := [
            column for column in recorded if column not in live and column not in by_old
        ]:
            new = [
                column
                for column in live
                if column not in recorded and column not in by_old.values()
            ]
            lines.append(
                f"{name}: recorded columns gone: {', '.join(gone)}; "
                f"live columns new: {', '.join(new) or '-'}"
            )
    if not lines:
        return ()
    return (
        *lines,
        "if a gone column was renamed to a new one, "
        "--rename <source> <old> <new> maps it",
    )


def warn_consumed(renames: Iterable[Rename]) -> None:
    for rename in renames:
        if (message := rename.warning) is not None:
            warnings.warn(message, RenameWarning, stacklevel=3)


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
    dry_run: bool = False,
    add: bool = True,
    builds_dir: str | Path | None = None,
    renames: Iterable[Rename | tuple[str, str, str]] = (),
    swaps: Iterable[Swap | tuple[str, str]] = (),
    profile: str | None = None,
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

    ``dry_run`` stops where the new entry would be written, as ``PREVIEW``.
    ``add=False`` writes the rebased build into ``builds_dir`` (``builds`` by
    default) instead, with the old entry's bundle, and returns it as ``BUILT``;
    it takes no aliases and does not sync. Neither pulls first, so a synced
    rebase can still refuse an alias the pull moved. ``RebaseOptions`` states
    which arguments combine. Either way the new build records ``catalog_entry``
    as its ancestor.

    ``renames`` (``Rename``s, or ``(source, old, new)``) reads each recorded
    column ``old`` from the live column ``new``, keeping the recorded name, so
    every op over the source rebuilds unchanged. A rename the record or the
    sweep refutes is refused, so a rename with no drift, or of an entry none
    of whose sources can be probed, is refused rather than ignored. One whose
    ``new`` the source recorded too raises a ``RenameWarning``; a derived
    result's ``renames`` are the ones applied.

    ``swaps`` (``Swap``s, or ``(old, new)``) re-point each source named
    ``old`` to ``new``: a table beside it, or a read's new path, read by the
    same method and kwargs; a bundled read copies ``new`` into the archive.
    The sweep probes ``new`` against what ``old`` recorded, and the rebase
    goes on as over drift, so a swap with no schema change is still a new
    entry. A swap is refused, before the sweep, as a ``--rename`` would be;
    after it, if a source no swap names is not ``EQUAL``, so a new entry is
    never both swapped and drifted. A rename names a swapped source by
    ``old``. ``profile`` names a stored profile every swapped source moves
    to; a missing one is refused before anything is read.
    """
    from xorq.ibis_yaml.compiler import ExprDumper  # noqa: PLC0415

    options = RebaseOptions(
        alias=alias,
        move_aliases=move_aliases,
        only_aliases=only_aliases,
        dry_run=dry_run,
        add=add,
        builds_dir=builds_dir,
        renames=renames,
        swaps=swaps,
        profile=profile,
    )
    only_aliases, builds_dir = options.only_aliases, options.build_target_dir
    # `ExprDumper` validates `cache_dir` as a `Path`.
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    # Every check below opens the archive, which a pointer or annex catalog
    # may not have locally yet (a fresh clone has only the pointer).
    fetch_archive(catalog_entry)
    # Then: reading the record opens a `BuildZip`, which refuses a
    # bundle-less archive with a bare assertion.
    check_bundle(catalog_entry)
    # Connected once: a build records its connection's profile, which fills in
    # the defaults a stored profile may leave out.
    con = options.profile and profile_connection(
        catalog_entry.name, options.profile, profile
    )
    target = con and Profile.from_con(con)
    record = with_profile(read_entry_record(catalog_entry), target)
    moved_to = target and target.hash_name
    unprobed = check_rebasable(catalog_entry, record)
    # With no new name yet, only an unregistered alias is refused.
    plan = AliasPlan.read(
        catalog_entry,
        alias=alias,
        move_aliases=move_aliases,
        only_aliases=only_aliases,
        entry_alias=entry_alias,
    )
    swapped = swapped_leaves(catalog_entry, record, options.swaps, moved_to)
    moved_off = vacated(record, swapped)
    swap_records = swapped_sources(record, swapped)
    # A swapped source is probed where it now points, not where it was: every
    # binding of it, though bindings that share a key share one `swapped` entry.
    tables = {swap.old: swap.new for swap in options.swaps}
    swapped_bundled = tuple(
        leaf for leaf in bundled_reads(record) if leaf.name in tables
    )
    probed = checkable_leaves(record) + swapped_bundled
    leaves = tuple(
        swapped_leaf(leaf, tables[leaf.name], moved_to) if leaf.name in tables else leaf
        for leaf in probed
    )
    # With no swap, nothing is left unswapped for the swap guard to refuse.
    unswapped = tuple(
        leaf for leaf in checkable_leaves(record) if tables and leaf.name not in tables
    )
    reports = tuple(iter_leaf_reports(record, leaves=leaves))
    # A refusal from here on reports the sweep, as a conflict does.
    with refusing_with(record=record, reports=reports):
        if refusal := unswapped_refusal(catalog_entry, reports, unswapped):
            raise refusal
        # Inside: a source whose profile is unreadable is refused with the
        # sweep, as it is without --rename.
        # A bundled read is renamed only where swapped: elsewhere it can't drift.
        planned, checked_renames, failed = plan_renames(
            catalog_entry,
            record,
            options.renames,
            unprobed,
            record.external_leaves + swapped_bundled,
        )
        # `--rename` names a swapped source as recorded; the sweep keys it, and
        # the rebuild finds it, where it now points. Reported by its old name.
        moved = {key: leaf_key(leaf, record) for key, leaf in swapped.items()}
        planned = {moved.get(key, key): by_old for key, by_old in planned.items()}
        names = {new: old[2] for old, new in moved.items()}
        # Before the no-op: a rename of an unchanged source is refused, not
        # ignored.
        planned = check_live_renames(
            catalog_entry, record, reports, planned, failed, names
        )
        # Only a sweep that probed every source can prove a no-op on its own,
        # and a swap re-points a source even where its schema is unchanged.
        if not (unprobed or swapped) and all(
            report.verdict == Verdict.EQUAL for report in reports
        ):
            # The new entry is this one, so each of `only_aliases` must be on it;
            # `alias` is not added.
            plan.evolved(new=catalog_entry.name, alias=None)
            return RebaseResult(
                RebaseStatus.NOOP, catalog_entry, catalog_entry, reports, record
            )
        if unprobed and swapped:
            raise RebaseError(
                f"{catalog_entry.name}: {quoted(unprobed)} could not be checked; "
                "a swap needs every other source unchanged",
                RebaseFailure.REFUSED,
            )
        if unprobed:
            # Nothing can be refreshed, so nothing is loaded or written, and the
            # aliases are checked as on a no-op.
            plan.evolved(new=catalog_entry.name, alias=None)
            return RebaseResult(
                RebaseStatus.UNPROBED,
                catalog_entry,
                catalog_entry,
                reports,
                record,
                unprobed=unprobed,
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
        # A failed cleanup must not end a rebase whose entry is already committed.
        with (
            tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td,
            keeping_read_posture(),
        ):
            # Only before loading, which unpickles the UDFs: a no-op or an
            # unprobed entry loads nothing, whatever minor it was built on.
            check_python_minor(catalog_entry, ignore_mismatch)
            # Loaded inside the block: `load_expr` keeps the archive's extract dir,
            # which bundled reads point into, alive only as long as the expression.
            loaded = catalog_entry.load_expr(cache_dir=cache_dir)
            try:
                repointed = repoint_tables(
                    loaded,
                    swapped,
                    con and loaded_con(loaded, con),
                )
                # Whatever still runs on a connection a source moved off would
                # run it beside that source's new one: two backends.
                if any(
                    Profile.from_con(source).content_hash in moved_off.values()
                    for source in find_all_sources(repointed)
                ):
                    raise RebaseError(
                        f"{catalog_entry.name}: --profile {profile} moves "
                        f"{quoted(moved_off)} off a connection the entry still "
                        "uses; swap every source onto it",
                        RebaseFailure.REFUSED,
                    )
                # Only the loaded expression holds each binding's marked rename:
                # a --rename one would skip, onto a column it reads as its own,
                # is the user's to fix, so refused as a bad --rename, not a
                # conflict. Over `repointed`, which keys a swapped source as
                # `planned` does.
                refuse_renames(
                    catalog_entry, ambiguous_renames(repointed, live, planned, names)
                )
                expr = refresh_schemas(repointed, live, planned)
            except UnmatchedSourceError as e:
                raise RebaseError(
                    f"{catalog_entry.name}: {e.cause}", RebaseFailure.REFUSED
                ) from e
            except SchemaRefreshError as e:
                changed = [r for r in reports if r.verdict == Verdict.CHANGED]
                # The cause alone: the CLI's headline already names the op.
                return conflicted(
                    RebaseConflict(
                        format_error(e.cause),
                        changed,
                        e.op_name,
                        hints=rename_hints(record, changed, planned, names),
                    )
                )
            warn_consumed(checked_renames)
            # `relocate_reads=False` keeps each read's recorded posture: a bundled
            # read stays bundled, an external one external.
            dumper = ExprDumper(
                expr, builds_dir=td, cache_dir=cache_dir, relocate_reads=False
            )
            if dumper.expr_hash == catalog_entry.name:
                plan.evolved(new=catalog_entry.name, alias=None)
                return RebaseResult(
                    RebaseStatus.NOOP, catalog_entry, catalog_entry, reports, record
                )
            # A refusal from here on reports the hash the rebase derived.
            with refusing_with(new_hash=dumper.expr_hash):
                # As a table: an entry can be a column or a scalar.
                changes = column_changes(
                    loaded.as_table().schema(), expr.as_table().schema()
                )
                # The new name is known now: an alias on another entry is refused
                # before the dump and the sync's pull, not after them.
                plan = plan.evolved(new=dumper.expr_hash)
                if not add:
                    # Refused before the dump: nothing is built over a taken path.
                    free_build_target(catalog_entry, builds_dir, dumper.expr_hash)
                if dry_run:
                    return RebaseResult(
                        RebaseStatus.PREVIEW,
                        catalog_entry,
                        catalog_entry,
                        reports,
                        record,
                        # What the rebase would do to them, before any pull.
                        aliases=AliasOutcome.of(plan, plan),
                        new_hash=dumper.expr_hash,
                        output_changes=changes,
                        renames=checked_renames,
                        swaps=swap_records,
                    )
                build_path = dumper.dump_expr()
                stamp_rebased_from(build_path, catalog_entry.name, swap_records)
                stage_bundle(catalog_entry, build_path)
                if not add:
                    return RebaseResult(
                        RebaseStatus.BUILT,
                        catalog_entry,
                        catalog_entry,
                        reports,
                        record,
                        new_hash=build_path.name,
                        output_changes=changes,
                        renames=checked_renames,
                        swaps=swap_records,
                        build_path=publish_build(
                            catalog_entry, build_path, builds_dir
                        ).resolve(),
                    )
                catalog = catalog_entry.catalog
                if sync:
                    pull_before(catalog)
                new_entry, aliases, created = add_with_aliases(
                    plan, build_path, pulled=sync
                )
        push_error = push_after(catalog) if sync else None
        return RebaseResult(
            RebaseStatus.REBASED if push_error is None else RebaseStatus.PUSH_FAILED,
            catalog_entry,
            new_entry,
            reports,
            record,
            aliases=aliases,
            push_error=push_error,
            created=created,
            output_changes=changes,
            renames=checked_renames,
            swaps=swap_records,
        )
