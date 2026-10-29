"""Re-derive a catalog entry over its live sources, as a new entry (#2323).

The old entry is never edited or removed. Every refusal comes before the first
write: the checks read only the archive, the sweep connects through ``drift``'s
no-create guard, and nothing loads until the sweep has compared every source it
can probe. With a sync, the alias checks run again after its pull, so a refusal
there leaves what the pull merged in place, unpushed. A pull that fails is
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
and ``null`` on a conflict or a failure. ``created`` is set only once an entry
is cataloged (``rebased``, ``push-failed``): ``false`` when it already was, from
an earlier rebase, and its archive was kept. ``build_path`` is set only on
``built``. ``error`` is present only when there is one, always as ``Type:
message``: what refused or failed the rebase, what a conflict's op or source
raised, or what a failed push raised. ``warnings`` holds each warning the
rebase raised, as ``Category: message``, so stderr stays empty. On
``push-failed`` the new entry and its aliases are committed locally, and `xorq
catalog sync` publishes them.
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
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from attr import Factory, asdict, converters, field, frozen
from attr.validators import deep_iterable, in_, instance_of, optional
from git.exc import GitCommandError

from xorq.catalog.aliases import AliasOutcome, AliasPlan, add_with_aliases, str_tuple
from xorq.catalog.annex import AnnexError
from xorq.catalog.catalog import (
    Catalog,
    CatalogEntry,
    CatalogMergeConflict,
    CatalogPullError,
)
from xorq.catalog.constants import REBASED_FROM
from xorq.catalog.drift import (
    LeafReport,
    format_error,
    iter_leaf_reports,
    make_profile,
    missing_database,
    read_record,
    reports_document,
    roll_up,
    state_and_code,
    unchecked_leaves,
)
from xorq.catalog.enums import RebaseFailure, RebaseStatus, Verdict
from xorq.catalog.exceptions import (
    CatalogPushError,
    ContentIntegrityError,
    ContentStoreError,
    PythonMinorWarning,
    RebaseError,
    RebaseOptionsError,
)
from xorq.catalog.inspection import BuildRecord
from xorq.catalog.refresh import check_refreshable, live_schemas, refresh_schemas
from xorq.catalog.zip_utils import BuildZip, bundle_members, harvest_entry_from_zip
from xorq.common.exceptions import (
    LocalReadPathWarning,
    SchemaRefreshError,
    UnmatchedSourceError,
)
from xorq.ibis_yaml.enums import DumpFiles
from xorq.ibis_yaml.packager import parse_python_minor


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

    @property
    def build_target_dir(self) -> Path:
        return self.builds_dir if self.builds_dir is not None else Path("builds")


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
    """The failure ``e`` ends a rebase with: a ``RebaseError``'s own."""
    return e.failure if isinstance(e, RebaseError) else RebaseFailure.FAILED


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
        swept = isinstance(outcome, RebaseError) and outcome.record is not None
        document = make_rebase_document(
            rebase_failure(outcome),
            catalog_entry.name,
            drift=reports_document(outcome.record, outcome.reports)[1]
            if swept
            else None,
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
def refusing_with_sweep(
    record: BuildRecord, reports: tuple[LeafReport, ...]
) -> Iterator[None]:
    """Attach the sweep to a ``RebaseError`` raised inside, for its document."""
    try:
        yield
    except RebaseError as e:
        raise e.after_sweep(record, reports) from e


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


def stamp_rebased_from(build_path: Path, ancestor: str) -> None:
    """Record ``ancestor`` in ``build_path``'s ``build_metadata.json``.

    In the archive rather than only the sidecar: a catalog copy re-derives
    each sidecar from its archive. The build hash reads ``expr.yaml`` only.
    """
    path = build_path / DumpFiles.build_metadata
    metadata = json.loads(path.read_text())
    path.write_text(json.dumps(metadata | {REBASED_FROM: ancestor}, indent=2))


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
    """
    from xorq.ibis_yaml.compiler import ExprDumper  # noqa: PLC0415

    options = RebaseOptions(
        alias=alias,
        move_aliases=move_aliases,
        only_aliases=only_aliases,
        dry_run=dry_run,
        add=add,
        builds_dir=builds_dir,
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
    # A refusal from here on reports the sweep, as a conflict does.
    with refusing_with_sweep(record, reports):
        # Only a sweep that probed every source can prove a no-op on its own.
        if not unprobed and all(report.verdict == Verdict.EQUAL for report in reports):
            # The new entry is this one, so each of `only_aliases` must be on it;
            # `alias` is not added.
            plan.evolved(new=catalog_entry.name, alias=None)
            return RebaseResult(
                RebaseStatus.NOOP, catalog_entry, catalog_entry, reports, record
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
            # Loaded inside the block: `load_expr` keeps the archive's extract dir,
            # which bundled reads point into, alive only as long as the expression.
            loaded = catalog_entry.load_expr(cache_dir=cache_dir)
            try:
                expr = refresh_schemas(loaded, live)
            except UnmatchedSourceError as e:
                raise RebaseError(
                    f"{catalog_entry.name}: {e.cause}", RebaseFailure.REFUSED
                ) from e
            except SchemaRefreshError as e:
                changed = [r for r in reports if r.verdict == Verdict.CHANGED]
                # The cause alone: the CLI's headline already names the op.
                return conflicted(
                    RebaseConflict(format_error(e.cause), changed, e.op_name)
                )
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
                )
            build_path = dumper.dump_expr()
            stamp_rebased_from(build_path, catalog_entry.name)
            stage_bundle(catalog_entry, build_path)
            if not add:
                return RebaseResult(
                    RebaseStatus.BUILT,
                    catalog_entry,
                    catalog_entry,
                    reports,
                    record,
                    new_hash=build_path.name,
                    build_path=publish_build(
                        catalog_entry, build_path, builds_dir
                    ).resolve(),
                )
            catalog = catalog_entry.catalog
            if sync:
                pull_before(catalog)
            # Read after the pull, which can bring in an earlier rebase's entry.
            created = not catalog.contains(build_path.name)
            new_entry, aliases = add_with_aliases(plan, build_path, pulled=sync)
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
        )
