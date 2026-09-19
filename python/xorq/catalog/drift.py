"""Compare the source schemas a build recorded against the ones the world has now.

Read-only: it reports drift and never repairs, and it never infers. A rename is
not observable from two schemas, so recorded and live print side by side and
nothing is editorialized.

The leaves come from ``inspection``, which reads them straight out of the
archive, so an entry whose expression can no longer load is still checkable.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from attr import field, frozen
from attr.validators import deep_iterable, in_, instance_of, optional

from xorq.catalog.enums import LeafKind, Verdict
from xorq.catalog.inspection import BuildRecord, SourceLeaf
from xorq.vendor.ibis.backends.profiles import Profile
from xorq.vendor.ibis.expr.schema import Schema


if TYPE_CHECKING:
    from xorq.catalog.catalog import CatalogEntry


# `Read` joins in xorq-labs/xorq#2296; this is the `DatabaseTable` spine.
CHECKABLE_KINDS = frozenset({LeafKind.DATABASE_TABLE})


@frozen
class LeafReport:
    """One leaf's verdict. ``live`` is ``None`` unless a schema was read."""

    leaf = field(validator=instance_of(SourceLeaf))
    verdict = field(validator=in_(tuple(Verdict)))
    live = field(default=None, validator=optional(instance_of(Schema)))
    error = field(default=None, validator=optional(instance_of(str)))

    @property
    def exit_code(self) -> int:
        return self.verdict.exit_code


@frozen
class EntryReport:
    """Every leaf report for one entry, plus the record they came from.

    Eager, so a caller that wants the whole answer (the `rebase` work, XOR-453)
    gets it in one call. The CLI streams ``iter_leaf_reports`` instead.
    """

    name = field(validator=instance_of(str))
    record = field(validator=instance_of(BuildRecord))
    leaf_reports = field(
        converter=tuple,
        validator=deep_iterable(instance_of(LeafReport), instance_of(tuple)),
    )

    @property
    def exit_code(self) -> int:
        """The worst verdict wins: a drifted leaf outranks an unreachable one."""
        return max((report.exit_code for report in self.leaf_reports), default=0)

    @classmethod
    def from_catalog_entry(cls, catalog_entry: CatalogEntry) -> EntryReport:
        record = BuildRecord.from_catalog_entry(catalog_entry)
        return cls(catalog_entry.name, record, tuple(iter_leaf_reports(record)))


def make_profile(profile_dict: dict) -> Profile:
    """A ``Profile`` from its ``profiles.yaml`` form.

    ``kwargs_tuple`` round-trips as either a mapping or a list of pairs, same as
    ``compiler.hydrate_cons`` handles it.
    """
    kwargs = dict(profile_dict)
    kwargs_tuple = kwargs["kwargs_tuple"]
    kwargs["kwargs_tuple"] = tuple(
        kwargs_tuple.items()
        if isinstance(kwargs_tuple, dict)
        else map(tuple, kwargs_tuple)
    )
    return Profile(**kwargs)


def table_location(leaf: SourceLeaf) -> tuple[str, str] | str | None:
    """``leaf``'s namespace as ibis spells it: a pair, a bare name, or nothing;
    a catalog with no database raises.

    The raw ``(catalog, database)`` pair is what gets matched, not a compacted
    one: ibis reads a lone string as a database, so demoting a catalog into that
    slot would probe somewhere the leaf never named. A catalog with nothing
    under it is malformed, and raising is what keeps it from being probed
    anywhere at all.

    A well-formed pair a backend cannot express is deliberately left alone: it
    is handed over as recorded and fails at the read, which the probe reports as
    an unreachable backend. Resolving it against the ``database`` annotation the
    backend declares on ``list_tables`` would be the honest verdict; it is out
    of scope here and has no ticket yet, only the xorq-labs/xorq#2293 epic.
    """
    match leaf.namespace:
        case (None | "", None | ""):
            return None
        case (None | "", database):
            return database
        case (catalog, None | ""):
            raise ValueError(f"catalog {catalog!r} without a database")
        case pair:
            return pair


def get_leaf_profile(leaf: SourceLeaf, record: BuildRecord) -> Profile:
    """The profile ``leaf`` records.

    Resolved before the probe opens a connection, so a leaf naming no profile
    and a dangling profile ref both raise a ``ValueError`` instead of being
    reported as an unreachable backend.
    """
    if (profile_dict := record.get_profile_dict(leaf)) is None:
        raise ValueError(f"node {leaf.node_ref!r} records no profile")
    return make_profile(profile_dict)


def sqlite_no_create(profile: Profile) -> tuple[str | None, dict]:
    """The path to check before connecting, and the kwargs that never create.

    sqlite spells it as a URI mode; `rw` not `ro`, since recovering a hot WAL
    needs write access to the sidecars. A plain path is percent-encoded into
    one: sqlite cuts at the first `?`, so an unescaped one truncates the path,
    loses the mode, and `rwc` creates a database there. It is also the only
    checkable case, since a recorded URI would have to be resolved back through
    the encoding and the optional `//localhost` authority. In-memory is ``None``.
    """

    def opens_without_creating(uri: str) -> bool:
        """Whether ``uri`` already refuses to create its database.

        No ``mode=`` means `rwc`, and every mode present has to be non-creating:
        sqlite takes the first of a repeat. Hand-split, since sqlite cuts the
        path at `?` and the query at `#`.
        """
        query = uri.partition("#")[0].partition("?")[2]
        modes = {
            param.removeprefix("mode=")
            for param in query.split("&")
            if param.startswith("mode=")
        }
        return bool(modes) and modes <= {"ro", "rw", "memory"}

    def with_mode_rw(uri: str) -> str:
        """``uri`` with every ``mode=`` replaced by one `rw`; the fragment goes
        too, since an appended mode behind `#` is never read."""
        path, _, query = uri.partition("#")[0].partition("?")
        params = [
            param
            for param in query.split("&")
            if param and not param.startswith("mode=")
        ]
        return f"{path}?{'&'.join([*params, 'mode=rw'])}"

    target = profile.kwargs_dict.get("database")
    if target in (None, ""):
        return None, {}
    target = str(target)
    # Without `uri=True` the whole string is a filename, and `rwc` creates it.
    if not profile.kwargs_dict.get("uri") or not target.startswith("file:"):
        return target, {"database": f"file:{quote(target)}?mode=rw", "uri": True}
    if opens_without_creating(target):
        return None, {}
    return None, {"database": with_mode_rw(target), "uri": True}


def duckdb_no_create(profile: Profile) -> tuple[str | None, dict]:
    """The path to check before connecting, and the kwargs that never create.

    `read_only=True` fails on a missing database and blocks writes besides. Not
    for an in-memory one (`:memory:`, optionally named), which duckdb refuses
    read-only, nor a MotherDuck handle, which is no local file.
    """
    target = profile.kwargs_dict.get("database")
    if target is None or str(target).startswith((":memory:", "md:", "motherduck:")):
        return None, {}
    return str(target), {"read_only": True}


def connect(profile: Profile) -> Any:
    """The connection ``profile`` names, or the error every leaf behind it gets.

    Only sqlite and duckdb create their database on open, and a read-only
    command must not: the fresh empty database would report `table-missing`.
    """
    match profile.con_name:
        case "sqlite":
            path, kwargs = sqlite_no_create(profile)
        case "duckdb":
            path, kwargs = duckdb_no_create(profile)
        case _:
            path, kwargs = None, {}
    # Before the connect: once the driver has raised, the file exists. It also
    # names the cause, which sqlite's message does not.
    if path is not None and not Path(path).exists():
        return FileNotFoundError(f"{profile.con_name} database {path} does not exist")
    try:
        # The check can go stale; the kwargs are what close that window.
        return profile.get_con(**kwargs)
    except Exception as e:
        # Returned, not raised, so it caches: one timeout per dead backend.
        return e.with_traceback(None)


def open_con(profile: Profile, con_cache: dict) -> Any:
    """The backend connection ``profile`` names, dialled at most once.

    Its own function so that the cache has one place to live. ``con_cache`` is
    required and the caller owns closing it, so every connection this module
    opens is one ``close_cons`` can reach.
    """
    from xorq.ibis_yaml.compiler import profile_content_key  # noqa: PLC0415

    if (key := profile_content_key(profile)) not in con_cache:
        con_cache[key] = connect(profile)
    if isinstance(con := con_cache[key], Exception):
        # Cleared on the way out as well: re-raising one instance appends the
        # raising frame to its traceback, so a profile behind N leaves would
        # otherwise hang N frames off a cache the sweep keeps for its whole run.
        raise con.with_traceback(None)
    return con


def close_cons(con_cache: dict) -> None:
    """Close what a sweep opened. A backend that cannot close is already gone,
    and a failure here is not evidence about any source."""
    for con in con_cache.values():
        if isinstance(con, Exception):
            continue
        try:
            con.disconnect()
        except Exception:
            pass


def get_table_schema(
    con: Any, leaf: SourceLeaf, location: tuple[str, str] | str | None
) -> Schema | None:
    """``leaf``'s live schema, or ``None`` when the backend does not list the table.

    The listing is the only positive evidence of absence. Vendored ibis raises a
    typed ``TableNotFound`` in some backends and a bare error carrying the same
    message in others (sqlite among them), so classifying on the exception would
    misreport a renamed sqlite table. ``location`` is ``table_location``'s
    result, passed through as ibis's ``database=``.
    """
    if leaf.table not in con.list_tables(database=location):
        return None
    return con.table(leaf.table, database=location).schema()


def get_schema_reader(
    leaf: SourceLeaf,
) -> Callable[[Any, SourceLeaf, tuple[str, str] | str | None], Schema | None]:
    """The reader that fetches ``leaf``'s live schema.

    Resolved before the probe opens a connection, so a missing arm raises a
    ``ValueError`` instead of being reported as an unreachable backend.
    """
    match leaf.kind:
        case LeafKind.DATABASE_TABLE:
            return get_table_schema
        # `Read` arrives in xorq-labs/xorq#2296; `checkable_leaves` filters every
        # other kind out before a probe can get here.
        case _:
            raise ValueError(f"no probe for leaf kind {leaf.kind}")


def format_error(e: Exception) -> str:
    """The one shape an error takes in this command's output."""
    return f"{type(e).__name__}: {e}"


def probe_leaf(
    leaf: SourceLeaf, record: BuildRecord, con_cache: dict | None = None
) -> LeafReport:
    """Reach ``leaf`` through its recorded profile and compare the schemas.

    Anything the connection or the read raises is ``unreachable``: no cause is
    guessed from an error message. An unhandled leaf kind raises out of
    ``get_schema_reader``, a malformed namespace out of ``table_location``, and
    a missing or dangling profile out of ``get_leaf_profile``, before the probe
    starts: all three are properties of the record, not evidence about a
    backend.

    Without a caller-owned ``con_cache`` the probe closes what it opened.
    """
    read_schema = get_schema_reader(leaf)
    location = table_location(leaf)
    profile = get_leaf_profile(leaf, record)
    owned = con_cache is None
    con_cache = {} if owned else con_cache
    try:
        con = open_con(profile, con_cache)
        live = read_schema(con, leaf, location)
    except Exception as e:
        return LeafReport(leaf, Verdict.UNREACHABLE, error=format_error(e))
    finally:
        if owned:
            close_cons(con_cache)
    if live is None:
        return LeafReport(leaf, Verdict.TABLE_MISSING)
    verdict = Verdict.EQUAL if live == leaf.recorded else Verdict.CHANGED
    return LeafReport(leaf, verdict, live=live)


def checkable_leaves(record: BuildRecord) -> tuple[SourceLeaf, ...]:
    """The leaves this command probes: external, and of a kind it can reach."""
    return tuple(
        leaf for leaf in record.external_leaves if leaf.kind in CHECKABLE_KINDS
    )


def iter_leaf_reports(
    record: BuildRecord, con_cache: dict | None = None
) -> Iterator[LeafReport]:
    """One report per checkable leaf, yielded as each probe finishes.

    Streaming matters: a dead remote can take ~19 s to fail and cannot be
    interrupted from Python, so buffering would turn slow progress into a hang.

    A caller sweeping several entries passes its own ``con_cache`` to share
    connections across them, and owns closing it; otherwise the connections
    this record opened are closed when the iterator finishes.
    """
    owned = con_cache is None
    con_cache = {} if owned else con_cache
    try:
        for leaf in checkable_leaves(record):
            # A defect in one leaf -- a malformed namespace out of
            # `table_location`, an unhandled kind out of `get_schema_reader`, a
            # missing or dangling profile out of `get_leaf_profile` --
            # is a property of that leaf, so it ranks `unreadable` and stays
            # with it: the leaves after it are still probed, and one of them
            # drifting still wins the exit code.
            try:
                report = probe_leaf(leaf, record, con_cache)
            except Exception as e:
                report = LeafReport(leaf, Verdict.UNREADABLE, error=format_error(e))
            yield report
    finally:
        if owned:
            close_cons(con_cache)


def format_schema(schema: Schema | None) -> str:
    """``name type, name type``, or ``-`` when there is no schema to show."""
    if schema is None:
        return "-"
    return ", ".join(f"{name} {dtype}" for name, dtype in schema.items()) or "-"


def format_leaf_report(report: LeafReport) -> Iterator[str]:
    """The lines for one leaf, indented under its entry."""
    yield f"  {report.leaf.kind} {report.leaf.name}: {report.verdict}"
    match report.verdict:
        case Verdict.CHANGED | Verdict.TABLE_MISSING:
            yield f"    recorded: {format_schema(report.leaf.recorded)}"
            yield f"    live:     {format_schema(report.live)}"
        case Verdict.UNREACHABLE | Verdict.UNREADABLE:
            yield f"    {report.error}"
        case _:
            pass


def format_no_external(record: BuildRecord) -> str:
    """Why an entry has nothing to check, counted by exempt kind.

    Under the default build rules most entries land here, so a bare string would
    read like a broken command.
    """
    counts = [f"{count} {kind or 'unknown'}" for kind, count in record.bundled_counts]
    # `bundled` already claimed the leaves it counted: a pin whose frozen read
    # was also bundled is one leaf, and counting it in both columns would make
    # the detail add up to more sources than the entry has.
    if pinned := sum(leaf.pinned and not leaf.bundled for leaf in record.source_leaves):
        counts.append(f"{pinned} pinned")
    detail = f" ({', '.join(counts)})" if counts else ""
    return f"  no external sources{detail}"


def format_unchecked(record: BuildRecord) -> str | None:
    """The external leaves ``checkable_leaves`` left out, or ``None``.

    Reported whatever else the entry produced: an entry whose other leaves are
    equal still exits 0, and staying quiet about the leaf nobody probed would
    make that a false negative stated as a positive claim.
    """
    unchecked = tuple(
        leaf for leaf in record.external_leaves if leaf.kind not in CHECKABLE_KINDS
    )
    if not unchecked:
        return None
    counts = Counter(str(leaf.kind) for leaf in unchecked)
    detail = ", ".join(
        kind if len(counts) == 1 else f"{count} {kind}"
        for kind, count in sorted(counts.items())
    )
    noun = "source" if len(unchecked) == 1 else "sources"
    return f"  {len(unchecked)} external {noun} not checkable ({detail})"
