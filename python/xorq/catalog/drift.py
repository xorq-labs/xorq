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
from typing import TYPE_CHECKING, Any

from attr import field, frozen
from attr.validators import deep_iterable, in_, instance_of, optional

from xorq.catalog.enums import LeafKind
from xorq.catalog.inspection import BuildRecord, SourceLeaf
from xorq.common.compat import StrEnum
from xorq.vendor.ibis.backends.profiles import Profile
from xorq.vendor.ibis.expr.schema import Schema


if TYPE_CHECKING:
    from xorq.catalog.catalog import CatalogEntry


# `Read` joins in xorq-labs/xorq#2296; this is the `DatabaseTable` spine.
CHECKABLE_KINDS = frozenset({LeafKind.DATABASE_TABLE})


class Verdict(StrEnum):
    """What a probe found. The exit code is a property of the verdict, not a
    table maintained beside it, so the two cannot drift apart."""

    EQUAL = "equal"
    UNREACHABLE = "unreachable"
    CHANGED = "changed"
    TABLE_MISSING = "table-missing"

    @property
    def exit_code(self) -> int:
        match self:
            case Verdict.EQUAL:
                return 0
            case Verdict.UNREACHABLE:
                return 2
            case _:
                return 3


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
    """``leaf``'s namespace as ibis spells it: a pair, a bare name, or nothing."""
    match tuple(part for part in leaf.namespace if part):
        case ():
            return None
        case (only,):
            return only
        case pair:
            return pair


def open_con(
    leaf: SourceLeaf, record: BuildRecord, con_cache: dict | None = None
) -> Any:
    """The backend connection ``leaf`` recorded.

    Its own function so that connection policy has one place to live.
    ``con_cache`` keys on the profile's content hash -- what
    ``compiler.profile_content_key`` returns -- so a sweep opens one connection
    per distinct profile rather than one per leaf.
    """
    if (profile_dict := record.get_profile_dict(leaf)) is None:
        raise ValueError(f"node {leaf.node_ref!r} records no profile")
    profile = make_profile(profile_dict)
    if con_cache is None:
        return profile.get_con()
    if (key := profile.content_hash) not in con_cache:
        try:
            con_cache[key] = profile.get_con()
        except Exception as e:
            # A failed connect is cached too: a dead backend takes the full
            # timeout to fail, and paying that once per leaf behind it is what
            # the cache exists to avoid.
            con_cache[key] = e.with_traceback(None)
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


def get_table_schema(con: Any, leaf: SourceLeaf) -> Schema | None:
    """``leaf``'s live schema, or ``None`` when the backend does not list the table.

    The listing is the only positive evidence of absence. Vendored ibis raises a
    typed ``TableNotFound`` in some backends and a bare error carrying the same
    message in others (sqlite among them), so classifying on the exception would
    misreport a renamed sqlite table.
    """
    database = table_location(leaf)
    if leaf.table not in con.list_tables(database=database):
        return None
    return con.table(leaf.table, database=database).schema()


def get_schema_reader(leaf: SourceLeaf) -> Callable[[Any, SourceLeaf], Schema | None]:
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


def probe_leaf(
    leaf: SourceLeaf, record: BuildRecord, con_cache: dict | None = None
) -> LeafReport:
    """Reach ``leaf`` through its recorded profile and compare the schemas.

    Anything the connection or the read raises is ``unreachable``: no cause is
    guessed from an error message. An unhandled leaf kind raises out of
    ``get_schema_reader`` before the probe starts.
    """
    read_schema = get_schema_reader(leaf)
    try:
        con = open_con(leaf, record, con_cache)
        live = read_schema(con, leaf)
    except Exception as e:
        return LeafReport(leaf, Verdict.UNREACHABLE, error=f"{type(e).__name__}: {e}")
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
            yield probe_leaf(leaf, record, con_cache)
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
        case Verdict.UNREACHABLE:
            yield f"    {report.error}"
        case _:
            pass


def format_no_external(record: BuildRecord) -> str:
    """Why an entry has nothing to check, counted by exempt kind.

    Under the default build rules most entries land here, so a bare string would
    read like a broken command.
    """
    counts = [f"{count} {kind or 'unknown'}" for kind, count in record.bundled_counts]
    if pinned := sum(leaf.pinned for leaf in record.source_leaves):
        counts.append(f"{pinned} pinned")
    detail = f" ({', '.join(counts)})" if counts else ""
    return f"  no external sources{detail}"


def format_unchecked(record: BuildRecord) -> str | None:
    """The external leaves ``checkable_leaves`` left out, or ``None``.

    Reported whatever else the entry produced: an entry whose other leaves are
    equal still exits 0, and staying quiet about the leaf nobody probed would
    make that a false negative stated as a positive claim.
    """
    checked = set(checkable_leaves(record))
    unchecked = tuple(leaf for leaf in record.external_leaves if leaf not in checked)
    if not unchecked:
        return None
    counts = Counter(str(leaf.kind) for leaf in unchecked)
    detail = ", ".join(
        kind if len(counts) == 1 else f"{count} {kind}"
        for kind, count in sorted(counts.items())
    )
    noun = "source" if len(unchecked) == 1 else "sources"
    return f"  {len(unchecked)} external {noun} not checkable ({detail})"
