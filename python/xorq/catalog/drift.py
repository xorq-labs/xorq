"""Compare the source schemas a build recorded against the ones the world has now.

Read-only: it reports drift and never repairs, and it never infers. A rename is
not observable from two schemas, so recorded and live print side by side and
nothing is editorialized.

The leaves come from ``inspection``, which reads them straight out of the
archive, so an entry whose expression can no longer load is still checkable.

The JSON document (``drift_document``, behind ``check-sources --json``) is the
machine-readable form of the same sweep, buffered and emitted once so a consumer
gets a complete document or none at all::

    {
      "state": "changed",              # the worst entry's state
      "exit_code": 3,                  # what the command exits with
      "unchecked_count": 1,            # unprobed sources; gate on zero, not size
      "entries": {
        "prod-matches": {
          "state": "changed",          # the worst leaf's state
          "exit_code": 3,
          "leaves": [
            {
              "kind": "DatabaseTable", # LeafKind
              "name": "public.matches",
              "state": "changed",      # Verdict
              "recorded": {"home_score": "int64", "city": "string"},
              "live": {"home_team_score": "int64", "city": "string"}
            },
            {
              "kind": "Read",
              "name": "/data/teams.parquet",
              "state": "table-missing",
              "recorded": {"team": "string"},
              "live": null
            },
            {
              "kind": "DatabaseTable",
              "name": "public.teams",
              "state": "unreachable",
              "recorded": {"team": "string"},
              "live": null,
              "error": "OperationalError: ..."
            }
          ],
          "unchecked": [{"kind": "Read", "name": "/data/audit.json"}],
          "bundled": {"memtables": 1, "reads": 1},
          "pinned": 0
        }
      }
    }

``state`` and ``exit_code`` sit at the root, beside ``entries`` rather than among
the entry names, so an entry named `state` cannot shadow the roll-up. Schemas are
column-name-to-dtype objects; there is deliberately no delta field, because the
delta is one line of set arithmetic on the consumer's side and publishing it
would be permanent API surface. Column *order* is part of what the probe
compares, so a table whose columns were only reordered reads `changed` while its
two objects still hold the same names and dtypes: a consumer that wants to see
why compares the two key sequences, which the document preserves, and not only
the sets. ``live`` is ``null`` for every verdict that read no schema
(`table-missing`, `unreachable`, `unreadable`); ``recorded`` never is, because a
leaf whose schema the record does not hold makes that whole record `unreadable`
rather than a leaf with nothing to compare against. ``error`` is present only
when there is one. A read path that no longer resolves is `table-missing`, the
file analogue of a renamed table, and carries no error; `unreachable` is what
the connection or the read raised, and is the verdict that carries one. An
entry whose record cannot be read carries ``error`` itself, no leaves, and empty
counts -- nothing was read, and the ``error`` beside them is what says those
zeros are not evidence.

A ``state`` of ``null`` is not a verdict spelled differently: it says the sweep
compared nothing there. An entry gets it when every external source it has went
unprobed, so ``leaves`` is empty and ``unchecked`` names them all, and the root
gets it when no entry reached a verdict. It still exits 0, because finding
nothing to compare is not a finding about a source. An entry whose sources are
all bundled keeps ``equal``: nothing outside the archive is a reason it cannot
drift, not a reason the question went unanswered. ``state`` describes the leaves
that were probed, so a consumer that needs every entry's sources accounted for
checks that entry's ``unchecked`` is empty rather than reading ``state`` alone.

The root rolls up only the entries that reached a verdict, so a sweep mixing a
null entry with an `equal` one still publishes `equal`: dropping the whole sweep
to null would bury a verdict that was reached, and would turn one unprobed source
into silence about every entry beside it. A root `equal` therefore does not say
every entry was compared. What says that is ``unchecked_count``, the unprobed
external sources of every entry the document carries, added up -- a count rather
than the entry's list, and named apart from ``unchecked`` because the two shapes
are not interchangeable. It is a gate and not a quantity: what it is for is
whether it is zero, and the two ways it departs from a strict tally of unprobed
sources are why a consumer must not display or threshold its magnitude. It is
added off those lists rather than off the entries behind them, so an entry
asked for under both its name and an alias owes the document two keys and
counts its unprobed sources once per key; and an entry whose record could not
be read counts nothing, having enumerated nothing -- what reports that one is
its own `unreadable` state and the exit code the root takes from it. Zero
therefore says every source of every readable entry was probed, which is what a
consumer gating on green reads beside ``state``.

The roll-up ranks on ``Verdict.severity``, a total order, so the state a sweep
publishes does not depend on the order its names were given.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from attr import field, frozen
from attr.validators import deep_iterable, in_, instance_of, optional

from xorq.catalog.enums import LeafKind, Verdict
from xorq.catalog.inspection import BuildRecord, SourceLeaf
from xorq.common.constants import READ_EXCLUDE_KEYS
from xorq.ibis_yaml.enums import BundledSourceTypes, ReadKwarg
from xorq.ibis_yaml.utils import namespace_to_database
from xorq.vendor.ibis.backends.profiles import Profile
from xorq.vendor.ibis.expr.schema import Schema
from xorq.vendor.ibis.util import normalize_filenames, promote_list


if TYPE_CHECKING:
    from xorq.catalog.catalog import CatalogEntry


# Both leaf kinds are probed. What is exempt was already dropped by
# `drift_exempt`, and that filter is load-bearing rather than an optimization: a
# bundled read's `hash_path` was rewritten to the archive-relative `read_path`,
# so probing one would resolve `reads/<hash>.parquet` against the process cwd
# and report `table-missing`, not the `equal` its bytes deserve.
CHECKABLE_KINDS = frozenset(LeafKind)

# The prefix every backend read method carries. `get_schema_reader` dispatches
# on a name that came out of the archive, so it is checked against this rather
# than handed to `getattr` on a live connection.
READ_METHOD_PREFIX = "read_"

# The recorded schema, under the two spellings `deferred_read_csv` gives it
# (`columns` for duckdb, `schema` for the rest) and duckdb's per-column
# override beside it. Never replayed: see `get_read_schema`.
RECORDED_SCHEMA_KEYS = frozenset({ReadKwarg.schema, ReadKwarg.columns, ReadKwarg.types})


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

    The mapping itself is ``namespace_to_database``, shared with the refreshing
    loader so that what the probe asks a backend for and what a rebase rebuilds
    over are the same place by construction. This function is the leaf-shaped
    door onto it.

    A well-formed pair a backend cannot express is deliberately left alone: it
    is handed over as recorded and fails at the read, which the probe reports as
    an unreachable backend. Resolving it against the ``database`` annotation the
    backend declares on ``list_tables`` would be the honest verdict; it is out
    of scope here and has no ticket yet, only the xorq-labs/xorq#2293 epic.
    """
    return namespace_to_database(*leaf.namespace)


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
    the encoding and the optional `//localhost` authority. In-memory is ``None``
    or `:memory:`.
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
    if target in (None, "", ":memory:"):
        return None, {}
    target = str(target)
    # Without `uri=True` the whole string is a filename, and `rwc` creates it.
    if not profile.kwargs_dict.get("uri") or not target.startswith("file:"):
        # An absolute path takes the empty authority: without the `//`, sqlite
        # reads the first segment of a `//`-prefixed path as one and refuses.
        authority = "//" if target.startswith("/") else ""
        return target, {
            "database": f"file:{authority}{quote(target)}?mode=rw",
            "uri": True,
        }
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


def read_call(leaf: SourceLeaf) -> tuple[tuple, dict]:
    """The paths and kwargs that replay ``leaf``'s read.

    Same split as ``Read.make_dt``, through the same shared exclusion constant,
    so the relocation bookkeeping keys are dropped in exactly one place.
    """
    args = tuple(value for key, value in leaf.read_kwargs if key == ReadKwarg.hash_path)
    kwargs = {
        key: value for key, value in leaf.read_kwargs if key not in READ_EXCLUDE_KEYS
    }
    return args, kwargs


def path_resolves(path: Any) -> bool:
    """Whether ``path`` still names something to read.

    ``normalize_filenames`` is the read's own resolver, so the probe asks the
    question the same way the read will answer it: a glob is expanded, a
    ``scheme://`` URI is left alone rather than paying for the fetch, and only a
    path that resolves to nothing raises. A bare ``Path.exists()`` cannot say
    any of that -- it reads a glob, an `az://` URI and a `file://` URI alike as
    files that are not there.

    Only the "nothing resolves" ``ValueError`` is read as absence. A path that
    is not a path at all is a damaged record, and it keeps raising rather than
    being answered with a positive claim that the source is gone.
    """
    try:
        return bool(normalize_filenames([path]))
    except ValueError:
        return False


def get_read_inference(method_name: str | None) -> Callable[[Any], Schema] | None:
    """The inference ``method_name``'s deferred read ran at build time, if any.

    A deferred read records the schema *its own* inference produced, not the
    one the bound backend would produce, so ``live`` has to come from that same
    inference. `deferred_read_csv` infers with pandas and `deferred_read_parquet`
    with datafusion -- both regardless of ``con`` -- and reading the file with
    the backend instead compares two inference engines about a file nobody
    touched: duckdb and datafusion type `2024-01-01` as a date and an all-empty
    csv column as text where pandas leaves the first a string and the second a
    float, and the engines disagree on parquet logical types too (float16,
    timestamp and time precision, unsigned and interval types, the large and
    nested variants). Either would report `changed` on an untouched file.

    The map is `defer_utils.DEFAULT_READ_INFERENCE`, the same object the
    deferred reads take their own defaults from -- read-only, so neither side
    can rebind an entry out from under the other and the two cannot drift apart.
    A method with no entry infers nothing at build time, and there the backend's
    read *is* the record.
    """
    from xorq.common.utils.defer_utils import DEFAULT_READ_INFERENCE  # noqa: PLC0415

    return DEFAULT_READ_INFERENCE.get(method_name)


def get_read_schema(
    con: Any, leaf: SourceLeaf, location: tuple[str, str] | str | None
) -> Schema | None:
    """``leaf``'s live schema, or ``None`` when its paths no longer resolve.

    A moved file is the file analogue of a renamed table. Nothing else is
    classified: every other read failure collapses to a bare error whose message
    does not even name the file, so there is nothing to classify on.

    Nothing durable is left behind either way: a read `get_read_inference`
    answers for never touches ``con``, and `is_checkable` has kept the replay
    arm below to the backends whose read registers a session-scoped table.
    ``location`` is unused: a read names its source by path, not by namespace.

    A schema the build *declared* rather than inferred -- the ``schema=`` of
    either `deferred_read_csv` or `deferred_read_parquet`, or a custom
    `deferred_read_csv` ``infer_schema=`` -- is the one thing this cannot
    answer. The archive records the declaration and nothing that says it was
    one, so the probe reports the file's own inference against it, and an
    override that disagreed with inference at build time still disagrees now: an
    untouched file reads as `changed`. Pinned for both readers by
    `test_a_declared_schema_is_compared_against_inference`.

    A read `get_read_inference` answers for reaches its source the way the
    *inference* does, not the way the profile's backend would: pandas' csv
    reader for `read_csv`, a fresh datafusion session for `read_parquet`. On a
    remote path that is a different filesystem stack from the one the build's
    backend used -- fsspec plus the scheme's driver (s3fs/adlfs/gcsfs) and their
    own credential resolution -- so a remote read the recorded backend could
    still reach reports as a bare error where that stack is missing. That is the
    price of a comparable answer: the build inferred through the same stack, so
    a schema fetched any other way would not be one.
    """
    args, kwargs = read_call(leaf)
    paths = tuple(path for arg in args for path in promote_list(arg))
    if not all(map(path_resolves, paths)):
        return None
    if (infer := get_read_inference(leaf.method_name)) is not None:
        # The path parameter unsplit, glob and all: the deferred read hands its
        # own to the inference the same way, and the two answers are only
        # comparable if the input was.
        return infer(args[0])
    # The recorded name is a *destination*. Replaying it registers the probe's
    # own table over whatever already carries that name on the sweep-wide
    # `con_cache` connection -- and `list_tables`, which `get_table_schema`
    # treats as the only positive evidence of absence, would then be reading
    # this probe's own work. The backend generates one.
    kwargs.pop(ReadKwarg.table_name, None)
    # The recorded schema is an *instruction*: a read told what its columns are
    # returns them whatever the file now holds, so `live` would be derived from
    # `recorded` and every comparison would be `equal`. Dropped rather than
    # rehydrated -- the serialized form is a plain mapping, and handing that
    # over raises instead, which is the same bug reported as an unreachable
    # backend. Unreached by the methods `get_read_inference` knows, which answer
    # above; it is what keeps a read that records a schema without a registered
    # inference from being asked to confirm its own record.
    for key in RECORDED_SCHEMA_KEYS:
        kwargs.pop(key, None)
    return getattr(con, leaf.method_name)(*args, **kwargs).schema()


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
        case LeafKind.READ:
            # The archive names the method the probe dispatches on, so it is
            # checked here, beside the other record defects, rather than
            # reaching `getattr` on a live connection: a record naming
            # `drop_table` must not be able to call it.
            if not str(leaf.method_name or "").startswith(READ_METHOD_PREFIX):
                raise ValueError(
                    f"node {leaf.node_ref!r} names no read method: {leaf.method_name!r}"
                )
            # `from_node_def` tolerates a read with no recorded path, naming it
            # after the node instead. There is nothing to replay for one, and
            # left to the probe it would call the read method with no argument
            # and rank the record's defect as an unreachable backend.
            if not any(key == ReadKwarg.hash_path for key, _ in leaf.read_kwargs):
                raise ValueError(f"node {leaf.node_ref!r} records no read path")
            return get_read_schema
        # Unreachable while `LeafKind` has exactly the two members above.
        case _:
            raise ValueError(f"no probe for leaf kind {leaf.kind}")


def read_is_session_scoped(con_name: str) -> bool:
    """Whether ``con_name``'s ``read_*`` registers a table in the session only.

    The memory backends hold their tables in the process rather than in a
    database, so replaying a read on one writes nothing that outlives the
    connection. Every other backend ingests: sqlite, postgres and databricks go
    through ADBC, and `deferred_read_*` records ``mode="replace"`` for them, so
    a replay would drop and overwrite whatever the recorded ``table_name``
    names. An unknown backend is assumed to ingest -- the cost of being wrong
    the other way is the user's data.
    """
    from xorq.ibis_yaml.compiler import memory_backends  # noqa: PLC0415

    return con_name in memory_backends


def leaf_con_name(leaf: SourceLeaf, record: BuildRecord) -> str | None:
    """The backend ``leaf`` would be probed on, or ``None`` when it cannot say.

    Never raises: a leaf naming no profile, or a dangling one, stays checkable
    so ``get_leaf_profile`` reports it as the record defect it is instead of
    being dropped from the report as an unprobeable kind.
    """
    try:
        profile_dict = record.get_profile_dict(leaf)
    except ValueError:
        return None
    return None if profile_dict is None else profile_dict.get("con_name")


def read_needs_con(leaf: SourceLeaf) -> bool:
    """Whether probing ``leaf`` has to dial the backend it recorded at all.

    A read `get_read_inference` answers for is read out of the file by the
    inference itself and never touches ``con``, so the recorded profile has no
    part in the answer. Dialling it anyway would report a parquet leaf whose
    duckdb file has since moved as `unreachable` on a question the file alone
    settles.
    """
    return not (
        leaf.kind == LeafKind.READ and get_read_inference(leaf.method_name) is not None
    )


def is_checkable(leaf: SourceLeaf, record: BuildRecord) -> bool:
    """Whether this command can probe ``leaf`` without writing to its backend.

    A ``Read`` with no registered inference is replayed against the connection
    it recorded, and on an ingesting backend that replay is a write, in a
    command whose whole contract is that it never repairs. Such a leaf is left
    unprobed and named by ``format_unchecked`` rather than probed destructively.

    A read `get_read_inference` answers for is checkable wherever it is bound:
    the inference reads the file and writes nothing, so a postgres- or
    sqlite-bound `deferred_read_parquet` is probed like any other.
    """
    if leaf.kind not in CHECKABLE_KINDS:
        return False
    if leaf.kind != LeafKind.READ or not read_needs_con(leaf):
        return True
    con_name = leaf_con_name(leaf, record)
    return con_name is None or read_is_session_scoped(con_name)


def read_record(catalog_entry: CatalogEntry) -> BuildRecord | Exception:
    """``catalog_entry``'s record, or the error that makes it ``unreadable``.

    Leaf extraction is a ``cached_property``, so it is forced here rather than
    left to the first probe: a record that cannot be read is not evidence about
    any backend, and it must rank the same verdict whichever output the caller
    asked for. Returned rather than raised so both formatters branch on one
    value instead of each growing its own handler.
    """
    try:
        record = BuildRecord.from_catalog_entry(catalog_entry)
        record.source_leaves
        return record
    except Exception as e:
        return e


def roll_up(verdicts: Iterable[Verdict]) -> Verdict:
    """The verdict a set of them rolls up to: the worst by ``Verdict.severity``.

    Not derivable from the exit code, which is why it is computed over the
    verdicts themselves: `unreachable` shares 2 with `unreadable` and `changed`
    shares 3 with `table-missing`, so a consumer handed only the code would have
    to guess which of the pair it is looking at. Ranked on ``severity`` rather
    than the code for the same reason one step on -- a tie left to `max` would
    be broken by arrival order, so the same catalog would publish a different
    state depending on which name was typed first. Nothing to roll up is
    `equal`, the same 0 an entry with no external sources exits with.
    """
    return max(verdicts, key=lambda verdict: verdict.severity, default=Verdict.EQUAL)


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

    A leaf ``is_checkable`` rules out raises here too rather than being probed
    anyway. ``iter_leaf_reports`` never hands one over, so this is what makes
    "the probe never writes" a property of the probe instead of a property of
    every caller that filters first.

    The connection is opened only for a reader that uses one: a read answered
    by its own inference is handed ``None``, so an unopenable profile is not
    reported as drift evidence about a file the probe can still read.

    Without a caller-owned ``con_cache`` the probe closes what it opened.
    """
    read_schema = get_schema_reader(leaf)
    location = table_location(leaf)
    if not is_checkable(leaf, record):
        raise ValueError(
            f"node {leaf.node_ref!r} cannot be probed without writing to "
            f"{leaf_con_name(leaf, record)!r}"
        )
    profile = get_leaf_profile(leaf, record)
    owned = con_cache is None
    con_cache = {} if owned else con_cache
    try:
        con = open_con(profile, con_cache) if read_needs_con(leaf) else None
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
    """The leaves this command probes: external, and reachable without writing.

    Same predicate as ``format_unchecked``'s, so what is probed and what is
    named as unprobed cannot disagree and leave a source in neither column.
    """
    return tuple(leaf for leaf in record.external_leaves if is_checkable(leaf, record))


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


def bundle_label(kind: BundledSourceTypes | None) -> str:
    """The one name a bundle kind takes in this command's output.

    Both renderings read it from here: a bundle written by a version this one
    does not recognize must not be called one thing by the human output and
    another by the document.
    """
    return str(kind or "unknown")


def format_no_external(record: BuildRecord) -> str:
    """Why an entry has nothing to check, counted by exempt kind.

    Under the default build rules most entries land here, so a bare string would
    read like a broken command.
    """
    counts = [f"{count} {bundle_label(kind)}" for kind, count in record.bundled_counts]
    if pinned := record.pinned_count:
        counts.append(f"{pinned} pinned")
    detail = f" ({', '.join(counts)})" if counts else ""
    return f"  no external sources{detail}"


def unchecked_leaves(record: BuildRecord) -> tuple[SourceLeaf, ...]:
    """The external leaves ``checkable_leaves`` left out.

    Its complement over the same predicate, so no external source can fall into
    neither column and go unmentioned by either output.
    """
    return tuple(
        leaf for leaf in record.external_leaves if not is_checkable(leaf, record)
    )


def format_unchecked(record: BuildRecord) -> str | None:
    """The external leaves ``checkable_leaves`` left out, or ``None``.

    Reported whatever else the entry produced: an entry whose other leaves are
    equal still exits 0, and staying quiet about the leaf nobody probed would
    make that a false negative stated as a positive claim.
    """
    unchecked = unchecked_leaves(record)
    if not unchecked:
        return None
    counts = Counter(str(leaf.kind) for leaf in unchecked)
    detail = ", ".join(
        kind if len(counts) == 1 else f"{count} {kind}"
        for kind, count in sorted(counts.items())
    )
    noun = "source" if len(unchecked) == 1 else "sources"
    return f"  {len(unchecked)} external {noun} not checkable ({detail})"


def schema_document(schema: Schema | None) -> dict[str, str] | None:
    """``{column: dtype}``, or ``None`` when there is no schema to report.

    Dtypes as strings: the question a consumer asks of two schemas is whether
    they differ, and a structured dtype would publish this command's view of a
    type system it does not own.
    """
    if schema is None:
        return None
    return {name: str(dtype) for name, dtype in schema.items()}


def leaf_document(report: LeafReport) -> dict:
    """One leaf report, as the JSON document carries it.

    ``live`` is absent as a schema for every verdict that read none, which
    `probe_leaf` already encodes, so `table-missing` and `unreachable` fall out
    of the report rather than being special-cased here. ``error`` appears only
    when there is one: a null on an `equal` leaf would read as a claim that the
    probe looked for an error and found none.
    """
    document = {
        "kind": str(report.leaf.kind),
        "name": report.leaf.name,
        "state": str(report.verdict),
        "recorded": schema_document(report.leaf.recorded),
        "live": schema_document(report.live),
    }
    if report.error is not None:
        document["error"] = report.error
    return document


def state_and_code(verdict: Verdict | None) -> dict:
    """The two keys a verdict decides, or the pair that says none was reached.

    ``None`` is not a verdict spelled differently: it says this sweep compared
    nothing here, so there is no state to report. It still exits 0, since
    finding nothing to compare is not a finding about a source.
    """
    if verdict is None:
        return {"state": None, "exit_code": 0}
    return {"state": str(verdict), "exit_code": verdict.exit_code}


def make_entry_document(
    verdict: Verdict | None,
    *,
    error: str | None = None,
    leaves: Iterable[dict] = (),
    unchecked: Iterable[dict] = (),
    bundled: Iterable[tuple[BundledSourceTypes | None, int]] = (),
    pinned: int = 0,
) -> dict:
    """The keys every entry carries, built in one place.

    Both the probed entry and the unreadable one come through here, so the key
    set cannot grow on one and not the other. ``exit_code`` is read off
    ``verdict`` rather than reduced a second way over the same reports: the
    enum owns that mapping, and a second derivation of it is exactly the drift
    ``Verdict.exit_code`` exists to prevent.
    """
    document = state_and_code(verdict)
    if error is not None:
        document["error"] = error
    return document | {
        "leaves": list(leaves),
        "unchecked": list(unchecked),
        "bundled": {bundle_label(kind): count for kind, count in bundled},
        "pinned": pinned,
    }


def record_document(
    record: BuildRecord, con_cache: dict | None = None
) -> tuple[Verdict | None, dict]:
    """One readable record's whole probe, as the JSON document carries it.

    Buffered by construction: `iter_leaf_reports` is drained before anything is
    returned, so a consumer gets a complete entry or none at all.

    The verdict comes back beside the document rather than only inside it, so
    the sweep rolls up enum values and the string is written once, where the
    document is built. Parsing ``state`` back out would make every future
    writer of that key a source of ``ValueError`` -- raised after the whole
    sweep has been probed, discarding a complete document.
    """
    reports = tuple(iter_leaf_reports(record, con_cache))
    unchecked = unchecked_leaves(record)
    # No verdict at all where nothing was probed and a source was left
    # unprobed: rolling those up to `equal` would state the strongest positive
    # claim the document can make on the weakest evidence it has, and a
    # consumer gating on `state == "equal"` would go green over an entry this
    # command never compared. An entry with no external sources keeps `equal`:
    # nothing outside the archive is a reason it cannot drift, not a reason the
    # question went unanswered.
    verdict = None if unchecked and not reports else roll_up(r.verdict for r in reports)
    return verdict, make_entry_document(
        verdict,
        leaves=(leaf_document(report) for report in reports),
        # Named rather than omitted: an entry that exits 0 while a source went
        # unprobed is a false negative unless the document says which source.
        unchecked=({"kind": str(leaf.kind), "name": leaf.name} for leaf in unchecked),
        bundled=record.bundled_counts,
        pinned=record.pinned_count,
    )


def entry_document(
    catalog_entry: CatalogEntry, con_cache: dict | None = None
) -> tuple[Verdict | None, dict]:
    """One entry's probe, or the error that kept it from being probed at all.

    An unreadable record carries that error itself and no leaves. Its counts
    read empty because nothing was read, not because the record holds none --
    the ``error`` key beside them is what says which. Its state and exit code
    are the ones the human output prints for the same entry, since both come
    from ``Verdict``. The verdict is returned beside the document, same as
    ``record_document``.
    """
    record = read_record(catalog_entry)
    if isinstance(record, Exception):
        return Verdict.UNREADABLE, make_entry_document(
            Verdict.UNREADABLE, error=format_error(record)
        )
    return record_document(record, con_cache)


def drift_document(
    named_entries: Iterable[tuple[str, CatalogEntry]], con_cache: dict | None = None
) -> dict:
    """The whole sweep, entries keyed by the name they were asked for under.

    ``state`` and ``exit_code`` sit beside ``entries`` rather than among the
    entry names: an entry can be called anything, and at the root one called
    `state` would shadow the roll-up the document exists to carry.

    An entry that reached no verdict is carried but not rolled up, so a sweep
    where none did says so at the root too rather than reporting the `equal` of
    the leaves it never had. Where one entry did, that verdict is the root's:
    null is not propagated, because a single unprobed source must not erase
    what every other entry established. ``unchecked_count`` is what keeps that
    honest -- the unprobed sources of the entries the document already carries,
    added off their own lists, so a root `equal` beside a non-zero count reads
    as the partial answer it is. Off the lists and not the entries behind them:
    a name and its alias owe two keys and are counted twice, and an unreadable
    entry enumerated nothing and is counted not at all, which its own state and
    the exit code it forces are what report. Both departures are why the key is
    a gate rather than a quantity: zero or not zero is what it answers, and its
    magnitude is not a number of sources a consumer can size a fix by.

    The roll-up runs over ``Verdict`` values rather than the ``state`` strings
    beside them: re-parsing what was just serialized would make any future
    writer of that key raise after the sweep is already paid for.

    A name is swept once however many times it was asked for. Keying the sweep
    by name already collapses a repeat into one entry, and probing it twice
    besides would pay the probe twice and let the second verdict quietly
    replace the first -- so a source that changed between the two probes could
    drop out of the document the exit code was owed for. The skip is by
    requested name, not by the entry the name resolves to: one entry asked for
    under both its name and an alias owes the document two keys, and is swept
    once per key rather than once per entry.

    ``con_cache`` is the caller's, same as the human path, so one dead profile
    costs the sweep one timeout rather than one per entry. Without one the
    sweep owns a cache for its whole run, since threading ``None`` down would
    give every entry a private cache and put that timeout back per entry.
    """
    owned = con_cache is None
    con_cache = {} if owned else con_cache
    entries = {}
    verdicts = []
    try:
        for name, catalog_entry in named_entries:
            if name in entries:
                continue
            verdict, entry = entry_document(catalog_entry, con_cache)
            entries[name] = entry
            if verdict is not None:
                verdicts.append(verdict)
    finally:
        if owned:
            close_cons(con_cache)
    return state_and_code(roll_up(verdicts) if verdicts else None) | {
        "unchecked_count": sum(len(entry["unchecked"]) for entry in entries.values()),
        "entries": entries,
    }
