# ADR-2598: Detect source drift read-only by verdict; repair it by deriving a new entry

- **Status:** Accepted
- **Date:** 2026-11-09
- **Deciders:** Daniel Mesejo
- **Related:** ADR-0006, ADR-0012, ADR-0015

## Context

A catalog entry records the schema of every source it was built over: each database table and each file read. Upstream sources change after the build. A column is added, retyped or dropped, a table is renamed, a file moves. The entry still loads and still runs, against a world it was not built for, and the first sign is usually a wrong result or a failure deep inside an execution.

Two questions follow, and they have different risk profiles:

1. **Has a source drifted?** (`xorq catalog check-sources`, epic #2293.) Users run this on production catalogs, in CI, and on a schedule. It must be safe to run anywhere, at any time.
2. **Can the entry follow the drift?** (`xorq catalog rebase`, epic #2319.) This has to re-derive an expression without the author's build script, which the catalog never had.

Reasonable people could disagree on most of the choices below: whether detection may repair, whether a missing source is "drift", which engine a recorded schema is compared with, whether rebase edits the entry in place or creates a new one, and where re-derivation hooks into loading. This ADR records what was chosen, as built. It was deliberately written after `check-sources` shipped, from what the implementation established (#2300). Where the design as first specified chose differently, the choice is recorded under *Alternatives considered*.

## Decision drivers

- Looking must never change the user's sources or the catalog: no writes to the user's database, no database file created by looking for one, no cataloged state changed.
- The exit code must never contradict the verdicts printed beside it. It is the coarse answer; `--json` is the full one.
- A verdict of `changed` must mean the source changed, never that two tools disagree about an untouched source.
- A rebase must never lose the entry it started from, and a refused or conflicted rebase must write no entry.
- Nothing is guessed: a rename looks exactly like a drop plus an add, given two schemas.

## Decision

### Detection and repair are separate commands

`check-sources` reads sources and never writes to them or to the catalog. It does fetch an archive that is not local yet (a pointer or annex clone), which fills the local content cache. `rebase` is the only command that writes an entry. After the checks that need only the archive (its bundle, a pinned leaf, which sources can be probed, an `--only-alias` that names no alias) it sweeps the sources with the same `iter_leaf_reports` that `check-sources` uses (`python/xorq/catalog/drift.py`), so both commands judge a source the same way. `rebase --json` embeds that sweep as the entry's `check-sources --json` document under `drift`; its shape is in the `xorq.catalog.rebase` docstring.

### A closed set of verdicts; the exit code is a property of the verdict

Each probed source (a *leaf*) gets one `Verdict` (`python/xorq/catalog/enums.py`):

| Verdict | Meaning | Exit |
|---|---|---|
| `equal` | live schema matches the recorded one | 0 |
| `unreachable` | the backend could not be asked | 2 |
| `unreadable` | the record itself could not be read | 2 |
| `changed` | live and recorded schemas differ | 3 |
| `table-missing` | the backend answered and lists no such table, or a read's file path no longer resolves | 3 |

`Verdict.exit_code` is a `match` that names every member, and its fallback arm raises, so a new member fails at its first use instead of inheriting a code. An entry's state is the worst of its leaves by `Verdict.severity`, a total order that refines the exit code: within code 2, `unreadable` beats `unreachable` (retrying will not fix a record that does not parse); within code 3, `table-missing` beats `changed` (the source is gone, not different). The order is declared once, as the member order, so the same catalog rolls up to the same state whatever order the names were typed in. Exit 2 means the question could not be answered (infrastructure, or a record that does not parse); exit 3 means the world moved. A name that does not resolve, or a catalog that cannot be opened, exits 1 before any probe, a usage error exits click's own 2, and neither prints a report.

The drift taxonomy, one test per row:

| Mutation | Verdict | Test |
|---|---|---|
| add a column | `changed` | `test_drift.py::test_column_change_reports_both_schemas[added]` |
| drop a column | `changed` | `test_drift.py::test_column_change_reports_both_schemas[dropped]` |
| rename a column | `changed` | `test_drift.py::test_column_change_reports_both_schemas[renamed]` |
| retype a column | `changed` | `test_drift.py::test_column_change_reports_both_schemas[retyped]` |
| rename the table | `table-missing` | `test_drift.py::test_renamed_table_is_missing` |
| move the read's file | `table-missing` | `test_drift_reads.py::test_a_moved_file_is_missing` |

The fixture is a sqlite table plus a read built with `relocate_reads=False`, because those are the leaves a default build leaves outside the archive (next section).

The human footer and `--json` read the same per-entry state: the footer counts `N entries, D drifted, M missing` by that state, and the exit code is the roll-up of the same states (#2597).

### Missing needs positive evidence

`table-missing` is reported only when the backend answered and its table list does not contain the name, or when a read's file path no longer resolves. Any other failure is `unreachable`, a deleted database file included. A probe never infers absence from an error message.

### Probing never writes

- A sqlite database is opened with a no-create guard, so a deleted database file is reported `unreachable`, not recreated empty.
- A profile that failed to connect is not dialled again in the same sweep; connections are shared per profile through one cache.
- A `Read` on an ingesting backend (sqlite, postgres, databricks) is never replayed: its `read_*` is a write with `mode="replace"`. `is_checkable` keeps such reads unprobed unless they can be answered without the backend (next section).
- Two kinds of leaf are exempt, having nothing outside the archive to drift against: a *bundled* read, relocated into the archive, and a *pinned* leaf, reached only through a `CacheTag` pin (`.ls.pin`, `xorq catalog pin`).

An entry with no external sources says why: `no external sources (1 memtables, 1 reads)`, a count per bundle kind, plus `N pinned` for pinned leaves, which are exempt too. Under the default build rules most entries land here. Leaves that cannot be probed are named in the output as `unchecked`, never silently treated as `equal`. An entry whose only sources are unchecked has no verdict (`state: null` in `--json`), rather than claiming `equal` on no evidence.

### A read is compared using the tool that recorded it

`deferred_read_parquet` records the schema datafusion infers and `deferred_read_csv` the one pandas infers, whatever backend the read is bound to. The probe re-runs that same inference (`get_read_inference` in `python/xorq/catalog/drift.py`, reading `DEFAULT_READ_INFERENCE` in `python/xorq/common/utils/defer_utils.py`). Both sides come from one producer, so a backend's own type mapping (sqlite reading `int32` as `int64`, `date32` as `string`) is never reported as drift (#2311, pinned by #2574). As a side effect, such reads need no connection at all and are probed on ingesting backends too.

### Rebase derives a new entry and never mutates the old one

A rebase builds a new archive with a new hash and records its ancestor as `rebased_from` in the archive's build metadata (`python/xorq/catalog/constants.py`). The old entry is untouched. No alias moves unless asked: `--move-aliases` moves every alias of the old entry, `--only-alias` the ones it names, and `--alias` adds a new one. Ancestry lives in the archive, not only in the catalog sidecar, because a catalog copy re-derives sidecars from archives.

Rebase syncs by default (`--no-sync` opts out): it pulls the catalog's remotes before writing and pushes after. A refusal or a conflict writes no entry. Two cases do leave something behind: with a sync, the alias checks run again after the pull, so a refusal there leaves what the pull merged in place; and a write that fails is rolled back, unless the rollback fails too (`rollback-failed`), which names what is left.

### Re-derivation is a rewrite after loading, driven by the drift report

The build loads exactly as recorded (`load_expr` is untouched). `rebase_entry` (`python/xorq/catalog/rebase.py`) turns its sweep into live schemas, or a conflict when a source is gone, and `refresh_schemas` in `python/xorq/catalog/refresh.py` rebuilds the loaded graph over them. The sweep is not repeated: the loaded graph is refreshed from the one report rebase already holds.

- Only sources reported `changed` move. An `equal` source keeps its node, so undrifted subtrees and their cache keys stay put.
- Every op above a moved source is recreated through `__recreate__`, which re-runs its own signature validation. A dropped column fails at the `Field` that names it, and a column retyped to string fails at the `Mean` over it. The one exception is Flight ops, which validate their schema only when built through `from_expr`; `revalidate_flight` calls that check by hand. Recreation stops at a `CacheTag` pin, which is kept as recorded; rebase meets one only above a source a live branch also reaches, since it refuses a pinned leaf.
- A probed source is matched to its loaded node by the connection profile's content hash, table name, iceberg snapshot id and recorded schema, not by backend name, so one table name on two connections is two sources.
- The loader opens its own connections. The sweep's connections carry the no-create guard's URI options, which would otherwise leak into the rebased build's recorded profile.

### Rebase outcomes and exit codes

Rebase keeps `check-sources`' reading of 0 (nothing to do) and 2 (could not be answered; nothing written), and adds its own (`RebaseStatus`, `RebaseExit`, `RebaseFailure` in `python/xorq/catalog/enums.py`):

| Exit | Outcomes |
|---|---|
| 0 | `noop` (no drift, or drift that re-derives to the same hash), `unprobed` (no source could be probed; stderr says drift was not ruled out), `rebased` |
| 1 | refused, for a reason retrying will not fix (a pinned leaf, an alias on another entry, among others), or a failed write; `xorq catalog rebase --help` lists every cause |
| 2 | a source unreachable or unreadable, its database file missing, or its reads disagreeing on a live schema; the record unreadable or lacking its wheel or requirements; the sync's pull failed; invalid options; nothing written |
| 3 | derived but not cataloged: `--dry-run` (`preview`) or `--no-add` (`built`) |
| 4 | `conflict`: an op no longer fits its inputs (`failing_op` names it), or a source's table is gone (`failing_op` is null); names the sources involved and what each lost and gained |
| 5 | `push-failed`: rebased locally, the sync's push failed |

An entry with only some sources unprobeable is refused rather than rebased, because those sources would come back unchanged beside refreshed ones.

### Renames are explicit

Rebase never guesses a rename. A renamed column is a removed one unless `--rename SOURCE OLD NEW` says which live column it is now; ops above the source keep seeing `OLD`, and a later rebase of the result keeps following the rename.

## Alternatives considered

The first five are what `check-sources` was first specified to do; each was replaced while building it.

### Exit 2 whenever any source is unreachable

Rejected because an entry that drifted would then report "could not answer" whenever an unrelated leaf was down, hiding the one finding that needs action. Replaced by the worst verdict by severity, under which `changed` (3) outranks `unreachable` (2).

### A duckdb file database and a parquet read as the fixture

Rejected because it has nothing to check. duckdb is a memory backend, so its tables are materialized into the archive (`database_tables/`) at build time, and a default build relocates the read into `reads/`. Both leaves come out bundled. Replaced by a sqlite table and a read built with `relocate_reads=False`.

### Probe bundled reads while the archive is extracted

Rejected because a bundled file is content-addressed: its name is its hash, so comparing it can only return `equal` unless the archive is corrupt, and a corrupt archive is `xorq catalog check`'s job. Dropping it also removed any need to extract the archive: leaves are read from the build record by plain traversal.

### `no external sources` as a bare string

Rejected because the default build rules send most entries there, and a bare string does not say why. Replaced by the count per bundle kind.

### Write this ADR before building

Rejected because the taxonomy and decision table were settled by building and measuring, not in advance. `check-sources` shipped first, and this ADR adopts its six-row taxonomy and decision table as built rather than re-deriving them, adding the two things that came later: the `unreadable` verdict and the severity order within an exit code.

### Let `check-sources` repair what it finds

Rejected because:
- Detection runs in CI and on schedules against production catalogs; any write there is a write nobody reviewed.
- Repair has failure modes (conflicts, push failures) that detection has no reason to carry.

### Count `table-missing` as drift, or report only exit codes

Rejected because `changed` and `table-missing` share exit 3, so a consumer handed only the code cannot tell a retyped column from a vanished table. The footer collapsed them until #2466 and #2597.

### Record or compare read schemas through the bound backend

Three alternatives were weighed in #2311: record the backend's schema at build time (changes `expr.yaml` and every build hash), normalize type mappings before comparing (weakens the check, and every mapping is a judgement call), or read the live schema from the file header with pyarrow (a reader per format). Rejected in favour of re-running the recording inference, which needs none of them.

### A `load_expr(refresh_schemas=True)` loader mode

Prototyped as #2327 and benchmarked against the rewrite on 2026-09-24. Rejected because:
- A refreshing load hydrates connections without the no-create guard, so it recreated a deleted sqlite database before failing.
- `ibis_yaml` cannot import `catalog`, so the loader had to re-derive which reads are safe to probe and how; it covered only the two `deferred_read_*` methods.
- Rebase sweeps anyway, and the loader probed every source a second time: two snapshots that could disagree.
- It changed the default load path (a dispatch `try`, an `lru_cache` key field, six handlers). The rewrite adds files and leaves loading alone.

The cost difference was milliseconds either way, next to the seconds `catalog.add` spends packaging.

### Rebase in place

Rejected because the old entry is the only record of what ran before, and aliases already give users a stable name that can move.

### Infer renames

Rejected because a rename is indistinguishable from a drop plus an add given two schemas. The explicit form is `--rename`.

## Consequences

### Positive

- `check-sources` is safe to run anywhere: it never writes to a source or the catalog, and a missing database stays missing. Its only local write is fetching archives that are not yet local.
- A rebase can be previewed (`--dry-run`) or built without cataloging (`--no-add`), and since aliases move only when asked, adopting it is a separate, reversible step.
- Undrifted subtrees keep their cache keys across a rebase.

### Negative

- A schema the user *declared* (`schema=` on a deferred read) is compared against inference, so an override that disagreed with inference at build time reads as `changed` on an untouched file. Pinned by `test_a_declared_schema_is_compared_against_inference`.
- A read with no registered inference on an ingesting backend stays unprobed, so an entry built only from such reads can never be rebased past `unprobed`.
- A column added to a source stops at any op that lists its outputs (`Project`, a join's `values`): the refreshed source gains it and the result does not.
- Rebases accumulate superseded archives and cache files. Reclaiming the caches is #2321, blocked on recording every cache key an entry uses; today an entry records only its outermost one.
- The footer does not count `unreachable` or `unreadable` entries, and a drifted entry in the same sweep takes the exit code to 3; only the leaf lines and each entry's `--json` state report them.

## References

- Epic #2293 (`check-sources`): #2294, #2295, #2296, #2297, #2298, #2299, #2300; follow-ups #2310, #2311, #2313, #2314
- Epic #2319 (`rebase`): #2320, #2321, #2322, #2323, #2324, #2325, #2326
- Footer by entry state: #2466, #2597
- Leaf identity in `--json`: #2571; progress on stderr: #2572
- Rejected loader mode: #2327; adopted rewrite: #2337
- How-to: `docs/how_to/check_source_drift.qmd`
