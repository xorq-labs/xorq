# ADR-XXXX: Detect source drift read-only by verdict; repair it by deriving a new entry

- **Status:** Accepted
- **Date:** 2026-11-09
- **Deciders:** Daniel Mesejo
- **Related:** ADR-0006, ADR-0012, ADR-0015

## Context

A catalog entry records the schema of every source it was built over: each database table and each file read. Upstream sources change after the build. A column is added, retyped or dropped, a table is renamed, a file moves. The entry still loads and still runs, against a world it was not built for, and the first sign is usually a wrong result or a failure deep inside an execution.

Two questions follow, and they have different risk profiles:

1. **Has a source drifted?** (`xorq catalog check-sources`, epic #2293.) Users run this on production catalogs, in CI, and on a schedule. It must be safe to run anywhere, at any time.
2. **Can the entry follow the drift?** (`xorq catalog rebase`, epic #2319.) This has to re-derive an expression without the author's build script, which the catalog never had.

Reasonable people could disagree on most of the choices below: whether detection may repair, whether a missing source is "drift", which engine a recorded schema is compared with, whether rebase edits the entry in place or creates a new one, and where re-derivation hooks into loading. This ADR records what was chosen, as built.

## Decision drivers

- Looking must never change anything: no writes to the user's database, no database file created by looking for one, no cataloged state changed.
- A script must be able to branch on the exit code alone, and the code must never contradict the text printed beside it.
- A verdict of `changed` must mean the source changed, never that two tools disagree about an untouched source.
- A rebase must never lose the entry it started from, and a failed rebase must write nothing.
- Nothing is guessed: a rename looks exactly like a drop plus an add, given two schemas.

## Decision

### Detection and repair are separate commands

`check-sources` only reads. `rebase` is the only command that writes, and it runs the same sweep as its first step (`EntryReport` in `python/xorq/catalog/drift.py`), so both commands see one answer about the same sources. `rebase --json` embeds the entry's `check-sources --json` document under `drift`, built by the same code.

### A closed set of verdicts; the exit code is a property of the verdict

Each probed source (a *leaf*) gets one `Verdict` (`python/xorq/catalog/enums.py`):

| Verdict | Meaning | Exit |
|---|---|---|
| `equal` | live schema matches the recorded one | 0 |
| `unreachable` | the backend could not be asked | 2 |
| `unreadable` | the record itself could not be read | 2 |
| `changed` | live and recorded schemas differ | 3 |
| `table-missing` | the backend answered and lists no such table, or the file path no longer resolves | 3 |

`Verdict.exit_code` is a `match` over every member with no catch-all, so a new member fails loudly instead of inheriting a code. An entry's state is the worst of its leaves by `Verdict.severity`, a total order that refines the exit code: within code 2, `unreadable` beats `unreachable` (retrying will not fix a record that does not parse); within code 3, `table-missing` beats `changed` (the source is gone, not different). The order is declared once, as the member order, so the same catalog rolls up to the same state whatever order the names were typed in. Exit 2 means retryable infrastructure; exit 3 means the world moved.

The human footer and `--json` read the same per-entry state: the footer counts `N entries, D drifted, M missing` by that state, and the exit code is the roll-up of the same states (#2597).

### Missing needs positive evidence

`table-missing` is reported only when the backend answered and its table list does not contain the name, or when a file path no longer resolves. Any other failure is `unreachable`. A probe never infers absence from an error message.

### Probing never writes

- A sqlite database is opened with a no-create guard, so a deleted database file is reported, not recreated empty.
- A profile that failed to connect is not dialled again in the same sweep; connections are shared per profile through one cache.
- A `Read` on an ingesting backend (sqlite, postgres, databricks) is never replayed: its `read_*` is a write with `mode="replace"`. `is_checkable` keeps such reads unprobed unless they can be answered without the backend (next section).
- A relocated read whose bytes were bundled into the archive is exempt: it cannot drift.

Leaves that cannot be probed are named in the output as `unchecked`, never silently treated as `equal`. An entry whose only sources are unchecked has no verdict (`state: null` in `--json`), rather than claiming `equal` on no evidence.

### A read is compared using the tool that recorded it

`deferred_read_parquet` records the schema datafusion infers and `deferred_read_csv` the one pandas infers, whatever backend the read is bound to. The probe re-runs that same inference (`get_read_inference` in `python/xorq/catalog/drift.py`, reading `DEFAULT_READ_INFERENCE` in `python/xorq/common/utils/defer_utils.py`). Both sides come from one producer, so a backend's own type mapping (sqlite reading `int32` as `int64`, `date32` as `string`) is never reported as drift (#2311, pinned by #2574). As a side effect, such reads need no connection at all and are probed on ingesting backends too.

### Rebase derives a new entry and never mutates the old one

A rebase builds a new archive with a new hash, records its ancestor as `rebased_from` in the archive's build metadata (`python/xorq/catalog/constants.py`), and advances aliases to it. The old entry is untouched. Ancestry lives in the archive, not only in the catalog sidecar, because a catalog copy re-derives sidecars from archives.

Every refusal comes before the first write. A conflict writes nothing. A write that fails after starting is rolled back.

### Re-derivation is a rewrite after loading, driven by the drift report

The build loads exactly as recorded (`load_expr` is untouched). `refresh_build` in `python/xorq/catalog/refresh.py` then rebuilds the loaded graph over the live schemas the sweep found:

- Only sources reported `changed` move. An `equal` source keeps its node, so undrifted subtrees and their cache keys stay put.
- Every op above a moved source is recreated through `__recreate__`, which re-runs its own signature validation. A dropped column fails at the `Field` that names it, and a column retyped to string fails at the `Mean` over it. There is no second validator to fall out of step with the real semantics.
- A probed source is matched to its loaded node by the connection profile's content hash, table name, snapshot pin and recorded schema, not by backend name, so one table name on two connections is two sources.
- The loader opens its own connections. The sweep's connections carry the no-create guard's URI options, which would otherwise leak into the rebased build's recorded profile.

### Rebase outcomes and exit codes

Rebase reuses 0 and 2 with the meaning `check-sources` gives them, and adds its own (`RebaseStatus`, `RebaseExit`, `RebaseFailure` in `python/xorq/catalog/enums.py`):

| Exit | Outcomes |
|---|---|
| 0 | `noop` (no drift), `unprobed` (no source could be probed; stderr says drift was not ruled out), `rebased` |
| 1 | refused (pinned entry, some but not all sources unprobeable, Python-minor mismatch, alias errors) or a failed write |
| 2 | a source unreachable or unreadable; nothing written |
| 3 | derived but not cataloged: `--dry-run` (`preview`) or `--no-add` (`built`) |
| 4 | `conflict`: an op no longer fits its inputs, or a source is gone; names `failing_op`, the sources involved, and what each lost and gained |
| 5 | `push-failed`: rebased locally, the sync's push failed |

An entry with only some sources unprobeable is refused rather than rebased, because those sources would come back unchanged beside refreshed ones.

### Renames are explicit

Rebase never guesses a rename. A renamed column is a removed one unless `--rename SOURCE OLD NEW` says which live column it is now; ops above the source keep seeing `OLD`, and a later rebase of the result keeps following the rename.

## Alternatives considered

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

- `check-sources` is safe to run anywhere: it reads, and a missing database stays missing.
- Scripts branch on exit codes that cannot contradict the printed verdicts or the `--json` state.
- A rebase can be previewed (`--dry-run`), built without cataloging (`--no-add`), and undone by moving an alias back.
- Undrifted subtrees keep their cache keys across a rebase.

### Negative

- A schema the user *declared* (`schema=` on a deferred read) is compared against inference, so an override that disagreed with inference at build time reads as `changed` on an untouched file. Pinned by `test_a_declared_schema_is_compared_against_inference`.
- A read with no registered inference on an ingesting backend stays unprobed, so an entry built only from such reads can never be rebased past `unprobed`.
- A column added to a source stops at any op that lists its outputs (`Project`, a join's `values`): the refreshed source gains it and the result does not.
- Rebases accumulate superseded archives and cache files. Reclaiming the caches is #2321, blocked on recording every cache key an entry uses; today an entry records only its outermost one.
- The footer does not count `unreachable` or `unreadable` entries; the leaf lines and the exit code report them.

## References

- Epic #2293 (`check-sources`): #2294, #2295, #2296, #2297, #2298, #2299, #2300; follow-ups #2310, #2311, #2313, #2314
- Epic #2319 (`rebase`): #2320, #2321, #2322, #2323, #2324, #2325, #2326
- Footer by entry state: #2466, #2597
- Leaf identity in `--json`: #2571; progress on stderr: #2572
- Rejected loader mode: #2327; adopted rewrite: #2337
- How-to: `docs/how_to/check_source_drift.qmd`
