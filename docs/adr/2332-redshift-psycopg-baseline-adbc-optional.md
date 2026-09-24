# ADR-2332: Make the Columnar Redshift driver optional via a psycopg baseline, and ship it as a repackaged wheel

- **Status:** Proposed — its central alternative was tested on 2026-09-24 and
  passed, so the decision below is under review; see kata xorq#nw8d
- **Date:** 2026-09-03
- **Revised:** 2026-09-24, against a live Redshift endpoint
- **Deciders:** dlovell
- **Related:** ADR-0003

## Context

Adding a backend is not by itself an architecture decision — twenty ADRs exist
and none is about a data backend. This one is not really about Redshift. It is
about what xorq does when the best driver for a source is distributed in a way
that no `[project.optional-dependencies]` entry can express.

Every backend with an external driver declares it as a PyPI extra in
`pyproject.toml`. The Columnar ADBC Redshift driver cannot be declared that way
today:

| | |
|---|---|
| Distribution | Not on PyPI — four candidate package names were checked and all 404, and Columnar publishes no driver as a wheel. The *installer*, `dbc`, is on PyPI; the driver it fetches is not |
| Environment | `dbc install --level` accepts only `user` and `system`. There is no environment level |
| Platforms | Builds exist for `linux_amd64`, `linux_arm64`, `macos_arm64` and `windows_amd64`. **No `macos_amd64`** — for any version |

**The environment row is the load-bearing one**, and it is not the row that is
usually cited. A `dbc`-installed driver is machine-global: it cannot be captured
in `uv.lock`, cannot be reconstructed by `uv sync`, and two projects on one
machine cannot pin different driver versions. "Not on PyPI" is a fact about one
index that Columnar could retire tomorrow by publishing anywhere; the
environment defect is a property of `dbc`'s install model and survives that.

The irony is sharp: `adbc_driver_manager` already searches
`sys.prefix/etc/adbc/drivers`, but only when running inside a real virtualenv.
The hook `dbc` cannot target is one the driver manager already honours.

Two things that are *not* obstacles, both of which earlier drafts of this
decision treated as though they were:

- **The driver is not closed-source, and its licence permits redistribution.**
  It is source-visible, and ships under the Permissive Binary License v1.0,
  which opens "Redistribution and use in binary form, without modification, are
  permitted provided that the following conditions are met". The conditions are
  notice reproduction, no reverse engineering, dependency-file inclusion, and no
  endorsement use. Downloads are ungated — no account, key or trial — and
  Redshift is not among the drivers Columnar gates.
- **The macOS gap is narrower than "no macOS build" and is the registry norm.**
  Apple Silicon is covered by a Developer-ID-signed binary present in every
  version. Only Intel Macs lack one — as do 14 of the 20 drivers in the same
  registry, *including bigquery, which xorq already ships*. xorq has already
  accepted this exact constraint once.

Optionality is only worth discussing if a PyPI-installable driver can do the
job at all, and for Redshift the open question was authentication: IAM mints a
temporary password through `redshift-serverless:GetCredentials`, and whether a
generic PostgreSQL client can present one was unknown. A live IAM rig, since
torn down, settled it: psycopg connected with such a password and queried
successfully against a namespace with no admin password
(`adminPasswordSecretArn: null`), so the connection can only have gone through
IAM.

## Decision drivers

- A backend must be installable by `uv sync`, and the resulting environment must
  be reproducible from the lockfile.
- Platform coverage must not silently narrow.
- The driver's Arrow-native path is a real win and should stay reachable.
- No user should get a broken install because an accelerator is unavailable.

## Decision

Two parts, and the second only makes sense because of the first.

**1. psycopg is the baseline.** Connect, DDL, introspection and query all work
with psycopg alone. Every driver-backed path degrades to it.

**2. The accelerator ships as a repackaged platform wheel**, not as an
out-of-band `dbc install`. xorq builds `xorq-adbc-driver-redshift` wheels that
bundle the unmodified Columnar shared library and resolve its absolute path at
import, for the platforms upstream builds.

| Path | Driver | Role |
|---|---|---|
| connect, DDL, introspection, query | psycopg | baseline, PyPI-installable |
| `to_pyarrow_batches` | Columnar ADBC | accelerator, degrades |
| `read_record_batches` (ingest) | psycopg | baseline, dispatching on driver availability |

### Why a wheel rather than `dbc install`

The wheel is what makes the accelerator a *declarable, lockable* dependency
instead of a machine-global side effect. It is also not a novel mechanism:
`adbc_driver_snowflake` is a PyPI wheel that ships its Go shared library inside
the Python package and hands the driver manager an absolute path, and xorq
already depends on three ADBC drivers packaged exactly that way.

Path resolution must happen **in Python at import**, not through a shipped
manifest. A driver manifest carrying a relative path does not work — resolution
falls through to a bare `dlopen` and fails — and a wheel cannot write an
absolute-path manifest at build time, because the install prefix is not known
then and wheels have no post-install hook. The manifest inside Columnar's own
tarball is additionally not spec-compliant; `dbc` translates it on install.

Two build hazards, both silent, both already hit:

- Build backends that honour VCS ignore rules will happily build a wheel with
  **no driver in it** — exit 0, no warning, a few kilobytes. CI must assert the
  payload's presence and size, not merely that the build succeeded.
- A wheel carrying a native library must not be tagged `purelib`. Purity is a
  build-time value, so it needs a build hook; left alone the wheel mistags and
  still installs, which is why this goes unnoticed.

**Publishing to a public index is gated on asking Columnar first.** The licence
permits redistribution, but publishing another company's binary under an
`xorq-*` name is a courtesy call at minimum, and the better outcome is that they
publish it themselves. Until that conversation happens, the wheels are built and
consumed privately. This ADR decides the *mechanism*, not the publication.

If the wheel path is ever abandoned, the fallback is already precedented in this
repo: `dbc install <driver>` plus `driver="<name>"`, exactly as the bigquery
backend does, with the CI recipe in
`.github/workflows/ci-test-bigquery.yml` transferring directly.

### Every driver-backed path needs a baseline, and one did not have one

Optionality is a claim that has to hold for *each* path, and on the postgres
backend it does not: `read_record_batches` is an unconditional ADBC ingest call
with no psycopg branch. Nor is that fixed by installing the driver, because the
driver's ingest is `COPY`-from-S3 underneath and refuses to run without a
staging bucket: `INVALID_STATE: [redshift] Must set redshift.ingest.bucket to
ingest data`.

v1 therefore ingests through a psycopg temp-table + `INSERT` round-trip, which
was exercised and works. `COPY`-from-S3 stays out of scope for v1 **because the
baseline does not need it**, rather than v1 shipping with a scope list that
contradicts its own implementation. Two constraints follow:

1. Dispatch on driver availability from the outset, so an ADBC ingest branch
   arrives later as an addition rather than a restructuring.
2. `redshift.ingest.bucket` would be a non-secret `do_connect` kwarg, so it
   lands in the build hash. Adding it later changes hashes only for users who
   set it — acceptable, but recorded so its arrival is not a surprise.

**Revised 2026-09-24: psycopg `INSERT` is not a baseline an accelerator later
replaces. It is the only ingest that works.** Measured live, the *other* ADBC
driver cannot ingest into Redshift either, and it fails earlier and for a
different reason than Columnar's does — `adbc_driver_postgresql` ingests by
`COPY … FROM STDIN WITH (FORMAT binary)`, and Redshift's `COPY` reads from S3
only:

    INVALID_ARGUMENT: [libpq] COPY query failed:
    ERROR: syntax error at or near "STDIN"    SQLSTATE: 42601

That is a *syntax* error, not the `INVALID_STATE: Must set
redshift.ingest.bucket` above. No bucket setting reaches it, and the driver has
no `INSERT` fallback.

This turns constraint 1 into a live defect rather than a piece of foresight.
Dispatching ingest on driver availability was meant to let an ADBC branch arrive
as an addition; because `_adbc_unavailable_reason()` answers `None` on every
credentialed install, it instead selects the branch that cannot work and leaves
the one that does as dead code. One predicate is serving two paths whose correct
answers are opposite. Filed as **kata xorq#v68a**; the fix is a separate ingest
predicate, false for any driver that ingests by `COPY`.

### Degrading must distinguish "driver absent" from "driver failed"

The postgres backend tries ADBC first in `to_pyarrow_batches` and falls back to
psycopg on *any* exception; its own comment acknowledges this swallows genuine
errors. Under a mandatory driver that is a wart. Under an optional one it is the
mechanism itself: absent-driver is now normal and permanent on Intel Mac, so a
blanket `except` makes every real failure — an expired credential, a permission
error — indistinguishable from it, and the operator sees a slow query instead of
an error. The fallback must catch driver-absence specifically.

## Alternatives considered

### `dbc install redshift`, out-of-band, as the primary mechanism

The pattern the bigquery backend uses today.

Rejected as *primary*, retained as fallback. It cannot be captured in
`uv.lock`, so the environment is not reproducible and two projects on one
machine cannot pin different driver versions. It also requires an out-of-band
step before the accelerator exists at all.

### Make the driver mandatory

Ship one Arrow-native code path.

Rejected: darwin x86_64 would be dropped outright, and an accelerator that is
unavailable on a supported platform cannot be a hard requirement. Before the
rig ran this would also have been forced rather than chosen; psycopg
authenticating is what made it a choice.

### Declare the upstream driver in `[project.optional-dependencies]`

Rejected because no requirement string resolves — the driver payload is not on
any index. This is the alternative the repackaged wheel exists to synthesise.

### Use `adbc_driver_postgresql` as the accelerator — **TESTED 2026-09-24, and it supersedes most of this decision**

Redshift speaks the PostgreSQL wire protocol, and `adbc-driver-postgresql` is
already declared in the `postgres` extra, is already what the inherited
`to_pyarrow_batches` reaches for, and publishes PyPI wheels covering *more*
platforms than the Columnar driver — including the Intel-Mac build Columnar does
not provide for any version.

This was recorded as untested because the rig had been destroyed. A second rig
settled it. Measured against live Redshift Serverless with
`adbc_driver_postgresql` 1.11.0, the driver connects and correctly identifies
the server — `vendor_name = "Redshift"`, `vendor_version = "1.0.434008"` — and
every read path passes:

| path | result |
|---|---|
| `SELECT` → Arrow table | 12 rows |
| `fetch_record_batch` (what `to_pyarrow_batches` streams) | 12 rows |
| `adbc_get_table_schema` | full 11-column schema |
| `adbc_get_objects` (catalog listing) | passes |
| `to_pyarrow_batches` end to end | 12 rows, values matching psycopg |
| a `numeric` column through that path | → `decimal128(5, 4)` |

**The prediction in the paragraph this replaces was exactly backwards.** The
fear was that this driver would fail on Redshift's incomplete `pg_catalog`, the
way `CURRENT_SCHEMA` did. ADBC's introspection is fine. The `pg_catalog`
failures on Redshift are all in *xorq's own psycopg path* — `pg_my_temp_schema()`,
`pg_catalog.pg_enum` and `CREATE TEMPORARY VIEW`, which between them make
`con.table()` and `con.sql()` unreachable — and those are kata xorq#1szn, fixed
on a branch stacked on this one.

The `numeric` case deserves naming because it looked like the likely failure and
is not one. ADBC returns Redshift `numeric` as
`extension<arrow.opaque[storage_type=string, type_name=numeric, vendor_name=Redshift]>`,
and `to_pyarrow_batches`' per-batch cast to the ibis schema converts it to
`decimal128(5, 4)` with values identical to the psycopg branch.

So for the **read** accelerator the repackaging decision above is unnecessary,
the platform gap closes, and the accelerator becomes an ordinary declared
dependency — the "strictly better outcome" this section was written to hold open.
What the measurement does **not** settle is whether the Columnar driver is
materially *faster* than `adbc_driver_postgresql` on Redshift, which is the only
thing the wheel would still buy; this ADR asserts the Arrow-native path is a win
against *psycopg*, and never against the other ADBC driver. Whether that retires
the wheel outright or demotes it to a documented fallback is filed as a decision
rather than taken here: **kata xorq#nw8d**.

For **ingest** it changes nothing, because neither ADBC driver can ingest into
Redshift at all — see the section above.

### Do not offer an accelerator at all

Rejected. The Arrow-native path is a genuine performance win, and the wheel
makes it available without compromising installability on any platform that has
a build.

## Implementation status

This ADR is `Proposed`, and the consequences below are **targets, not
descriptions**. As of this revision the psycopg baseline is written for every
path this ADR names — connect, DDL, introspection, query and ingest — and what
remains unbuilt is the accelerator and its packaging. "Written" is deliberately
weaker than "reachable": see the second caveat below.

The table's verdicts were revised on 2026-09-24 against a live endpoint. Two
rows that read "implemented" did not survive contact with one.

| Decision | Status |
|---|---|
| psycopg baseline for connect/DDL/introspection/query | **partially implemented** — connect and query hold live; *introspection does not*. `get_schema` reads `pg_catalog.pg_enum` and `pg_my_temp_schema()`, `_get_schema_using_query` builds a `CREATE TEMPORARY VIEW`, and Redshift has none of the three, so `con.table()` and `con.sql()` are both unreachable. Fixed by kata xorq#1szn, on a branch stacked on this one |
| `redshift` extra so the backend installs with `uv sync` | **partially implemented** — the extra exists and mirrors `postgres`; `boto3` still undeclared |
| psycopg `read_record_batches` (the ingest baseline) | **implemented, and not reached** — the `CREATE TABLE` + parameterised `INSERT` works live, but the dispatch selects ADBC on every credentialed install and ADBC ingest cannot run on Redshift. kata xorq#v68a |
| `to_pyarrow_batches` discriminating driver-absent from auth-failed | **implemented** — availability is decided before connecting, so a rejected credential propagates. Verified live over both branches: 12 rows each, agreeing on values including `numeric` |
| `xorq-adbc-driver-redshift` wheels and the CI that asserts payload presence | **not implemented** — referenced nowhere in `pyproject.toml` or the workflows |

The first caveat is now discharged, and it is worth keeping the question
visible because the answer split. Dispatch is on the driver being *installed and
credentialed*, which is not the same as its being known to work; the only ADBC
driver that can satisfy the check today is `adbc_driver_postgresql`. A live
endpoint settled it on 2026-09-24: that driver **does** work against Redshift for
reads, and **cannot** work for ingest. "Accelerator available" is therefore a
true statement about Redshift for one path and a false one for the other, which
is precisely why a single predicate cannot serve both.

The second caveat stands, and the live run sharpened it: **no supported install
reaches the psycopg baseline.** The `redshift` extra mirrors `postgres`, which
pins `adbc-driver-postgresql` alongside `psycopg`, so `_adbc_unavailable_reason()`
returns `None` for any credentialed install and the psycopg branch is dead code
there. `postgres/__init__.py` and `postgres_utils.py` also import ADBC at module
scope, so importing the backend requires the driver whatever the extras say.

When this was written the consequence was thought to be an untested branch. It is
worse: on the read path the live branch is the right one, and on the ingest path
the live branch is the one Redshift rejects outright. A dead branch that is also
the only correct one is not an untested path, it is an outage.

Stated, not resolved. The fix is either an extra that installs psycopg without
ADBC, or dropping the fallback — a decision this ADR does not make. The dispatch
seam is written so either answer changes the extras or
`_adbc_unavailable_reason()`, not the Arrow paths. What the live run adds is that
ingest must not consult that seam at all (kata xorq#v68a).

## Consequences

### Positive

- The backend installs with `uv sync`, and the accelerator is lockable rather
  than machine-global.
- No supported platform gets a broken install; Intel Mac degrades to the
  baseline.
- The Arrow-native path reaches Linux, Apple Silicon and Windows users without
  an out-of-band step.
- `boto3` gets declared. It is currently undeclared repo-wide despite being
  imported by the catalog's S3 utilities.
- The pattern generalises to the next source whose driver is not on PyPI.

### Negative

- **xorq becomes a redistributor of a third party's binary**, with the
  maintenance tail that implies: a wheel per platform, a version-drift watcher,
  and signature verification at build time. This is the real cost of the
  decision and it is ongoing.
- Two read paths exist, so both need testing and the boundary between them is a
  real source of bugs — which is why the blanket `except` is not inherited.
- Ingest is row-oriented `INSERT` in v1, slower than `COPY`-from-S3 for large
  loads. **Revised 2026-09-24:** and it has no ADBC successor waiting. Both ADBC
  drivers ingest by `COPY`, which on Redshift means from S3, so `INSERT` is the
  only driver-independent ingest there is until the staging-bucket work lands.
- ~~Intel Macs get no accelerator~~, and neither does an x86_64 Python running
  under Rosetta on Apple Silicon — Rosetta translates x86_64 to arm64, not the
  reverse, so an arm64 dylib cannot be loaded either way. **Revised 2026-09-24:**
  this consequence followed from the Columnar driver being the only accelerator.
  `adbc_driver_postgresql` builds for `darwin x86_64` and is now shown to work
  against Redshift, so on the read path it does not follow. It returns only if
  xorq#nw8d keeps the Columnar wheel as the accelerator of record.
- Performance depends on platform, so two users on identical code can see
  materially different throughput.

## References

- `python/xorq/backends/postgres/__init__.py` — the backend subclassed, and the
  ADBC-first fallback discussed above
- `python/xorq/common/utils/adbc_utils.py` — the bulk-ingest capability probe,
  not reached when the driver fails to load
- `.github/workflows/ci-test-bigquery.yml` — the out-of-band `dbc install`
  fallback pattern, already in use for another backend
- [ADR-0003](0003-optional-git-annex-backend.md) — making an external
  dependency optional behind an abstraction
- kata xorq#v68a — ingest dispatches to an ADBC `COPY` Redshift cannot run;
  its body carries the 2026-09-24 measurement for the ingest paths
- kata xorq#nw8d — the open decision this revision creates: does the tested
  alternative retire the repackaged wheel; its body carries the 2026-09-24
  measurement for the read paths
- kata xorq#1szn — the psycopg introspection failures that make `con.table()`
  unreachable on Redshift
- kata xorq#df8e — the endpoint this revision was measured against, and its
  teardown gate
