# ADR-XXXX: Make the Columnar Redshift driver optional via a psycopg baseline

- **Status:** Proposed
- **Date:** 2026-09-03
- **Deciders:** dlovell
- **Related:** ADR-0003

## Context

Adding a backend is not by itself an architecture decision — twenty ADRs exist
and none is about a data backend. This one is not about Redshift. It is about
what xorq does when the best driver for a source **cannot be a Python
dependency at all**, which has not come up before and which the Redshift work
forces.

Every backend with an external driver declares it as a PyPI extra in
`pyproject.toml`. The Columnar ADBC Redshift driver cannot be declared that
way, for three independent reasons:

| | |
|---|---|
| Distribution | Not on PyPI. It 404s. Installation is out-of-band, via the `dbc` tool — `dbc` itself is on PyPI, but the driver it fetches is not |
| License | Closed-source `LicenseRef-PBL` |
| Platforms | No `macos_amd64` build; the flake targets darwin x86_64 |

There is no requirement string that resolves, so "declare it as an optional
dependency" — the shape every other external-driver backend uses — is not
available. The real choice is narrower and worse:

- **Mandatory**: the backend works well, and `uv sync` cannot install it. A
  closed-source binary becomes a hard requirement of an open-source project,
  and a platform xorq supports is dropped.
- **Optional**: the backend must do useful work without the driver, which means
  something else has to be the baseline.

Optionality is only possible if a PyPI-installable driver can actually do the
job. Redshift speaks the PostgreSQL wire protocol and psycopg is already in the
`postgres` extra, so the candidate is obvious — but the question is not
protocol compatibility, it is **authentication**. Redshift Serverless IAM auth
mints a temporary password through
`redshift-serverless:GetCredentials`, and whether a generic PostgreSQL client
can present one was unknown. If it could not, the ADBC driver would be the only
way in and this decision would be forced rather than made.

A live IAM rig was built to settle it, and torn down afterwards. It is
deliberately not part of this repository: it is throwaway AWS infrastructure,
not xorq functionality. Experiment 1: psycopg connected with a
`GetCredentials` temporary password and queried successfully, against a
Redshift Serverless namespace with no admin password at all
(`adminPasswordSecretArn: null`) — so the connection can only have gone through
IAM. `current_user` came back as the assumed role, not a database user.

That is what makes optionality available rather than theoretical.

## Decision drivers

- A backend must be installable by `uv sync` with no out-of-band steps.
- xorq must not take a hard dependency on a closed-source binary.
- Platform coverage must not silently narrow.
- The driver's Arrow-native path is a real win and should stay reachable, not
  be designed out.

## Decision

**A driver that cannot be a PyPI dependency is optional, and the backend has a
PyPI-installable baseline that works without it. For Redshift: psycopg is the
baseline, the Columnar ADBC driver is an optional accelerator, and every path
that uses the driver degrades to the baseline.**

This is ADR-0003's shape. There, git-annex was a system dependency that not
every user should have to install, and the decision was not "how the catalog
works" but "make the dependency optional, behind an abstraction that has a
working implementation without it." The same move: the abstraction is the
existing postgres backend, and psycopg is the implementation that works without
the optional binary.

| Path | Driver | Role |
|---|---|---|
| connect, DDL, introspection, query | psycopg | baseline, PyPI-installable |
| `to_pyarrow_batches` | Columnar ADBC | optional accelerator, degrades |
| `read_record_batches` (ingest) | psycopg | baseline — see below |

The `redshift` extra declares `psycopg` and `boto3`. The driver is never a
declared dependency; it is documented as `dbc install redshift`, and the
missing `macos_amd64` build is stated rather than worked around.

### Every driver-backed path needs a baseline, and one did not have one

This is the substantive consequence, and the reason the decision is not simply
a packaging note. Optionality is a claim that has to be true of *each* path,
and on the postgres backend it is not: `read_record_batches` is an
unconditional ADBC ingest call with no psycopg branch. Inheriting it would mean
the common case — driver absent — is the failing one.

Nor can that be fixed by installing the driver, because the driver's ingest is
`COPY`-from-S3 underneath and refuses to run without a staging bucket:
`INVALID_STATE: [redshift] Must set redshift.ingest.bucket to ingest data`. So
ADBC ingest is not an accelerator that degrades — it drags in an S3 bucket as
required user-facing configuration.

v1 therefore ingests through a psycopg temp-table + `INSERT` round-trip, which
experiment 6 also exercised and which works. `COPY`-from-S3 stays out of scope
for v1 **because the baseline does not need it**, rather than v1 shipping with
a scope list that contradicts its own implementation. Two constraints follow,
cheap now and expensive later:

1. Dispatch on driver availability from the outset, so an ADBC ingest branch
   arrives later as an addition rather than a restructuring.
2. `redshift.ingest.bucket` would be a non-secret `do_connect` kwarg, so it
   lands in the build hash. Adding it later changes hashes only for users who
   set it — acceptable, but recorded so its arrival is not a surprise.

### Degrading must distinguish "driver absent" from "driver failed"

The postgres backend tries ADBC first in `to_pyarrow_batches` and falls back to
psycopg on *any* exception; its own comment acknowledges this swallows genuine
errors. Under a mandatory driver that is a wart. Under an optional one it is
the mechanism itself: the absent-driver case is now normal and permanent, so a
blanket `except` makes every real failure — an expired credential, a permission
error — indistinguishable from it, and the operator sees a slow query instead
of an error. The fallback must catch driver-absence specifically.

## Alternatives considered

### Make the driver mandatory

Ship one Arrow-native code path.

Rejected: `uv sync` could not install the backend, a closed-source
`LicenseRef-PBL` binary would become a hard dependency, and darwin x86_64 would
be dropped. It was only necessary if psycopg could not authenticate, and
experiment 1 showed it can.

### Declare the driver in `[project.optional-dependencies]`

Rejected because it is not possible, not because it is unwise. The package
404s on PyPI, so no requirement string resolves. This is a distribution fact.

### Vendor or repackage the driver so it can be declared

Rejected. Redistributing a closed-source `LicenseRef-PBL` binary is a licensing
question xorq should not answer by doing it, and it would not fix the missing
`macos_amd64` build.

### Support ADBC ingest in v1 with a required `redshift.ingest.bucket`

Deferred, not rejected. It adds an S3 staging bucket as user-facing
configuration for a path that only works when an out-of-band driver is present.
The dispatch point above exists so this can arrive as an addition.

## Consequences

### Positive

- The backend installs with `uv sync` and no out-of-band steps.
- No closed-source binary becomes a hard dependency; no supported platform is
  dropped.
- The Arrow-native read path stays available to anyone who runs
  `dbc install redshift`.
- `boto3` gets declared. It is currently undeclared repo-wide despite being
  imported by the catalog's S3 utilities.
- The pattern generalises: the next source whose best driver is not on PyPI has
  a precedent to follow.

### Negative

- Two read paths exist, so both need testing and the boundary between them is a
  real source of bugs — which is why the blanket `except` is not inherited.
- Ingest is row-oriented `INSERT` in v1, slower than `COPY`-from-S3 for large
  loads.
- Users wanting the accelerated paths must install a driver out-of-band, and
  macOS x86_64 users cannot at all.
- Performance now depends on whether an out-of-band step was taken, so two
  users on identical code can see materially different throughput.

## References

The IAM auth rig that produced the experimental results above was throwaway
infrastructure and is not in this repository. Its findings are recorded in this
ADR and in the Redshift backend plan; the rig itself was destroyed after use.

- `python/xorq/backends/postgres/__init__.py` — the backend subclassed, and the
  ADBC-first fallback discussed above
- `python/xorq/common/utils/adbc_utils.py` — the bulk-ingest capability probe,
  not reached when the driver fails to load
- [ADR-0003](0003-optional-git-annex-backend.md) — making an external
  dependency optional behind an abstraction
