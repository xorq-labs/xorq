# ADR-2332: Make the Redshift ADBC accelerator optional over a psycopg baseline

- **Status:** Accepted
- **Date:** 2026-09-25
- **Deciders:** dlovell
- **Related:** ADR-0003

## Context

This ADR is about what xorq does when the best driver for a source is
distributed in a way no `[project.optional-dependencies]` entry can express.

Backends declare their Python drivers as PyPI extras in `pyproject.toml`; the
drivers installed out of band today are bigquery's and databricks', by `dbc
install` (see *Alternatives*). The Columnar ADBC Redshift driver cannot be declared as an
extra:

| | |
|---|---|
| Distribution | Not on PyPI, and Columnar publishes no driver as a wheel. The *installer*, `dbc`, is on PyPI; the driver it fetches is not |
| Environment | `--level` accepts only `user` and `system`. With it omitted, `dbc install` writes to the first of `$ADBC_DRIVER_PATH`, `$VIRTUAL_ENV/etc/adbc/drivers` and `$CONDA_PREFIX/etc/adbc/drivers` that is set, else the user level. No lockfile records it |
| Platforms | Builds exist for `linux_amd64`, `linux_arm64`, `macos_arm64` and `windows_amd64`. **No `macos_amd64`** — for any version |

**The environment row is the load-bearing one**, and it is not the row that is
usually cited. A `dbc`-installed driver sits outside the Python dependency
graph: it cannot be captured in `uv.lock` and cannot be reconstructed by
`uv sync`. It can land inside a virtualenv — `adbc_driver_manager` searches
`sys.prefix/etc/adbc/drivers` there, and `dbc` targets that directory when
`$VIRTUAL_ENV` is set — but only as an out-of-band step, whose destination is
decided by the environment variables of the shell that ran it rather than by the
project. "Not on PyPI" is a fact about one index that Columnar could retire
tomorrow by publishing anywhere; the lockability defect is a property of `dbc`'s
install model and survives that.

Optionality is only worth discussing if a PyPI-installable driver can do the job
at all, and for Redshift the open question was authentication: IAM mints a
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

**1. psycopg is the baseline.** Connect, DDL and query run over psycopg, and the
read path degrades to it when the driver is *absent*, never when it fails. The
two read branches build Arrow differently — the accelerator casts
each fetched batch to the ibis schema, the baseline builds a record batch from a
struct array — so a defect in one cast can surface on one branch and not the
other. The alias failure below is exactly that.

**2. The accelerator is `adbc_driver_postgresql`.** Redshift speaks the
PostgreSQL wire protocol. That driver is already declared in the `postgres`,
`redshift` and `examples` extras, already in the lockfile, and already what the inherited
`to_pyarrow_batches` reaches for. It publishes wheels for more platforms than
Columnar, Intel Mac included, so the accelerator is an ordinary declared
dependency with no packaging work behind it.

| Path | Driver | Role |
|---|---|---|
| connect, DDL, introspection, query | psycopg | baseline, PyPI-installable |
| `to_pyarrow_batches` | `adbc_driver_postgresql` | accelerator, degrades |
| `read_record_batches` (ingest) | psycopg | the only ingest that works; **must not dispatch on the read-path availability predicate** |

Measured against live Redshift Serverless, the driver connects and correctly
identifies the server — `vendor_name = "Redshift"` — and the reads exercised all
passed: `SELECT` to an Arrow table, `fetch_record_batch`, `adbc_get_table_schema`,
`adbc_get_objects`, and `to_pyarrow_batches` end to end, with values matching
psycopg. None of them carried an auto-generated alias, which is why this does not
contradict the alias failure under *Implementation status*. It is a record of what
was run, not a claim that every read works.

The `numeric` case deserves naming because it looked like the likely failure and
is not one. ADBC returns Redshift `numeric` as
`extension<arrow.opaque[storage_type=string, type_name=numeric, vendor_name=Redshift]>`,
and `to_pyarrow_batches`' per-batch cast converts it to the ibis schema's type —
`decimal128(5, 4)` for the column measured — with values identical to psycopg's.

The fear this measurement retired was the opposite of what happened. The worry
was that a PostgreSQL-targeted driver would fail on Redshift's incomplete
`pg_catalog`. ADBC's introspection is fine; the `pg_catalog` failures on
Redshift are all in xorq's own psycopg path.

### Ingest has no accelerator, and must not pretend otherwise

Optionality is a claim that has to hold for *each* path, and on the postgres
backend it did not: `read_record_batches` is an unconditional ADBC ingest call
with no psycopg branch. Installing a driver does not fix it. Neither ADBC driver
can ingest into Redshift at all, and they fail differently:

- The Columnar driver's ingest is `COPY`-from-S3 underneath and refuses to run
  without a staging bucket: `INVALID_STATE: [redshift] Must set
  redshift.ingest.bucket to ingest data`.
- `adbc_driver_postgresql` ingests by `COPY … FROM STDIN WITH (FORMAT binary)`,
  and Redshift's `COPY` reads from S3 only:

      INVALID_ARGUMENT: [libpq] COPY query failed:
      ERROR: syntax error at or near "STDIN"    SQLSTATE: 42601

  That is a *syntax* error, not the bucket error above. No bucket setting
  reaches it, and the driver has no `INSERT` fallback.

So psycopg `INSERT` is not a baseline an accelerator later replaces. It is the
only ingest there is, and `COPY`-from-S3 stays out of scope **because the
baseline does not need it**, rather than the backend shipping a scope list that
contradicts its own implementation.

This is why the ingest row above carries a prohibition. Dispatching ingest on
driver availability was meant to let an ADBC branch arrive later as an addition
rather than a restructuring. It did the opposite: the availability
predicate reports the accelerator available on every connection given a
password, so dispatching that way selected the branch that cannot work.
One predicate cannot serve two paths whose correct answers are opposite, so
ingest no longer consults that predicate at all.

`redshift.ingest.bucket`, if it is ever added, would be a non-secret
`do_connect` kwarg, so it lands in the build hash. Adding it later changes
hashes only for users who set it — acceptable, but recorded so its arrival is
not a surprise.

### Degrading must distinguish "driver absent" from "driver failed"

The postgres backend tried ADBC first inside `to_pyarrow_batches` and fell back
to psycopg on *any* exception; its own comment acknowledged this swallows
genuine errors. Under a mandatory driver that is a wart. Under an optional one
it is the mechanism itself: a blanket `except` makes every real failure — an
expired credential, a permission error — indistinguishable from an absent
driver, and the operator sees a slow query instead of an error.

The fallback must therefore catch driver-absence specifically, and it does at
the connect stage: `_open_adbc_conn_or_none` decides availability before
dialling and does not wrap the connect, so a rejected credential propagates.

**This rule governs the connect stage; execute keeps one deliberate catch.**
`to_pyarrow_batches` is inherited with an `except ADBCProgrammingError` around
`cur.execute` that re-runs the query on psycopg. It is what reads session-local
temporary tables, which the fresh ADBC connection cannot see and which a
`temporary=True` ingest creates, so it cannot simply be removed. Its cost: a
server-side SQL error at execute is retried on psycopg without notice (and raises
there if genuine), and an ADBC-specific one is masked.

## Alternatives considered

### Repackage the Columnar driver as a platform wheel

Build `xorq-adbc-driver-redshift` wheels bundling the unmodified Columnar shared
library, resolving its absolute path at import, for the platforms upstream
builds. This would make the accelerator a declarable, lockable dependency
instead of an out-of-band install step, and it is not a novel mechanism:
`adbc_driver_snowflake` is a PyPI wheel shipping its Go shared library inside
the Python package and handing the driver manager an absolute path, and xorq
already depends on several ADBC drivers packaged that way (`pyproject.toml` is
the list).

Rejected. Every cost it carries is paid in advance for its only remaining
advantage over `adbc_driver_postgresql` — speed — which has never been measured.
Those costs are real: xorq becomes a redistributor of a third party's binary,
with a wheel per platform, a version-drift watcher and signature verification at
build time. Publishing under an `xorq-*` name would also be a courtesy call to
Columnar at minimum, and the better outcome is that they publish it themselves.
A measured speed margin would be an input to a *new* decision, not a trigger that
resumes this one.

### `dbc install redshift`, out-of-band, as the primary mechanism

The pattern the bigquery and databricks backends use today.

Rejected as *primary*. It cannot be captured in `uv.lock`, so the environment is
not reproducible from the lockfile, and the installed driver version is whatever
the last out-of-band run left. It also requires that step before the
accelerator exists at all.

It remains a precedent from *other* backends, not a working Redshift path: no
code here accepts a `driver=` name, and the availability probe looks for
`adbc_driver_postgresql` alone.

### Make the driver mandatory

Ship one Arrow-native code path.

Rejected: an accelerator that is unavailable on a supported platform cannot be a
hard requirement. Before the IAM rig ran this would also have been forced rather
than chosen; psycopg authenticating is what made it a choice.

### Do not offer an accelerator at all

Rejected. The Arrow-native path avoids psycopg's per-row Python conversion, and
`adbc_driver_postgresql` offers it with no packaging work and no platform gap.
The speed margin is expected, not measured: no benchmark against psycopg on
Redshift is recorded, and none against the Columnar driver exists either.

## Implementation status

The consequences below are targets rather than descriptions, and "written" is
deliberately weaker than "reachable" — see the second caveat.

- **psycopg baseline for connect, DDL, introspection and query** — connect and
  query hold live; *introspection does not*. Three `pg_catalog` constructs
  Redshift does not provide were each raised live and separately observed:
  `pg_my_temp_schema()`, which `get_schema` calls when no database is passed;
  `pg_catalog.pg_enum`, which an explicit database routes onto instead; and
  `CREATE TEMPORARY VIEW`, which schema inference from a query needs. So
  `con.table()` fails either way, and the one shipped route to a bound table
  expression is `con.sql(query, schema=...)`. The live table-bound reads recorded
  below ran with a separate introspection fix merged in. `con.list_tables()` does work — it does not introspect.
- **`redshift` extra so the backend installs with `uv sync`** — the extra exists
  and mirrors `postgres`; `boto3` is still undeclared.
- **psycopg `read_record_batches`** — implemented and now the only ingest path,
  reached unconditionally; still not clean. The `CREATE TABLE` plus parameterised
  `INSERT` lands rows, but the method's tail re-introspects
  (`return self.table(...)`) and raises for the introspection reason above. That
  remaining failure is the introspection gap, not the ingest.
- **Driver-absent distinguished from auth-failed** — implemented at connect only.
  Availability is decided before dialling, so a credential rejected there
  propagates. The inherited execute-stage catch is untouched.
- **`adbc_driver_postgresql` as the accelerator** — implemented by construction:
  already declared in both extras and already what the inherited
  `to_pyarrow_batches` reaches for. What is **not** implemented is its safety
  net: no upper pin, and no live-Redshift CI job. Neither is assigned by this
  ADR, and the pin is not a drive-by — it is shared with every other extra that
  declares the package, so it needs its own decision.

One measured caveat on the read path: it raises `ValueError` when an expression
carries an **auto-generated** alias containing upper case — `t.count().execute()`
is the minimal case — because Redshift folds identifiers to lower case and the
per-batch cast rejects the mismatch. Measured since: Redshift folds the alias
even though xorq emits it **quoted**, and the psycopg branch returns the same
expression cleanly. So this is not evidence about drivers at all. The
discriminator is xorq's cast, which matches on field names, against a baseline
that builds its batches positionally and cannot see a mismatch. Any driver
handing back the folded name meets it identically, the Columnar driver included;
the fix belongs in the cast, and is not part of this change.

A further read-path defect is closed rather than carried: on a connection opened
with `schema=`, the ADBC read connection ran with the server's default
`search_path`, because `PgADBC`'s URI names no schema and only the psycopg
connection runs `_post_connect`'s `set_config`. The compiler emits unqualified
table names, so every table-bound read failed on ADBC and was re-run on psycopg
by the execute-stage catch: correct rows, and an accelerator that never served
one. The Redshift read connection now carries the search path in libpq
`options`. Its consequence is that table-bound reads on such connections now
reach the alias failure above instead of falling back past it. The postgres
backend has the same gap and is untouched here.

The second caveat is that **which read branch serves a query is decided per
connection, not per install.** The availability predicate returns a reason
whenever the connection's `_con_kwargs` carry no password, which is every
backend built by `from_connection` and every connect that leaves the password to
libpq (`PGPASSWORD`, `.pgpass`). Those connections read through psycopg on the
standard `redshift` extra; a connection given a password reads through
`adbc_driver_postgresql`. So the psycopg read branch is live, on a population the
extra does not select, and the same expression can succeed on one connection
and fail on another: the alias case above succeeds through psycopg and raises
through ADBC. `from_connection` skips `do_connect`, so it applies
`prepare_threshold=None` to the given connection itself, and refuses one whose
encoding psycopg cannot decode rather than setting `client_encoding`.

Importing the backend requires an ADBC package — the driver *manager*, not the
driver: `postgres/__init__.py` imports `adbc_driver_manager` at module scope,
while the module that imports `adbc_driver_postgresql` is itself loaded lazily.
The measurement is recorded beside the import guard in
`python/xorq/tests/test_redshift_backend.py`. An install with the
manager but not `adbc-driver-postgresql` (no declared extra produces one; the dev
group's `adbc-driver-sqlite` brings the manager alone) reads every connection through psycopg; one without the manager
cannot import the backend at all.

Ingest is not affected: it consults no predicate, so its psycopg path is the live
one on every connection.

## Consequences

### Positive

- The backend installs with `uv sync`, and the accelerator is lockable rather
  than installed out of band.
- No supported platform gets a broken install, and no platform has to degrade for
  want of an accelerator build — `adbc_driver_postgresql` ships wheels for every
  platform xorq supports, Intel Mac included.
- The Arrow-native path reaches every one of those platforms with no out-of-band
  step, subject to the alias caveat under *Implementation status*.
- The IAM question of *Context* is settled for the baseline: psycopg can present
  a temporary password, so no AWS SDK is needed to *connect*. Minting that
  password stays the caller's job; no backend code reads `boto3`, and `boto3`
  remains undeclared repo-wide despite the catalog's S3 utilities importing it.
  That gap is real but this decision neither causes nor repairs it, and whether
  the declaration belongs to a backend extra or to the catalog is unsettled.

### Negative

- **The accelerator is a bet on wire-protocol compatibility, and the bet is
  unhedged.** `adbc_driver_postgresql` targets PostgreSQL; Redshift's
  compatibility is a courtesy, so a release could regress against Redshift with
  nobody upstream treating it as a bug. Every extra declaring it pins `>=1.4.0`
  with no ceiling (`pyproject.toml` owns the list), so such a release would
  install automatically, and no live-Redshift CI job would catch it — nor would
  one that fails to assert *which* branch served the read, since the fallback
  would mask the regression. Note what that leaves open: the inherited catch
  masks one class of failure — an `ADBCProgrammingError` raised by the query
  execution — and the discrimination above does not cover it. Other stages and
  other error classes propagate. So a regression is masked or not depending on
  how it surfaces, and nothing here tells which; the masking ships, the detector
  does not.
- Migration to another accelerator is cheaper than the wheel but not free. It
  changes the availability predicate, the extras, **and** the connection factory
  the read path calls, which names `adbc_driver_postgresql` directly.
- Two read paths exist, so both need testing and the boundary between them is a
  real source of bugs. The blanket `except` is not inherited, but the narrow
  execute-stage catch is, with the cost described under *Degrading*.
- Ingest is row-oriented `INSERT`, slower than `COPY`-from-S3 for large loads,
  and it has no ADBC successor waiting. Both ADBC drivers ingest by `COPY`, which
  on Redshift means from S3, so `INSERT` is the only driver-independent ingest
  there is until staging-bucket work lands.

## References

- `python/xorq/backends/redshift/__init__.py` — the backend this ADR decides:
  the availability predicate, the connect-stage ADBC probe, and the psycopg
  ingest
- `python/xorq/backends/postgres/__init__.py` — the backend subclassed, and the
  ADBC-first read path discussed above
- `python/xorq/common/utils/adbc_utils.py` — the bulk-ingest capability probe
  the postgres ingest path consults; Redshift's ingest never reaches it
- `.github/workflows/ci-test-bigquery.yml`, `.github/workflows/ci-test-databricks.yml`
  — the out-of-band `dbc install` pattern, already in use for two backends
- [ADR-0003](0003-optional-git-annex-backend.md) — making an external
  dependency optional behind an abstraction
