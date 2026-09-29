# ADR-2332: Make the Redshift ADBC accelerator optional over a psycopg baseline

- **Status:** Accepted
- **Date:** 2026-09-28
- **Deciders:** dlovell
- **Related:** —

## Context

This ADR is about what xorq does when the best driver for a source is
distributed in a way no `[project.optional-dependencies]` entry can express.

Backends declare their Python drivers as PyPI extras in `pyproject.toml`; a few
install theirs out of band with `dbc install` (see *Alternatives*). The Columnar
ADBC Redshift driver cannot be declared as an extra:

| | |
|---|---|
| Distribution | Not on PyPI, and Columnar publishes no driver as a wheel. The *installer*, `dbc`, is on PyPI; the driver it fetches is not |
| Environment | `--level` accepts only `user` and `system`. With it omitted, `dbc install` writes to the first of `$ADBC_DRIVER_PATH`, `$VIRTUAL_ENV/etc/adbc/drivers` and `$CONDA_PREFIX/etc/adbc/drivers` that is set, else the user level. No lockfile records it |
| Platforms | Builds exist for `linux_amd64`, `linux_arm64`, `macos_arm64` and `windows_amd64`. **No `macos_amd64`** — for any version |

**The environment row is the load-bearing one.** A `dbc`-installed driver sits
outside the Python dependency graph: it cannot be captured in `uv.lock` or
reconstructed by `uv sync`. It can land inside a virtualenv, but only as an
out-of-band step whose destination the shell's environment decides, not the
project. "Not on PyPI" is a fact about one index that Columnar could change
tomorrow; the lockability defect is a property of `dbc`'s install model and
survives that.

Optionality is only worth discussing if a PyPI-installable driver can do the job
at all, and for Redshift the open question was authentication: IAM mints a
temporary password through `redshift-serverless:GetCredentials`, and whether a
generic PostgreSQL client can present one was unknown. A live IAM rig settled it:
psycopg connected with such a password and queried successfully against a
namespace with no admin password (`adminPasswordSecretArn: null`), so the
connection can only have gone through IAM.

## Decision drivers

- A backend must be installable by `uv sync`, and the resulting environment must
  be reproducible from the lockfile.
- Platform coverage must not silently narrow.
- An Arrow-native read path should stay reachable where a driver offers one.
- No user should get a broken install because an accelerator is unavailable.

## Decision

Two parts, and the second only makes sense because of the first.

**1. psycopg is the baseline.** Connect, DDL and introspection run over psycopg,
and it is the only ingest (below). Reads that produce Arrow (`execute`,
`to_pyarrow`, `to_pyarrow_batches`) use the accelerator when a connection can
have it and psycopg otherwise; *Degrading* says when each happens.

**2. The accelerator is `adbc_driver_postgresql`.** Redshift speaks the
PostgreSQL wire protocol, and that driver is an ordinary PyPI dependency already
declared for the postgres backend, and the `redshift` extra installs it too
(`pyproject.toml`), so on a declared install the accelerator is always present
and "optional" is decided per connection; whether a psycopg-only extra should
exist is left open there. It publishes wheels for more platforms than Columnar, Intel Mac included,
and it is what the inherited `to_pyarrow_batches` already reaches for.

| Path | Driver | Role |
|---|---|---|
| connect, DDL, introspection | psycopg | baseline, PyPI-installable |
| `to_pyarrow_batches` (and so `execute`) | `adbc_driver_postgresql` | accelerator, per connection |
| `read_record_batches` (ingest) | psycopg | the only ingest; **must not dispatch on the read path's availability predicate** |

What the choice of driver rests on, measured against live Redshift Serverless:
the driver connects and reports `vendor_name = "Redshift"`, and `SELECT` to an
Arrow table, `fetch_record_batch`, `adbc_get_table_schema`, `adbc_get_objects`
and `to_pyarrow_batches` all returned values matching psycopg. Redshift
`numeric` arrives as an opaque extension type that the per-batch cast converts
to the ibis schema's decimal. The one read failure seen live is not the
driver's: Redshift lower-cases an auto-generated upper-case alias even when it
is quoted, and the accelerated branch's per-batch cast matches fields by name,
so any driver returning the folded name meets it there. The psycopg branch
builds batches by position and does not. The feared failures on Redshift's incomplete
`pg_catalog` were all in xorq's own psycopg introspection, not in the driver.

### Ingest has no accelerator, and must not pretend otherwise

Optionality has to hold for *each* path, and the postgres backend's
`read_record_batches` is an unconditional ADBC ingest with no psycopg branch.
Neither ADBC driver can ingest into Redshift, and they fail differently:

- The Columnar driver's ingest is `COPY`-from-S3 underneath and refuses to run
  without a staging bucket: `INVALID_STATE: [redshift] Must set
  redshift.ingest.bucket to ingest data`.
- `adbc_driver_postgresql` ingests by `COPY … FROM STDIN WITH (FORMAT binary)`,
  and Redshift's `COPY` reads from S3 only:

      INVALID_ARGUMENT: [libpq] COPY query failed:
      ERROR: syntax error at or near "STDIN"    SQLSTATE: 42601

  That is a syntax error, so no setting reaches it, and the driver has no
  `INSERT` fallback.

So psycopg `INSERT` is not a baseline an accelerator later replaces; it is the
only ingest there is, and `COPY`-from-S3 stays out of scope because the baseline
does not need it.

Hence the prohibition in the table. Dispatching ingest on the read path's
availability predicate selects ADBC whenever the accelerator is usable for
reads, and its ingest never works on Redshift. One predicate cannot serve
two paths whose correct answers are opposite, so ingest consults none. A future
`COPY`-from-S3 path would branch on whether a staging bucket is configured,
inside the psycopg ingest; a bucket setting would be a non-secret `do_connect`
kwarg, so it would change build hashes only for users who set it.

### Degrading must distinguish "driver absent" from "driver failed"

The postgres backend falls back to psycopg on *any* exception from its ADBC
connect. Under an optional driver a blanket `except` is the mechanism itself: it
makes an expired credential or a rejected login indistinguishable from an absent
driver, and the operator sees a slow query instead of an error. So the Redshift
backend decides availability from local facts before dialling, and does not
wrap the connect; a failed connect raises, on the first read of the batches.
`_open_adbc_conn_or_none` and
`_adbc_unavailable_reason` in `python/xorq/backends/redshift/__init__.py` are
the implementation.

Execute is different. The inherited `to_pyarrow_batches`
(`python/xorq/backends/postgres/__init__.py`) catches every
`ADBCProgrammingError` from the ADBC query and re-runs the query on psycopg.
That covers what the separate ADBC connection cannot see (session-local
temporary tables, which a `temporary=True` ingest creates, and any session
state set through psycopg), and also syntax, permission and missing-relation
errors. Its cost is under *Negative*.

The accelerator is a second connection, opened per read, so it must be
configured like the first. `PgADBC` (`python/xorq/common/utils/postgres_utils.py`)
builds it from the psycopg connection's address, the caller's libpq settings
and the schema; `test_adbc_connection_settings.py` is the contract.

## Alternatives considered

### Repackage the Columnar driver as a platform wheel

Build `xorq-adbc-driver-redshift` wheels bundling the unmodified Columnar shared
library, resolving its absolute path at import, for the platforms upstream
builds. This would make the accelerator a declarable, lockable dependency, and
the mechanism is not novel: `adbc_driver_snowflake` ships its shared library
inside a PyPI wheel the same way.

Rejected. Every cost it carries is paid in advance for its only remaining
advantage over `adbc_driver_postgresql`, speed, which has never been measured:
xorq would become a redistributor of a third party's binary, with a wheel per
platform, a version-drift watcher and signature verification at build time. The
better outcome is that Columnar publishes it. A measured speed margin would be
an input to a *new* decision, not a trigger that resumes this one.

### `dbc install redshift`, out-of-band, as the primary mechanism

The pattern the backends with `dbc install` steps in `.github/workflows/` use.

Rejected as *primary*. It cannot be captured in `uv.lock`, so the environment is
not reproducible from the lockfile, and the installed driver version is whatever
the last out-of-band run left. Those backends open their driver by name
(`dbapi.connect(driver=...)` in `python/xorq/common/utils/databricks_utils.py`);
this one has no such path, so it is a precedent, not a working Redshift path.

### Make the driver mandatory

Ship one Arrow-native code path.

Rejected: an accelerator that is unavailable on a supported platform cannot be a
hard requirement. Before the IAM rig ran this would also have been forced rather
than chosen; psycopg authenticating is what made it a choice.

### Do not offer an accelerator at all

Rejected. The Arrow-native path avoids psycopg's per-row Python conversion, and
`adbc_driver_postgresql` offers it with no packaging work and no platform gap.
The margin is expected, not measured: no benchmark against psycopg or against
the Columnar driver on Redshift is recorded.

## Consequences

### Positive

- The backend installs with `uv sync`, and the accelerator is lockable rather
  than installed out of band.
- No supported platform gets a broken install or has to go without the
  accelerator for want of a build.
- The IAM question of *Context* is settled for the baseline: psycopg can present
  a temporary password, so no AWS SDK is needed to connect. Minting the password
  stays the caller's job.

### Negative

- **The accelerator is a bet on wire-protocol compatibility, and the bet is
  unhedged.** `adbc_driver_postgresql` targets PostgreSQL; Redshift's
  compatibility is a courtesy, so a release could regress against Redshift with
  nobody upstream treating it as a bug. `pyproject.toml` gives it no upper bound,
  so such a release would install automatically. A live-Redshift job would only
  catch it by asserting which branch served each read.
- **The execute-stage catch hides and repeats failures.** A query that fails on
  ADBC for any server-side reason runs a second time on psycopg, without a log
  line, and the error the caller sees is psycopg's. A read that silently lands
  on psycopg looks the same as one ADBC served.
- **A temporary IAM password is read at connect and reused.** The accelerator
  dials with the password in the connection's kwargs on every read, so once it
  expires every accelerated read raises while the psycopg session stays
  authenticated. `clone` reuses it too, unless it was passed as an environment
  reference, which `clone` resolves again. Nothing here refreshes it.
- **Which read branch serves a query is decided per connection, not per
  install.** `_adbc_unavailable_reason` rules the accelerator out for a
  connection with no password to hand it, which includes every backend built by
  `from_connection` and every connect that leaves the password to libpq. The
  same expression can therefore run on either branch, and the two build Arrow
  differently (by field name, and by position), so a defect in one surfaces on
  one population of connections only.
- The backend cannot be imported without the ADBC driver *manager*, which the
  postgres backend imports at module scope; it can without the driver.
  `test_import_needs_the_driver_manager_but_not_the_driver` in
  `python/xorq/tests/test_redshift_backend.py` pins both halves.
- Migration to another accelerator changes the availability predicate, the
  extras and `PgADBC`, which names `adbc_driver_postgresql`.
- Ingest is row-oriented `INSERT`, slower than `COPY`-from-S3 for large loads,
  and has no ADBC successor waiting.

## References

- `python/xorq/backends/redshift/__init__.py` — the availability predicate, the
  connect-stage probe, and the psycopg ingest
- `python/xorq/backends/postgres/__init__.py` — the backend subclassed, and the
  ADBC-first read path with its execute-stage catch
- `python/xorq/common/utils/postgres_utils.py` — `PgADBC`, which builds the
  accelerator's connection
- `python/xorq/tests/test_redshift_backend.py` — the offline contract, including
  the import requirement
- `python/xorq/backends/postgres/tests/test_adbc_connection_settings.py` — the
  two connections agreeing, against a server
