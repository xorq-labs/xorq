# ADR-0017: Incremental WAP publish — one `PublishMode`, capability-routed `PublishStrategy`

<!-- Provisional number: 0015 and 0016 are in flight on remote branches; bump if they land first. -->

- **Status:** Proposed
- **Date:** 2026-06-24
- **Deciders:** George (george@xorq.dev)
- **Context area:** `python/xorq/writes/` (`wap.py`, `enums.py`, new `incremental.py`), `python/xorq/backends/pyiceberg`, `python/xorq/backends/xorq_datafusion`

## Context

The WAP library (`python/xorq/writes/wap.py`, ADR-0014's TeeNode underneath) is **append-only**.
`make_wap_expr` writes a staging artifact via a tee, audits the full stream with an aggregate
pipeline-breaker, and the publish step *concatenates* staging into final — parquet merges row
groups (`make_publish_with_parquet`), iceberg `add_files`/fast-forwards a branch
(`publish_staging_table` / `publish_branch`). There is no way to publish a **changeset**:
insert-or-update by key, or apply deletes.

We want incremental publish — upsert and full merge (insert/update/delete) into a target —
across xorq's backends. The backends are **heterogeneous** in merge capability, which is the crux:

- **Native single-statement `MERGE INTO`:** duckdb, postgres (15+), snowflake, databricks, trino (ACID connectors).
- **Row-level upsert + delete, no `MERGE`:** pyiceberg (`Backend.upsert`, `python/xorq/backends/pyiceberg/__init__.py:197`).
- **`UPDATE`/`DELETE` but no portable `MERGE`:** sqlite.
- **No DML at all (immutable):** datafusion, xorq_datafusion, pandas.

So "publish a merge" means different mechanisms per backend, and reasonable people could pick a
different cut (one publish function per backend; a single rewrite-everywhere path; mode on the sink
vs. mode on publish). This ADR records the cut.

Scope target: the **5 supported backends plus xorq's embedded datafusion**. Whichever five, xorq's
datafusion lands in the immutable tier, so the manual rewrite path is mandatory, not optional.

### Verified groundwork

The design leans on three capabilities confirmed against the pinned deps:

- **sqlglot 28.6.0** exposes `Merge`/`When`/`Whens` and transpiles a single canonical `MERGE`
  string cleanly to duckdb/postgres/snowflake/trino/databricks. One authored statement, rendered
  per dialect by each backend's existing `raw_sql` (which calls `query.sql(dialect=…)`).
- **pyiceberg 0.11.1** exposes **both** `Transaction.upsert` and `Transaction.delete` on the same
  transaction, so iceberg upsert+delete commits as **one snapshot** — atomic, no two-commit window.
- **Transactions exist** where statement-DML is needed: postgres and sqlite both expose a `begin()`
  context manager (`vendor/ibis/backends/postgres/__init__.py:137`,
  `vendor/ibis/backends/sqlite/__init__.py:147`).

Consequence: **every tier can be made atomic.** A non-atomic publish is now a narrow fallback, not
the norm.

## Decision drivers

- **One user-facing knob.** A caller declares *intent* (append / upsert / merge); they must not
  have to pick the mechanism. Mechanism is derived from the backend.
- **Capability is a backend fact; the strategy choice is a WAP decision.** Derive the capability by
  dispatching on the backend type, but keep the mechanism table in WAP — not a name allowlist or a
  `hasattr` probe in the writes module.
- **Reuse the WAP skeleton.** `make_wap_expr` (tee → audit-breaker → mutate → publish) is unchanged;
  only the publish UDF and a thin wrapper are added.
- **Atomic publish wherever the backend allows it**, matching the all-or-nothing expectation WAP
  sets for parquet today.
- **Engine-agnostic fallback.** The immutable backends (datafusion/pandas) must work via one
  expression-level rewrite that needs no DML.
- **Append-only stays a first-class, zero-new-code path** — today's behavior, expressed as a mode.

## Decision

### Two enums: intent (user-facing) vs. mechanism (internal)

`python/xorq/writes/enums.py`, alongside the existing `WriteMode`:

```python
from xorq.common.compat import StrEnum   # same base as the existing WriteMode

class PublishMode(StrEnum):
    """How the staged changeset combines into `final`. The one knob a caller sets."""
    APPEND = "append"   # add all staging rows; no key; duplicates allowed  (== today's WAP)
    UPSERT = "upsert"   # insert-or-update by key; no _op column
    MERGE  = "merge"    # upsert + delete; requires an _op column ('D' deletes, else upsert)

class PublishStrategy(StrEnum):
    """Mechanism, auto-resolved from backend capability. Rarely set by hand (tests/override)."""
    APPEND        = "append"         # add rows: existing concat / add_files / INSERT…SELECT
    NATIVE_MERGE  = "native_merge"   # one MERGE INTO statement
    UPSERT_DELETE = "upsert_delete"  # pyiceberg Transaction.upsert + .delete (one snapshot)
    STATEMENT_DML = "statement_dml"  # UPDATE + INSERT + DELETE in one transaction
    REWRITE       = "rewrite"        # anti-join + union-all -> materialize -> atomic swap
```

`PublishMode` mirrors the sink's `WriteMode`: `WriteMode` governs how *staging* is written (always
`CREATE` for WAP — a fresh quarantine), `PublishMode` governs how staging combines into *final*.
The two were conflated before; they are now orthogonal. `PublishStrategy` is plumbing; the only
reason it is an enum is to let a test force a mechanism and to let warnings name one.

### Changeset contract: `key` and `_op`

`expr` (the upstream computation the tee captures, ADR-0014) produces the delta. Staging is a
verbatim materialization of it (the tee is an identity pass-through), so staging *is* the changeset.

- **`key`** — the column(s) that define row identity. Required for `UPSERT`/`MERGE`, forbidden for `APPEND`.
- **`_op`** — present only in `MERGE`. It only needs to distinguish delete from not: `'D'` ⇒ delete
  this key; **any other value ⇒ upsert** (insert-or-update). We never split insert from update —
  the merge decides that by key match. `UPSERT` forbids `_op`; `MERGE` requires it.
- **`key` is unique within a changeset** — at most one staging row per `key` value (for `MERGE`,
  that includes the `'D'` rows: a key may not appear as both a delete and an upsert in the same
  delta). This is not a stylistic nicety: native `MERGE INTO` raises a cardinality error
  ("MERGE command cannot affect a row a second time") when two source rows match one target row,
  whereas `REWRITE`'s anti-join + union-all and `STATEMENT_DML`'s UPDATE/INSERT/DELETE would each
  resolve a duplicate-keyed delta *differently and silently*. Requiring uniqueness is what keeps the
  three mechanisms observably equivalent. It is a **data**-level invariant, so build-time `_validate`
  (which sees only `mode`/`key`/`columns`, not rows) cannot check it; it is enforced by the
  recommended `audit_fn` — a "no duplicate keys" gate (see *Wrapper + the vacuous audit* and open
  question 5) that fails the publish before any tier runs.

### Capability routing — dispatch on the backend, decide in WAP

Merge capability is a **backend** fact ("can this engine do a keyed merge, only DML, or nothing?");
*which* `PublishStrategy` to use is a **WAP** decision. An earlier draft conflated the two with a
name allowlist (`con.name in {"duckdb", …}`) and a `hasattr(con, "upsert")` probe — re-deriving a
backend fact inside the writes module from a string or an incidental method. Instead, dispatch on the
backend **type** while keeping the decision table in one place, via `lazy_singledispatch` (the
vendored register-by-string dispatcher the datafusion/pandas backends already use):

```python
# writes/incremental.py — all strategy logic lives here; dispatch keys on the Backend type.
# lazy_singledispatch registers by string type-path, so NO backend (and none of its ADBC/driver
# deps) is imported at module load — the class is resolved only when a con of that type dispatches.
from xorq.vendor.ibis.common.dispatch import lazy_singledispatch

@lazy_singledispatch
def _strategy(con, mode) -> PublishStrategy:
    return PublishStrategy.REWRITE                              # default: immutable / unregistered

@_strategy.register("xorq.backends.duckdb.Backend")            # one handler, registered per dialect
@_strategy.register("xorq.backends.snowflake.Backend")
@_strategy.register("xorq.backends.databricks.Backend")
@_strategy.register("xorq.backends.trino.Backend")
def _(con, mode): return PublishStrategy.NATIVE_MERGE

@_strategy.register("xorq.backends.postgres.Backend")          # version-aware — a name list cannot be
def _(con, mode):
    return (PublishStrategy.NATIVE_MERGE if int(con.version.split(".")[0]) >= 15
            else PublishStrategy.STATEMENT_DML)

@_strategy.register("xorq.backends.pyiceberg.Backend")         # replaces hasattr(con, "upsert")
def _(con, mode): return PublishStrategy.UPSERT_DELETE

def resolve_strategy(con, mode) -> PublishStrategy:
    return PublishStrategy.APPEND if mode is PublishMode.APPEND else _strategy(con, mode)
```

`APPEND` is a mode-level fact (every backend appends), so it short-circuits before dispatch.
Registration is by string type-path, so no backend — and none of its ADBC/driver deps — is imported
when `incremental.py` loads; the class is resolved only when a `con` of that type first dispatches
(the idiom datafusion already uses for `_read_in_memory`, pandas for `_convert_object`). Dispatch keys on the backend type,
not its runtime `.name`, so a `.name` rename cannot misroute. The postgres handler reads the **live**
server version — the case the allowlist could not express (open question 3). An unregistered backend
defaults to `REWRITE`; an unregistered *SQL* backend then fails fast at publish (no parquet-backed
`final`), surfacing the gap rather than corrupting. Capability stays sourced from the backend — like
`BackendWriteThrough._supports_mode` (`write_through.py:198`), which reads the backend's own
signature — while the WAP strategy table stays in `incremental.py`.

### Mode → requirement → mechanism

| `PublishMode` | `key` | `_op` | resolves to | reuses |
|---|---|---|---|---|
| `APPEND` | forbidden | forbidden | `APPEND` | **existing** publish (concat / `add_files` / `INSERT…SELECT`) |
| `UPSERT` | required | forbidden | native / upsert-delete / statement / rewrite | new |
| `MERGE`  | required | required | native / upsert-delete / statement / rewrite | new |

### The router publish UDF

One UDF closes over `con`, dispatches by strategy. Same `{STAGING, FINAL, PASSED}` schema and shape
as today's publish UDFs (`wap.py:91`): it operates on staging/final **by name** and never sees data rows.

```python
def make_incremental_publish_with_backend(con, *, mode, key, columns, strategy=None):
    import xorq.expr.datatypes as dt
    from xorq.expr.udf import make_pandas_udf
    from xorq.vendor.ibis import schema

    strategy = strategy or resolve_strategy(con, mode)
    _validate(mode, key, columns)                       # build-time, see Validation

    @make_pandas_udf(schema=schema({STAGING: str, FINAL: str, PASSED: bool}),
                     return_type=dt.boolean, name=f"incr_publish_{strategy.value}")
    def publish(df):
        if len(df) != 1:
            raise ValueError(f"expected 1 row, got {len(df)}")
        row = df.iloc[0]
        if not row[PASSED]:
            return [False]                              # audit failed -> staging retained
        dispatch = {
            PublishStrategy.APPEND:        _publish_append,
            PublishStrategy.NATIVE_MERGE:  _publish_native_merge,
            PublishStrategy.UPSERT_DELETE: _publish_upsert_delete,
            PublishStrategy.STATEMENT_DML: _publish_statement_dml,
            PublishStrategy.REWRITE:       _publish_rewrite,
        }[strategy]
        dispatch(con, row[STAGING], row[FINAL], key, columns, mode)
        return [True]
    return publish
```

The five `_publish_*` are keyed on **strategy (mechanism), not backend** — five functions cover all
ten backends. `_publish_native_merge` serves every native-merge dialect through one sqlglot
render; `_publish_rewrite` serves datafusion/xorq/pandas; `_publish_statement_dml` serves
sqlite/old-postgres; `_publish_upsert_delete` serves pyiceberg. The only per-backend code is the
`lazy_singledispatch` registration in `resolve_strategy` and the thin convenience wrappers below — sugar
that binds `con` + sink + publish, mirroring today's `make_iceberg_wap_expr`.

### Tier 1 — `NATIVE_MERGE` (duckdb, postgres 15+, snowflake, databricks, trino)

Author one canonical `MERGE`, parse to a dialect-agnostic sqlglot AST, hand to `raw_sql` which
renders the backend dialect. Verified to transpile cleanly to all five.

```python
def _merge_query(final, staging, key, columns, mode):
    import sqlglot
    data = [c for c in columns if c not in key and c != "_op"]
    on   = " AND ".join(f'f."{k}" = s."{k}"' for k in key)
    sets = ", ".join(f'"{c}" = s."{c}"' for c in data)
    cols = ", ".join(f'"{c}"' for c in key + data)
    vals = ", ".join(f's."{c}"' for c in key + data)
    # Key-only changeset (no non-key, non-_op columns): a matched row is a no-op, and SQL forbids an
    # empty SET, so omit the UPDATE arm entirely rather than emit `UPDATE SET `.
    upd  = f"UPDATE SET {sets}" if data else None
    parts = [f'MERGE INTO "{final}" AS f USING "{staging}" AS s ON {on}']
    if mode is PublishMode.MERGE:
        parts.append(f'''WHEN MATCHED AND s."_op" = 'D' THEN DELETE''')
        if upd:
            parts.append(f'''WHEN MATCHED AND s."_op" <> 'D' THEN {upd}''')
        parts.append(f'''WHEN NOT MATCHED AND s."_op" <> 'D' THEN INSERT ({cols}) VALUES ({vals})''')
    else:
        if upd:
            parts.append(f"WHEN MATCHED THEN {upd}")
        parts.append(f"WHEN NOT MATCHED THEN INSERT ({cols}) VALUES ({vals})")
    return sqlglot.parse_one("\n".join(parts))

def _publish_native_merge(con, staging, final, key, columns, mode):
    con.raw_sql(_merge_query(final, staging, key, columns, mode))   # one statement -> atomic
    con.drop_table(staging)                                         # post-commit cleanup
```

### Tier 2 — `UPSERT_DELETE` (pyiceberg)

Drive the iceberg table's `Transaction` directly (not the `con.upsert` *wrapper*, which commits on
its own) so upsert and delete share **one snapshot**:

```python
def _publish_upsert_delete(con, staging, final, key, columns, mode):
    import pyarrow.compute as pc
    staged = con.catalog.load_table(f"{con.namespace}.{staging}").scan().to_arrow()
    full_final = f"{con.namespace}.{final}"
    if not con.catalog.table_exists(full_final):
        # _drop_op strips _op only if present, so it is a no-op on the UPSERT path
        con.catalog.create_table(full_final, schema=_drop_op(staged).schema)
    tgt = con.catalog.load_table(full_final)
    with tgt.transaction() as tx:                       # single snapshot
        if mode is PublishMode.MERGE:
            op = staged.column("_op")
            dele = staged.filter(pc.equal(op, "D"))
            if dele.num_rows:
                tx.delete(_key_filter(dele, key))       # In (single key) / Or-of-And (composite)
            up = _drop_op(staged.filter(pc.not_equal(op, "D")))
            if up.num_rows:
                tx.upsert(up, join_cols=key)
        else:
            tx.upsert(staged, join_cols=key)
    con.drop_table(staging)
```

### Tier 3 — `STATEMENT_DML` (sqlite; postgres < 15 fallback)

Genuine insert/**update**/delete in one transaction — *not* delete-then-reinsert, so in-place
`UPDATE` semantics (triggers, identity columns, FKs) are preserved:

```python
def _publish_statement_dml(con, staging, final, key, columns, mode):
    if not hasattr(con, "begin"):
        warnings.warn(f"{con.name}: no transaction primitive; statement-DML publish is NOT "
                      "atomic — a mid-publish failure can half-apply. See ADR-0017.")
    data = [c for c in columns if c not in key and c != "_op"]
    onf  = " AND ".join(f'"{final}"."{k}" = s."{k}"' for k in key)
    nd   = ''' AND s."_op" <> 'D' ''' if mode is PublishMode.MERGE else ""
    sets = ", ".join(f'"{c}" = s."{c}"' for c in data)
    cols = ", ".join(f'"{c}"' for c in key + data)
    stmts = []
    if data:                                            # key-only changeset: nothing to UPDATE
        stmts.append(f'''UPDATE "{final}" SET {sets} FROM "{staging}" s WHERE {onf}{nd}''')
    stmts.append(f'''INSERT INTO "{final}" ({cols}) SELECT {cols} FROM "{staging}" s
            WHERE NOT EXISTS (SELECT 1 FROM "{final}" WHERE {onf}){nd}''')
    if mode is PublishMode.MERGE:
        stmts.append(f'''DELETE FROM "{final}" WHERE EXISTS
            (SELECT 1 FROM "{staging}" s WHERE {onf} AND s."_op" = 'D')''')
    with con.begin() as cur:                            # postgres/sqlite -> atomic
        for s in stmts:
            cur.execute(s)
    con.drop_table(staging)
```

`UPDATE … FROM` is postgres-native and sqlite ≥ 3.33 (2020). Where a `UNIQUE`/`PK` index on `key`
exists, a single-statement `INSERT … ON CONFLICT (key) DO UPDATE` (+ a `DELETE` for `_op='D'`) is
the preferred form; DELETE+INSERT is the last-resort fallback when `UPDATE … FROM` is unavailable,
and only then with the identity/trigger warning. Statements will be generated via sqlglot for
dialect quirks (e.g. sqlite's `DELETE … FROM` alias handling), shown here as strings for clarity.

### Tier 4 — `REWRITE` (datafusion, xorq_datafusion, pandas)

No DML, but a full query engine. `final` is **the target exactly as WAP uses it today** — a table
name on `con` here (a path in the parquet publish). The immutable backends expose no `name → file`
lookup and this ADR does not invent one; durable cross-process persistence is the deferred
catalog-entry model (see Alternatives and Consequences). Express the merge as an ibis expression,
materialize it *before* replacing `final` (the expression reads `final`), then replace in one shot:

```python
def _publish_rewrite(con, staging, final, key, columns, mode):
    s = con.table(staging)
    survivors = con.table(final).anti_join(s, key)               # drop superseded + deleted keys
    applied   = s.filter(s["_op"] != "D").drop("_op") if mode is PublishMode.MERGE else s
    merged    = survivors.union(applied, distinct=False)         # UNION ALL
    materialized = con.to_pyarrow(merged)                        # fully read BEFORE replacing final
    con.create_table(final, materialized, overwrite=True)        # replace the target in place
    con.drop_table(staging)
```

Delete falls out for free: the anti-join drops every staging key (updated *and* deleted), and only
non-`D` rows are re-added. Materializing before `create_table(overwrite=True)` avoids reading
`final` after it has been replaced. For a **path** target the parquet publish
(`make_incremental_publish_with_parquet`) does the temp+rename file swap of `wap.py:131-143`
instead — same merge expression, durable artifact.

### Tier 5 / `APPEND` and the parquet target

- **`APPEND` strategy** delegates to the existing publish code (parquet concat, iceberg `add_files`,
  backend `INSERT … SELECT`). No new mechanism; `mode=APPEND` *is* today's WAP under the unified API.
- **`make_incremental_publish_with_parquet(mode, key, columns, engine="datafusion")`** is Tier 4
  with both inputs being files: an ephemeral engine registers `staging.parquet` + `final.parquet`,
  runs the same anti-join + union-all, `to_parquet` → atomic `.replace`.

### Wrapper + the vacuous audit

```python
def make_incremental_wap_expr(expr, *, staging, final, key, mode=PublishMode.UPSERT,
                              make_sink, publish, audit_fn=None):
    # The .aggregate(passed=…) breaker in make_wap_expr drains the whole stream, which is what
    # guarantees staging is fully committed before publish runs (ADR-0014 ordering invariant).
    # Incremental publish needs that barrier but not the audit semantic, so default to an
    # always-True audit rather than re-deriving the barrier. Callers may pass a real audit_fn
    # (e.g. a post-merge "no duplicate keys" gate).
    audit_fn = audit_fn or (lambda df: True)
    return make_wap_expr(expr, staging, final, audit_fn, make_sink=make_sink, publish=publish)
```

Convenience builders bind sink + publish, parallel to `make_iceberg_wap_expr`:

```python
make_backend_incremental_wap_expr(con, final, key, mode, *, staging=None)
make_iceberg_incremental_wap_expr(con, final, key, mode)     # forces UPSERT_DELETE
make_parquet_incremental_wap_expr(final, key, mode)
```

### Validation (build-time, in each `make_*`)

```python
def _validate(mode, key, columns):
    has_op = "_op" in columns
    if mode is PublishMode.APPEND:
        if key:    raise ValueError("APPEND takes no key")
        if has_op: raise ValueError("APPEND takes no _op column")
        return
    if not key:                       raise ValueError(f"{mode} requires a non-empty key")
    if not set(key).issubset(columns): raise ValueError(f"key {key} not all in {sorted(columns)}")
    if mode is PublishMode.MERGE and not has_op:
        raise ValueError("MERGE requires an '_op' column ('D' deletes, else upsert)")
    if mode is PublishMode.UPSERT and has_op:
        raise ValueError("UPSERT forbids '_op'; use MERGE to honor deletes")
```

### Atomicity

| strategy | backends | atomic? | mechanism |
|---|---|---|---|
| `NATIVE_MERGE` | duckdb, pg15+, snowflake, databricks, trino | yes | single `MERGE` statement |
| `UPSERT_DELETE` | pyiceberg | yes | one `Transaction` (upsert+delete) → one snapshot |
| `STATEMENT_DML` | sqlite, pg<15 | yes (with `begin()`) | UPDATE+INSERT+DELETE in one transaction |
| `REWRITE` | datafusion, xorq_datafusion, pandas | path target: yes / table target: see open Q4 | path target: temp parquet + atomic rename; table target: `create_table(overwrite=True)` is drop→create, not atomic |
| `APPEND` | all | per existing WAP | concat / `add_files` / `INSERT…SELECT` |

The non-atomic warning fires only in the narrow case of a DML backend with no `begin()`.

### File layout

- `enums.py` — add `PublishMode`, `PublishStrategy` next to `WriteMode`.
- `writes/incremental.py` (new) — `resolve_strategy`, `_validate`, the five `_publish_*`, the
  `make_incremental_*` builders. Imports `make_wap_expr` and the `STAGING/FINAL/PASSED/PUBLISHED`
  constants from `wap.py`, keeping `wap.py` focused on the append-only originals.

### Implementation phases

1. `enums.py` + `resolve_strategy` + `_validate` + the router skeleton.
2. **`NATIVE_MERGE` (duckdb)** and **`REWRITE` (datafusion)** behind round-trip tests — the two
   architectural extremes, covering both axes of the "5 + datafusion" target.
3. `UPSERT_DELETE` (iceberg, single-snapshot) and `make_incremental_publish_with_parquet`.
4. `STATEMENT_DML` (sqlite), postgres version probe, remaining native dialects (snowflake/databricks/trino) as connectors allow.

## Alternatives considered

### One enum that fuses intent and mechanism

A single `MergeMode` carrying both "what" and "how".

**Rejected.** Callers would have to know each backend's capability to pick a value, and the same
intent (`UPSERT`) maps to four different mechanisms. Splitting intent (`PublishMode`, the one knob)
from mechanism (`PublishStrategy`, auto-routed) is the whole point.

### Two similarly-named enums `MergeMode` / `MergeStrategy`

An earlier draft. **Rejected** as confusing (near-identical names) and because it omitted `APPEND`.
`PublishMode` (with `APPEND`) / `PublishStrategy` replaces it.

### `DELETE`-then-`INSERT` for the statement-DML tier

Express upsert as delete-the-key + re-insert, avoiding `UPDATE … FROM`.

**Rejected as default, kept as fallback.** It is only correct when the merge key is the row's
identity; otherwise it reassigns identity/serial columns, fires DELETE+INSERT triggers instead of
UPDATE, and can trip FK cascades. For a tier defined by *having* `UPDATE`, not using it discards the
in-place semantics that distinguish it from `REWRITE`. Retained only where `UPDATE … FROM` is
unavailable, with a warning.

### Mode on the sink (`WriteMode`) instead of on publish

Extend `WriteMode` to carry merge semantics.

**Rejected.** The sink writes a fresh staging quarantine every run (`CREATE`); merge semantics belong
to how staging combines into final. Conflating them is what this ADR untangles.

### One publish function per backend (no router)

`make_publish_with_duckdb`, `…_with_postgres`, etc.

**Deferred.** A capability router with five mechanisms covers ten backends without per-backend
functions; `strategy=` override remains for the rare case a backend needs a bespoke path.

### Capability declared on the backend (`con.publish_strategy(mode)`)

Instead of `lazy_singledispatch` in `incremental.py`, each backend declares its own
`publish_strategy(self, mode) -> PublishStrategy` next to its `read_record_batches`, and
`resolve_strategy` just calls `con.publish_strategy(mode)`. Capability genuinely *is* a backend
property, so it has a real claim to living there — the same way `read_record_batches` and the
`mode`-in-signature that `_supports_mode` inspects already do.

**Deferred (door open).** It is the more **extensible** design: a third-party backend can opt into
incremental publish without editing `incremental.py`. Two costs deferred it: it scatters
WAP-specific code across the ~6 in-scope backend modules, and — if the method returns
`PublishStrategy` directly — leaks the `writes.enums` type into `backends/` (a WAP-agnostic
capability flag mapped in WAP avoids the leak but reintroduces a mapping table). For a closed,
small strategy table, `lazy_singledispatch` keeps all WAP logic in one file; `resolve_strategy` is
the only call site, so switching to backend-declared capability later is a localized change. Revisit
if external backends need to register a strategy out-of-tree.

### Rewrite-everywhere (use Tier 4 for all backends)

Skip native merge; always anti-join + union-all + swap.

**Rejected.** Throws away in-place native `MERGE`/upsert, forcing an O(final) rewrite even on engines
that can do a keyed merge cheaply, and loses transactional in-place semantics. Reserved for the
immutable backends that have no choice.

### `final` as a versioned catalog entry

Make the publish target a xorq catalog entry, so each incremental publish writes a new
content-addressed version rather than mutating a file or table in place. This gives immutable-backend
targets durable, auditable history for free and fits xorq's catalog model — and is the natural answer
to immutable-backend cross-process durability.

**Deferred.** It is a different durability model than today's WAP (which mutates `final` in place),
so it is a separate design, not part of this ADR. Until it lands, `final` is the target exactly as
WAP uses it now — a path (durable via swap) or a backend table name (durable per the backend's own
persistence). The `REWRITE` tier and the out-of-scope note in Consequences are written so that
adopting catalog-entry targets later changes the publish destination without reopening the
mode/strategy design.

## Consequences

### Positive

- One user-facing knob (`PublishMode`), mechanism auto-routed; mirrors the sink's `WriteMode`.
- The WAP skeleton (`make_wap_expr`) is untouched; only a publish UDF family and a thin wrapper are added.
- Every tier is atomic given the verified primitives (single `MERGE`, one iceberg `Transaction`,
  `con.begin()`, temp+rename). Non-atomic is a narrow, warned fallback.
- `APPEND` unifies today's append-only WAP into the same API at zero new code.
- The immutable backends work through one engine-agnostic expression (anti-join + union-all),
  identical in spirit to the parquet path; delete falls out of the anti-join for free.
- A real `audit_fn` can still gate incremental publish (e.g. post-merge invariants).

### Negative

- **`REWRITE` is O(final) per publish**, regardless of delta size — inherent to immutable-merge.
  Mitigation (and the point of the `incremental-wap` branch): partition `final` and rewrite only the
  partitions the delta touches, turning O(final) into O(touched partitions). Not in this ADR.
- **`REWRITE` cross-process durability is out of scope.** `final` is the target as used today — a
  table name on `con`. For datafusion/pandas that is a session-scoped registered provider, so a
  REWRITE merge updates the in-session table but does not by itself persist across processes. File
  (path) targets are durable via swap. Durable, versioned persistence for immutable-backend table
  targets is the deferred catalog-entry model (see Alternatives), deliberately not invented here.
- **Postgres `MERGE` needs 15+**; an old server silently wants the `STATEMENT_DML` fallback, gated on
  a version probe (open Q4) — a wrong probe means a runtime SQL error.
- **The `_op` column is a leaky contract.** It rides in `expr`'s schema as a data column and must be
  stripped before it reaches `final`; a backend or tier that forgets to drop it corrupts the target
  schema. Enforced only by each `_publish_*` and `_validate`, not by a type.
- **`UPDATE … FROM` / `ON CONFLICT` availability is engine-specific**; the statement-DML tier needs a
  per-engine capability check, not just a name allowlist.
- **`drop_table(staging)` is post-commit cleanup**, not part of the atomic unit; a failure there
  leaves an orphan staging table after a successful publish (must not mask success — mirror
  `make_publish_with_parquet`'s "removing staging is cleanup" stance, `wap.py:141`).
- Re-`register` after a `REWRITE` swap is process-local; another process holding the old registration
  keeps the pre-swap file until it re-registers.
- **New backends are invisible to the router by default** — an unregistered backend silently routes
  to `REWRITE`. This is the flip side of centralizing the table: out-of-tree backends cannot opt in
  without editing `incremental.py` (the extensibility the deferred per-backend method would buy).

## Open questions

1. **Composite-key delete filter** for iceberg — `In` (single key) vs. `Or`-of-`And` (composite)
   against the pyiceberg 0.11 expression API. (`_key_filter`.)
2. **First-run `final` creation** — every tier needs a "final doesn't exist yet" branch
   (create empty / CTAS-from-staging); factor a shared `_ensure_final(con, final, staging)`. In
   `REWRITE` this is also what `con.table(final)` needs before the anti-join.
3. **Postgres version probe** — detect server < 15 and downgrade `NATIVE_MERGE` → `STATEMENT_DML`.
4. **Replace semantics for `REWRITE`** — confirm `con.create_table(final, arrow, overwrite=True)` on
   datafusion/pandas, including the drop→create window; and `con.to_parquet` `(expr, path)` for the
   parquet publish (`xorq_datafusion/__init__.py:954`).
5. **Duplicate-key gate as the default audit** — the changeset contract requires unique keys per
   delta (a data-level invariant `_validate` cannot see). Should `make_incremental_wap_expr` default
   to a "no duplicate keys" `audit_fn` for `UPSERT`/`MERGE` (failing fast, portable across tiers)
   rather than the vacuous always-True audit, leaving always-True only for `APPEND`? The breaker
   already drains the full stream, so the uniqueness check is essentially free there.

## References

- ADR-0014: TeeNode, deferred write as a side effect — `docs/adr/0014-teenode-deferred-writes.md`
- WAP library: `python/xorq/writes/wap.py` (`make_wap_expr`, `make_publish_with_parquet`, `make_publish_with_iceberg`)
- WriteThrough sinks: `python/xorq/writes/write_through.py` (`BackendWriteThrough`, `_supports_mode:198`)
- pyiceberg backend: `python/xorq/backends/pyiceberg/__init__.py` (`upsert:197`, `publish_staging_table:398`, `read_record_batches:314`)
- xorq_datafusion backend: `python/xorq/backends/xorq_datafusion/__init__.py` (`register:449`, `to_parquet:954`, `deregister_table`)
- raw_sql / begin: `vendor/ibis/backends/{duckdb,postgres,sqlite}/__init__.py`
- `lazy_singledispatch` (string-registered, import-on-dispatch): `python/xorq/vendor/ibis/common/dispatch.py`; existing use in `backends/datafusion/__init__.py:149`, `backends/pandas/__init__.py:357`
- examples: `examples/wap_audit_parquet.py`, `examples/wap_audit_iceberg.py`
