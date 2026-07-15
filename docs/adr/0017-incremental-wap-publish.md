# ADR-0017: Incremental WAP publish — one `PublishMode`, capability-routed `PublishStrategy`

<!-- 0015 landed; 0016 is still in flight on a remote branch (#2103) — 0017 stands. -->

- **Status:** Accepted — implemented (XOR-444); remaining: trino `publish_strategy` + databricks live-run verification (XOR-445)
- **Date:** 2026-06-24 (implemented 2026-07)
- **Deciders:** George (george@xorq.dev)
- **Context area:** `python/xorq/writes/` (`wap.py`, `enums.py`, new `publish.py`), `python/xorq/backends/pyiceberg`, `python/xorq/backends/xorq_datafusion`

## Context

The WAP library (`python/xorq/writes/wap.py`, ADR-0014's TeeNode underneath) is **append-only**.
`make_wap_expr` writes a staging artifact via a tee, audits the full stream with an aggregate
pipeline-breaker, and the publish step *concatenates* staging into final — parquet merges row
groups (`make_publish_with_parquet`), iceberg `add_files`/fast-forwards a branch
(`publish_staging_table` / `publish_branch`). There is no way to publish a **changeset**:
insert-or-update by key, or apply deletes.

We want incremental publish — upsert and full merge (insert/update/delete) into a target —
across xorq's backends. The backends are **heterogeneous** in merge capability, which is the crux:

- **Native single-statement `MERGE INTO`:** duckdb, postgres (15+), snowflake, databricks, trino (ACID connectors), gizmosql (DuckDB-backed Flight SQL).
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

- **sqlglot** parses and transpiles a single canonical `MERGE` string cleanly to
  duckdb/postgres/snowflake/trino/databricks — verified identical rendering across 24.0–28.6. One
  authored statement, `parse_one`d, rendered per dialect via `query.sql(dialect=…)`. (The builder
  API is deliberately *not* used — see Alternatives: it needs sqlglot ≥26, but co-installation with
  ibis 9.5 caps sqlglot <25.21.)
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
  and `UPSERT_DELETE` likewise rejects a non-unique source — pyiceberg's `Transaction.upsert` raises
  on duplicate join-key values — whereas `REWRITE`'s anti-join + union-all and `STATEMENT_DML`'s
  UPDATE/INSERT/DELETE would each resolve a duplicate-keyed delta *differently and silently*.
  Requiring uniqueness is what keeps the four mechanisms observably equivalent. It is a
  **data**-level invariant, so structural `_validate` (which sees only `mode`/`key`/`columns`, not
  rows) cannot check it; it is enforced by an `audit_fn` — a "no duplicate keys" gate that fails the
  publish before any tier runs. The WAP builders therefore **default `UPSERT`/`MERGE` to this gate**
  (`_default_audit`; `APPEND` keeps always-True), so the wrapper enforces it by default. The standalone
  `publish` primitives have no audit, so there the invariant is a caller responsibility.

### Capability routing — each backend declares its own strategy

Merge capability is a **backend** fact ("can this engine do a keyed merge, only DML, or nothing?"),
so the backend declares it. Each backend that can do more than a full rewrite implements
`publish_strategy(self) -> PublishStrategy`; `resolve_strategy` in WAP short-circuits `APPEND`
(a mode-level fact) and otherwise just asks the backend. The declaration deliberately takes no
`mode`: modes must behave uniformly across backends, so the mechanism never varies by mode —
everything mode-shaped (delete clauses, the `_op` contract) is the publish layer's job:

```python
# writes/publish.py — WAP only asks; the capability lives on each backend.
def _backend_strategy(con, mode) -> PublishStrategy:
    # Looked up on type(con), not the instance, so it never trips a backend's
    # table-access __getattr__; and it's called on the con we already hold, so
    # routing one backend imports none of the others. A backend that declares no
    # publish_strategy (datafusion, xorq_datafusion, pandas) falls back to REWRITE.
    declare = getattr(type(con), "publish_strategy", None)
    return declare(con) if declare is not None else PublishStrategy.REWRITE

def resolve_strategy(con, mode) -> PublishStrategy:
    return PublishStrategy.APPEND if mode is PublishMode.APPEND else _backend_strategy(con, mode)
```
```python
# writes are declared where the capability is — one small method per backend, e.g.:
# xorq/backends/duckdb/__init__.py    -> return PublishStrategy.NATIVE_MERGE
# xorq/backends/sqlite/__init__.py    -> version-aware: STATEMENT_DML on sqlite 3.33+, else REWRITE
# xorq/backends/pyiceberg/__init__.py -> return PublishStrategy.UPSERT_DELETE
# xorq/backends/postgres/__init__.py  -> version-aware: NATIVE_MERGE on pg15+, else STATEMENT_DML
from xorq.writes.enums import PublishStrategy   # module-level: enums-only, no cycle

class Backend(IbisDuckDBBackend):
    def publish_strategy(self):
        return PublishStrategy.NATIVE_MERGE
```

Why per-backend rather than a central registry in WAP:

- **Capability genuinely *is* a backend property** — this mirrors how xorq already sources capability
  from the backend: `BackendWriteThrough._supports_mode` (`write_through.py:198`) reads the backend's
  own `read_record_batches` signature, and `read_record_batches` is itself a per-backend method.
- **Nothing else is imported.** `resolve_strategy` calls a method on the con you already hold, so a
  single-backend install (`xorq[duckdb]`) never needs pyiceberg/snowflake/etc. importable just to
  route. The `PublishStrategy` import is module-level: the `writes` package is light at import time
  (its heavy imports — udf machinery, ibis schema — are in-function) and never imports backends at
  module scope (pyiceberg already imports `WriteMode` this way), so the edge is cheap and acyclic.
- **The postgres version probe lives on postgres**, where it belongs, instead of as a special case in
  a central router.
- Looked up on `type(con)` (not `.name`), so a `.name` rename cannot misroute; and via the class (not
  the instance) so a backend's table-access `__getattr__` is never triggered. An unregistered backend
  defaults to `REWRITE`; an unregistered *SQL* backend then fails fast at publish (no parquet-backed
  `final`), surfacing the gap rather than corrupting.

This was the "capability declared on the backend" alternative in earlier drafts (see Alternatives);
it is now the decision. It replaces two rejected shapes: a `con.name` allowlist / `hasattr(con,
"upsert")` probe (re-derives a backend fact in WAP from a string or an incidental method), and
ibis's `lazy_singledispatch` (buckets lazy registrations by top-level package and imports the whole
`xorq.*` bucket on first dispatch — fatal in a single-backend install; and `functools.singledispatch`
is worse still, importing every backend class at module load).

### Mode → requirement → mechanism

| `PublishMode` | `key` | `_op` | resolves to | reuses |
|---|---|---|---|---|
| `APPEND` | forbidden | forbidden | `APPEND` | **existing** publish (concat / `add_files` / `INSERT…SELECT`) |
| `UPSERT` | required | forbidden | native / upsert-delete / statement / rewrite | new |
| `MERGE`  | required | required | native / upsert-delete / statement / rewrite | new |

### Two layers: `publish()` (reconciliation) and the WAP UDF that wraps it

The reconciliation is a plain function in `writes/publish.py` that knows nothing about WAP — it
operates on staging/final **by name**, discovers columns, validates, and dispatches by strategy:

```python
# writes/publish.py — the reconciliation layer, no tee/audit/WAP knowledge
def publish(con, staging, final, *, key=(), mode) -> None:
    key = list(key)
    try:
        columns = _staging_columns(con, staging)        # discovered at call time
    except Exception:
        raise _staging_missing(staging) from None        # empty input / async sink
    _validate(mode, key, columns)                        # full contract check, see Validation
    _PUBLISH[resolve_strategy(con, mode)](con, staging, final, key, columns, mode)
```

`writes/wap.py` then wraps that in the publish UDF — thin glue over the same `{STAGING, FINAL,
PASSED}` schema as today's publish UDFs (`wap.py:91`), never seeing data rows:

```python
# writes/wap.py — the orchestration layer; imports publish, never the reverse
def make_publish_with_backend(con, *, key=(), mode):
    from xorq.writes.publish import publish
    if mode is not PublishMode.APPEND and not key:       # build-time key presence
        raise ValueError(f"{mode} requires a non-empty key")

    # name folds in the closure identity (con name + key, dasher-tokenized):
    # UDFs register into the session context and compile by name, so two
    # publish UDFs sharing a name in one plan would resolve to one closure.
    # The token is deterministic, keeping build hashes reproducible.
    @make_pandas_udf(schema=schema({STAGING: str, FINAL: str, PASSED: bool}),
                     return_type=dt.boolean,
                     name=_publish_udf_name(f"wap_publish_{mode.value}",
                                            getattr(con, "name", ""), tuple(key)))
    def publish_udf(df):
        if len(df) != 1:
            raise ValueError(f"expected 1 row, got {len(df)}")
        row = df.iloc[0]
        if not row[PASSED]:
            return [False]                               # audit failed -> staging retained
        publish(con, row[STAGING], row[FINAL], key=key, mode=mode)
        return [True]
    return publish_udf
```

Dependency flows one way — **wap imports publish, never the reverse** — so the standalone `publish()`
is usable without any WAP machinery, and the WAP path is a thin consumer of it.

The five `_publish_*` are keyed on **strategy (mechanism), not backend** — five functions cover all
eleven backends. `_publish_native_merge` serves every native-merge dialect through one sqlglot
render; `_publish_rewrite` serves datafusion/xorq/pandas; `_publish_statement_dml` serves
sqlite/old-postgres; `_publish_upsert_delete` serves pyiceberg. The only per-backend code is each
backend's small `publish_strategy` method (and the thin convenience wrappers below — sugar that binds
`con` + sink + publish, mirroring today's `make_iceberg_wap_expr`).

### Tier 1 — `NATIVE_MERGE` (duckdb, postgres 15+, snowflake, databricks, gizmosql; trino pending)

Trino has native `MERGE` (ACID connectors) and the canonical statement transpiles to its dialect,
but its backend does not yet declare `publish_strategy` — it currently falls back to `REWRITE`.
The XOR-445 remainder is more than the declaration (no tee sink, connector-gated `MERGE`) — see
*Remaining* under Open questions.

Author one canonical `MERGE` with `"`-quoted identifiers (`_q`), `parse_one` to a dialect-agnostic
sqlglot AST, render the backend dialect via `.sql(dialect=con.dialect)`, hand to `raw_sql`.
Verified to transpile cleanly to all five dialects; gizmosql is a DuckDB-backed Flight SQL server
(`compiler = sc.duckdb.compiler`), so it renders through the duckdb dialect and adds no sixth
dialect to verify. String authoring over the builder API is deliberate (see Alternatives).

```python
def _merge_query(final, staging, key, columns, mode):
    import sqlglot
    data = [c for c in columns if c not in key and c != "_op"]
    on   = " AND ".join(f"f.{_q(k)} = s.{_q(k)}" for k in key)
    sets = ", ".join(f"{_q(c)} = s.{_q(c)}" for c in data)
    cols = ", ".join(_q(c) for c in [*key, *data])
    vals = ", ".join(f"s.{_q(c)}" for c in [*key, *data])
    parts = [f"MERGE INTO {_q(final)} AS f USING {_q(staging)} AS s ON {on}"]
    if mode is PublishMode.MERGE:
        # _NOT_DEL = (s."_op" <> 'D' OR s."_op" IS NULL) — parens load-bearing: the predicate
        # lands after the implicit AND in `WHEN MATCHED AND <cond>` (pinned by test).
        parts.append("""WHEN MATCHED AND s."_op" = 'D' THEN DELETE""")
        if data:   # key-only changeset: a matched row is a no-op and SQL forbids an empty SET
            parts.append(f"WHEN MATCHED AND {_NOT_DEL} THEN UPDATE SET {sets}")
        parts.append(f"WHEN NOT MATCHED AND {_NOT_DEL} THEN INSERT ({cols}) VALUES ({vals})")
    else:
        if data:
            parts.append(f"WHEN MATCHED THEN UPDATE SET {sets}")
        parts.append(f"WHEN NOT MATCHED THEN INSERT ({cols}) VALUES ({vals})")
    return sqlglot.parse_one("\n".join(parts))

def _publish_native_merge(con, staging, final, key, columns, mode):
    query = _merge_query(final, staging, key, columns, mode)        # (APPEND: INSERT…SELECT instead)
    con.raw_sql(query.sql(dialect=con.dialect))                     # one statement -> atomic
    _drop_staging(con, staging)                                     # post-commit cleanup
```

### Tier 2 — `UPSERT_DELETE` (pyiceberg)

Drive the iceberg table's `Transaction` directly (not the `con.upsert` *wrapper*, which commits on
its own) so upsert and delete share **one snapshot**:

```python
def _publish_upsert_delete(con, staging, final, key, columns, mode):
    if mode is PublishMode.APPEND:
        con.publish_staging_table(staging, final)       # add_files — metadata-only, no row rewrite
        return
    import pyarrow.compute as pc
    final_cols = _data_cols(columns, mode)              # column *names*; strips "_op" only for MERGE
    staged = con.catalog.load_table(f"{con.namespace}.{staging}").scan().to_arrow()
    full_final = f"{con.namespace}.{final}"
    if not con.catalog.table_exists(full_final):
        con.catalog.create_table(full_final, schema=staged.select(final_cols).schema)
    tgt = con.catalog.load_table(full_final)
    with tgt.transaction() as tx:                       # single snapshot
        if mode is PublishMode.MERGE:
            # 'D' deletes; anything else (incl. NULL _op) upserts.
            is_delete = pc.fill_null(pc.equal(staged.column("_op"), "D"), False)
            deletes = staged.filter(is_delete)
            if deletes.num_rows:
                tx.delete(_key_filter(deletes, key))    # In (single key) / Or-of-And (composite)
            upserts = staged.filter(pc.invert(is_delete)).select(final_cols)
            if upserts.num_rows:
                tx.upsert(upserts, join_cols=list(key))
        else:
            tx.upsert(staged.select(final_cols), join_cols=list(key))
    _drop_staging(con, staging)                         # force-drop (handles the duckdb-view case)
```

### Tier 3 — `STATEMENT_DML` (sqlite; postgres < 15 fallback)

Genuine insert/**update**/delete in one transaction — *not* delete-then-reinsert, so in-place
`UPDATE` semantics (triggers, identity columns, FKs) are preserved:

```python
def _publish_statement_dml(con, staging, final, key, columns, mode):
    import sqlglot
    data = [c for c in columns if c not in key and c != "_op"]
    onf  = " AND ".join(f'"{final}"."{k}" = s."{k}"' for k in key)   # UPDATE target: no alias
    # MERGE: rows with NULL _op upsert (only literal 'D' deletes) — same rule as every tier.
    nd   = f" AND {_NOT_DEL}" if mode is PublishMode.MERGE else ""
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
            cur.execute(sqlglot.parse_one(s).sql(dialect=con.dialect))
    _drop_staging(con, staging)
```

`UPDATE … FROM` is postgres-native and sqlite ≥ 3.33 (2020); sqlite's `publish_strategy` probes
`sqlite3.sqlite_version_info` and drops to the universal `REWRITE` floor below 3.33, the same
shape as postgres < 15 dropping a tier. Where a `UNIQUE`/`PK` index on `key`
exists, a single-statement `INSERT … ON CONFLICT (key) DO UPDATE` (+ a `DELETE` for `_op='D'`) is
the preferred form; DELETE+INSERT is the last-resort fallback when `UPDATE … FROM` is unavailable,
and only then with the identity/trigger warning. Statements are authored once like Tier 1 and
rendered per dialect via `parse_one(...).sql(dialect=con.dialect)`.

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
    # NULL _op upserts (only literal 'D' deletes) — same rule as every tier; first
    # run (no `final` yet) writes the changeset directly, skipping the anti-join.
    applied   = (s.filter((s["_op"] != "D") | s["_op"].isnull()).drop("_op")
                 if mode is PublishMode.MERGE else s)
    merged    = survivors.union(applied, distinct=False)         # UNION ALL
    materialized = con.to_pyarrow(merged)                        # fully read BEFORE replacing final
    con.create_table(final, materialized, overwrite=True)        # replace the target in place
    _drop_staging(con, staging)
```

Delete falls out for free: the anti-join drops every staging key (updated *and* deleted), and only
non-`D` rows are re-added. Materializing before `create_table(overwrite=True)` avoids reading
`final` after it has been replaced. For a **path** target the parquet publish
(`make_publish_with_parquet`) does the temp+rename file swap
instead (`_publish_parquet_merge` in `publish.py`) — same merge semantics, durable artifact.

### Tier 5 / `APPEND` and the parquet target

- **`APPEND` strategy** delegates to the existing publish code (parquet concat, iceberg `add_files`,
  backend `INSERT … SELECT`). No new mechanism; `mode=APPEND` *is* today's WAP under the unified API.
- **`make_publish_with_parquet(*, key, mode)`** is Tier 4 with both inputs being files: the same
  anti-join + union-all, `to_parquet` → atomic `.replace`. Implementation note: the merge runs in
  **pandas**, not xorq's embedded engine — the publish UDF executes *inside* the engine driving the
  WAP expression, and spinning up datafusion there nests one runtime in another ("cannot start a
  runtime from within a runtime"). Pandas is a core dep with no runtime and the file is rewritten
  whole either way. (`APPEND` short-circuits to a streaming pyarrow concat — no engine at all.)

### Wrapper + the default audit

Each builder binds the changeset config (`key`/`mode`) and returns a builder taking
`(expr, staging, final, audit_fn=None)` — so `audit_fn` is supplied **at the pipe**, exactly like the
core `make_wap_expr` and `make_iceberg_wap_expr`. Omit it and `UPSERT`/`MERGE` get the default
no-duplicate-keys gate; pass one for a real DQ check. The `.aggregate(passed=…)` breaker in
`make_wap_expr` drains the whole stream (the ADR-0014 ordering invariant that commits staging before
publish) and is where that audit runs:

```python
def make_backend_wap_expr(con, *, key, mode):
    publish = make_publish_with_backend(con, key=key, mode=mode)
    def build(expr, staging, final, audit_fn=None):
        audit = audit_fn if audit_fn is not None else _default_audit(key, mode)
        return make_wap_expr(expr, staging, final, audit,
                             make_sink=lambda s: make_sink_with_backend(con, s), publish=publish)
    return build

def _default_audit(key, mode):          # UPSERT/MERGE: no-duplicate-keys gate; APPEND: vacuous
    if mode is PublishMode.APPEND:
        return lambda df: True
    cols = list(key)
    return lambda df: not df.duplicated(subset=cols).any()
```

The parquet builder mirrors this; iceberg routes through the backend builder for incremental.
Iceberg's branch mechanism folds in as a **staging strategy** — `staging_strategy=StagingStrategy.BRANCH`
stages on a branch of `final` itself and publishes by fast-forward (metadata-only, all-or-nothing, so
`APPEND`-only with no key; the backend's type must declare `publish_branch`, the same
capability-is-a-backend-fact lookup as `publish_strategy`). `make_iceberg_wap_expr` survives as a thin
APPEND alias. Usage — a deferred expr, `audit_fn` optional at the pipe:

```python
source.pipe(make_backend_wap_expr(con, key=["id"], mode=PublishMode.UPSERT), staging, final)
source.pipe(make_backend_wap_expr(ice, mode=PublishMode.APPEND,
            staging_strategy=StagingStrategy.BRANCH), staging_branch, final)
source.pipe(make_parquet_wap_expr(), staging, final, audit_no_nulls)   # append default + DQ audit
```

The **non-WAP** primitives run the same `resolve_strategy` + `_validate` + dispatch reconciliation
without the tee + audit gate, for an already-staged changeset:

```python
publish(con, staging, final, *, key, mode)                # backend target
publish_parquet(staging_path, final_path, *, key, mode)
```

### Validation (structural — sees `mode`/`key`/`columns`, never rows)

```python
def _validate(mode, key, columns):
    has_op = "_op" in columns
    if mode is PublishMode.APPEND:
        if key: raise ValueError("APPEND takes no key")
        return                            # _op is allowed for APPEND — ordinary data, not a control column
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
| `NATIVE_MERGE` | duckdb, pg15+, snowflake, databricks, gizmosql (trino pending — XOR-445) | yes | single `MERGE` statement |
| `UPSERT_DELETE` | pyiceberg | yes | one `Transaction` (upsert+delete) → one snapshot |
| `STATEMENT_DML` | sqlite, pg<15 | yes (with `begin()`) | UPDATE+INSERT+DELETE in one transaction |
| `REWRITE` | datafusion, xorq_datafusion, pandas | path target: yes / table target: see open Q4 | path target: temp parquet + atomic rename; table target: `create_table(overwrite=True)` is drop→create, not atomic |
| `APPEND` | all | per existing WAP | concat / `add_files` / `INSERT…SELECT` |

`STATEMENT_DML` is only ever routed to backends with a `begin()` context manager (sqlite,
postgres), so no non-atomic DML path exists in the implementation — the "no transaction primitive"
warning from earlier drafts was dropped rather than shipped dead.

### File layout

Two layers, one dependency arrow (`wap` → `publish`, never back):

- `enums.py` — add `PublishMode`, `PublishStrategy` (and `StagingStrategy` for the table/branch
  staging split) next to `WriteMode`.
- each backend (`backends/<b>/__init__.py`) — a small `publish_strategy(self)` declaring its
  mechanism (immutable backends declare none → `REWRITE`).
- `writes/publish.py` (new) — **pure reconciliation**: `resolve_strategy`, `_validate`, the five
  `_publish_*` (+ the parquet merge), and the standalone `publish` / `publish_parquet` entry points.
  Imports nothing from `wap.py`.
- `writes/wap.py` — **orchestration**: `make_wap_expr`, the `STAGING/FINAL/PASSED/PUBLISHED`
  constants, the sinks, `_default_audit`, the publish UDF factories (which wrap `publish`), and all
  the `make_*_wap_expr` builders. Imports `publish` / `publish_parquet` from `publish.py`.

### Implementation phases

1. `enums.py` + `resolve_strategy` + `_validate` + the router skeleton.
2. **`NATIVE_MERGE` (duckdb)** and **`REWRITE` (datafusion)** behind round-trip tests — the two
   architectural extremes, covering both axes of the "5 + datafusion" target.
3. `UPSERT_DELETE` (iceberg, single-snapshot) and `make_publish_with_parquet`.
4. `STATEMENT_DML` (sqlite), postgres version probe, remaining native dialects (snowflake/databricks/trino) as connectors allow.

Status: phases 1–4 landed except the trino declaration (XOR-445). Snowflake/databricks/gizmosql/
postgres carry gated integration tests (`backends/<b>/tests/test_wap_incremental.py`); databricks
is in-tree but not yet verified against a live workspace (credentials pending).

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

**Deferred.** A capability router with five mechanisms covers eleven backends without per-backend
functions; `strategy=` override remains for the rare case a backend needs a bespoke path.

### Central strategy registry in WAP (a table / dispatcher, not per-backend)

Instead of each backend declaring `publish_strategy` (the Decision), keep the backend→strategy
mapping in one place in `publish.py`. Three shapes were tried:

- **`lazy_singledispatch`** (register `"xorq.backends.duckdb.Backend"` → strategy). *Rejected:* it
  buckets lazy registrations by top-level package and imports the **whole `xorq.*` bucket** on first
  dispatch (`vendor/ibis/common/dispatch.py:65,105`), so routing one backend imports them all —
  `ModuleNotFoundError` on any single-backend install. (`functools.singledispatch` is worse: it
  registers by class, importing every backend at module load. A `match`/dict keyed on the
  type-path *string* avoids imports, but is just the next shape.)
- **A `BackendType` string table matched against `type(con).__mro__`.** *Rejected:* correct and
  import-safe, but it centralizes a fact that is genuinely per-backend, and it can't host the
  postgres version probe without a special case in the router.
- Either keeps all WAP logic in one file (the appeal) but re-derives a **backend** fact inside WAP.

**Chosen instead:** backend-declared `publish_strategy` (see *Capability routing*). Capability lives
where it belongs, imports nothing (the method is on the con you hold; `PublishStrategy` is imported
inside the method), and postgres's version probe sits on postgres. The cost — a small method in each
of ~7 backend modules — is the same idiom xorq already uses for `read_record_batches` /
`_supports_mode`. An out-of-tree backend can opt in without editing `publish.py`.

### sqlglot builder API instead of strings + `parse_one`

Author statements as ASTs (`exp.merge` / `exp.When` / `exp.Update`, identifiers `quoted=True`) —
no string assembly, no hand-rolled `_q`, injection-safe by construction.

**Tried and reverted** (it briefly shipped on the XOR-444 branch). The builder surface is too
version-sensitive for xorq's install matrix: `exp.merge` exists only on sqlglot ≥ 26, and
`Update`'s FROM kwarg is keyed `"from"` (silently dropped as `from_`) before 28 — while
**co-installation with ibis 9.5 caps sqlglot < 25.21** (the `ci-test-ibis-compatibility` matrix),
so no single version satisfies both. Strings + `parse_one` render identically across 24.0–28.6
(verified per-version). Two findings from the attempt are kept: the not-delete predicate's
**explicit parens** (sqlglot won't parenthesize an OR landing after the implicit AND in
`WHEN MATCHED AND <cond>` — pinned by test), and per-dialect rendering for the statement-DML tier.
Revisit when the supported ibis floor allows sqlglot ≥ 28 (tracked as XOR-449).

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
  `make_publish_with_parquet`'s "removing staging is cleanup" stance).
- Re-`register` after a `REWRITE` swap is process-local; another process holding the old registration
  keeps the pre-swap file until it re-registers.
- **A backend that declares no `publish_strategy` silently routes to `REWRITE`** — safe but
  O(final), and invisible: nothing warns that a capable engine is doing a full rewrite. Out-of-tree
  backends opt in by declaring the method (no `publish.py` edit needed — the point of per-backend
  declaration); the in-tree example of the silent-fallback cost is trino, which has native `MERGE`
  but no declaration yet (XOR-445).

## Open questions

Resolved by the XOR-444 implementation:

1. **Composite-key delete filter** for iceberg — `_key_filter` uses `In` for a single key and an
   `Or`-of-`And` for composite keys (pyiceberg 0.11 expression API).
2. **First-run `final` creation** — `_ensure_final(con, final, staging, columns, mode)` creates the
   target from the staging data-column schema when absent; in `REWRITE` the first-run branch writes
   the changeset directly (no anti-join against a missing `final`).
4. **Replace semantics for `REWRITE`** — `con.create_table(final, arrow, overwrite=True)` on
   datafusion/pandas is a drop→create (non-atomic — noted in the strategy table); the parquet target
   uses `to_parquet` + atomic `.replace`.
5. **Duplicate-key gate as the default audit** — **yes.** The WAP builders default `UPSERT`/`MERGE`
   to a "no duplicate keys" `audit_fn` (`_default_audit`); `APPEND` keeps always-True; the standalone
   `publish` primitives leave uniqueness a caller responsibility.
3. **Postgres version probe** — implemented on the postgres backend
   (`backends/postgres/__init__.py`, `publish_strategy`): major version from `self.version`,
   `NATIVE_MERGE` on 15+, else `STATEMENT_DML`.

Remaining (XOR-445):

- **Trino** — more than the one-line declaration: (a) trino has no `read_record_batches`, so it
  cannot be a tee **sink** for staging — the WAP write path needs that first; (b) trino `MERGE`
  works only on ACID connectors (iceberg/delta/hive-ACID), so `publish_strategy` needs connector
  detection + a guard/fallback, then the gated integration test. Until then trino declares no
  strategy and the standalone `publish()` falls back to `REWRITE` (debug-logged).
- **Databricks live verification** — code + tests are in-tree; run them against a live workspace
  once credentials are available.

## References

- ADR-0014: TeeNode, deferred write as a side effect — `docs/adr/0014-teenode-deferred-writes.md`
- WAP library: `python/xorq/writes/wap.py` (`make_wap_expr`, `make_publish_with_parquet`, `make_publish_with_iceberg`)
- WriteThrough sinks: `python/xorq/writes/write_through.py` (`BackendWriteThrough`, `_supports_mode:198`)
- pyiceberg backend: `python/xorq/backends/pyiceberg/__init__.py` (`upsert:197`, `publish_staging_table:398`, `read_record_batches:314`)
- xorq_datafusion backend: `python/xorq/backends/xorq_datafusion/__init__.py` (`register:449`, `to_parquet:954`, `deregister_table`)
- raw_sql / begin: `vendor/ibis/backends/{duckdb,postgres,sqlite}/__init__.py`
- `lazy_singledispatch` (evaluated, not used — imports the whole top-level-package bucket on first dispatch): `python/xorq/vendor/ibis/common/dispatch.py`; in use by `backends/datafusion/__init__.py:149`, `backends/pandas/__init__.py:357`
- examples: `examples/wap_audit_parquet.py`, `examples/wap_audit_iceberg.py`, `examples/wap_incremental_duckdb.py` (WAP upsert/merge + audit gate), `examples/publish_incremental_duckdb.py` (standalone `publish`, no WAP)
