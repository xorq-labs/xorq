# ADR-0015: Every op modifies the build hash; cache-hash neutrality is the exception

- **Status:** Accepted
- **Date:** 2026-06-22
- **Deciders:** Dan Lovell

## Context

Xorq computes two hashes from the same expression tokenizer, applied in different contexts:

- The **build hash** (`get_expr_hash` in `provenance_utils.py`) identifies the build artifact.
  It is the key a catalog search returns when someone asks "has this pipeline been built?"
- The **cache hash** (`expr.ls.tokenized`) determines cache hits. It is the key a
  `CachedNode` checks before deciding whether to recompute.

Both hashes are produced by the same structural tokenizer in `dasher/_opaque.py`. The
difference is which ops participate. Today that difference is controlled by a context
variable (`_include_tee_nodes`) toggled by `get_expr_hash`, plus a strip pass
(`_remove_non_hashing_tag_nodes` / `_remove_tee_nodes` in `api.py`) that removes certain
ops before the cache hash is computed.

The rules governing which ops participate in which hash are implicit. ADR-0014 documents
TeeNode's specific behavior (cache-hash-neutral, build-hash-bearing), and ADR-0006
documents the hash-path/read-path identity split for `Read` ops, but the general principle
is not stated anywhere. A contributor adding a new op, or an agent modifying hashing code,
has no single rule to check against.

The risk of getting this wrong is concrete: if an op does not modify the build hash, two
structurally different DAGs can produce the same build hash. A catalog search for one
pipeline could return the artifact of a different pipeline. This is a silent correctness
bug, not a performance issue.

## Decision drivers

- A build-hash collision between structurally different DAGs is a correctness bug that
  produces wrong results silently.
- Cache-hash neutrality is desirable for pure side-effect ops (the write should not
  invalidate the cache), but must be opt-in and justified.
- The rule must be discoverable by both human contributors and AI agents working on the
  codebase.

## Decision

### The build-hash invariant

**Every op in the DAG must participate in the build hash.** Different DAGs must produce
different build hashes. No op may be stripped, ignored, or otherwise excluded from the
build hash computation.

This is a structural invariant: if two expressions have different op graphs, their build
hashes must differ. The build hash is the identity of "what was built," and collisions
mean a catalog search can return the wrong artifact.

### Cache-hash neutrality is the exception

An op may be stripped from the cache hash **only** if its presence does not change the
logical result of the expression: same input rows in, same output rows out. The op exists
purely for a side effect (writing, tagging, metadata) that is orthogonal to what the
expression evaluates to.

Today two op families qualify:

| Op | Why cache-hash-neutral | Strip mechanism |
|---|---|---|
| `Tag` | Metadata annotation; schema and rows unchanged | `_remove_non_hashing_tag_nodes` in `api.py` |
| `TeeNode` | Side-effect write; schema and rows unchanged | `_remove_non_hashing_tag_nodes` and `_remove_tee_nodes` in `api.py` |

`HashingTag` is the counter-example: it is a `Tag` subclass whose metadata **does**
participate in the cache hash (via `__dasher_tokenize__`), because its metadata is
intended to distinguish otherwise-identical expressions.

### How the split is implemented

The build hash and cache hash share the same tokenizer. The split is achieved by two
mechanisms:

1. **Strip passes** run before cache-hash computation. `_remove_non_hashing_tag_nodes`
   replaces `Tag` and `TeeNode` with their parents. `_remove_tee_nodes` does the same on
   the SQL compilation path. These passes do not run on the build-hash path.

   A third, related pass is `_remove_tag_nodes`. It is **not** part of the cache-hash
   split: it strips *all* `Tag` nodes including `HashingTag` (whereas
   `_remove_non_hashing_tag_nodes` deliberately preserves `HashingTag`). It runs on the
   SQL compilation (`to_sql`) and execution-transform (`_transform_expr`) paths, where the
   tag metadata is irrelevant to the rows produced. Do not confuse it with the
   cache-hash strip pass; the distinguishing rule is that anything feeding `expr.ls.tokenized`
   must keep `HashingTag` (use `_remove_non_hashing_tag_nodes`), while SQL/execution may
   drop it (use `_remove_tag_nodes`).

2. **The `_include_tee_nodes` context variable** (`dasher/_opaque.py`) controls whether
   `_hash_expr_components` folds TeeNode writer identity into the structural hash.
   `get_expr_hash` enters the `include_tee_nodes()` context manager (which sets it to
   `True`); the cache path leaves it `False`. This is how TeeNode is build-hash-bearing
   but cache-hash-neutral without two separate tokenizers.

### Opaque sub-expressions participate via tokenizer descent, not manual walks

The build-hash invariant ("every op participates in the build hash") must hold even for
ops buried inside *opaque sub-expressions* — fields the native ibis graph walk does not
traverse: `RemoteTable.remote_expr`, `CachedNode.parent`, `FlightExpr.input_expr`,
`FlightUDXF.input_expr`, and `ExprScalarUDF.computed_kwargs_expr`. These wrap the five
sub-expression-bearing members of the `opaque_ops` tuple in `graph_utils.py`. The sixth
member, `Read`, has no wrapped sub-expression — its opaque content is the `read_kwargs`
path, whose hash-path/read-path split is the subject of ADR-0006.

The tokenizer already reaches these ops on its own. xorq's canonical `HASHER` is
`DEFAULT_HASHER.override(*_EXTRA_RULES)` (`dasher/__init__.py`): the `_EXTRA_RULES` replace
upstream `xorq_dasher` defaults with in-repo normalizers registered against the `Expr`,
`ScalarUDF`, and `Read` types. The `Expr` normalizer (`_normalize_expr_xorq`) rewrites every
opaque leaf via `_xorq_opaque_to_placeholder` (`dasher/_opaque.py`), which descends the
opaque field: the `RemoteTable` case folds in `remote_expr`, the `CachedNode` case folds in
`parent`, and `FlightExpr`/`FlightUDXF` fold in `input_expr`. `ExprScalarUDF.computed_kwargs_expr`
is folded by the `ScalarUDF` normalizer (`_normalize_scalar_udf_xorq`). An op hidden under any
of these boundaries therefore still folds into the hash; the invariant holds without help.

These in-repo overrides shadow the same-named `normalize_remote_table` /
`normalize_cached_node` / `normalize_scalar_udf` rules in the external `xorq_dasher` package —
for the types above, the upstream rules do not run when hashing through `HASHER`. Read the
in-repo `dasher/_opaque.py`, not `xorq_dasher`, to see what actually executes.

The corollary is a contributor rule: **do not write graph walks that descend into opaque
sub-expressions in order to "help" the hash.** Such walks are vestigial. They duplicate
descent the tokenizer already performs, and — because the normalizers deliberately
exclude identity-irrelevant fields — they often rewrite a field the rule ignores, making
them silent no-ops. The removed `SnapshotStrategy._replace_remote_table` was exactly this:
it rewrote `RemoteTable.name` to a content hash before tokenizing, but
`_xorq_opaque_to_placeholder`'s `RemoteTable` case ignores `name` (it folds schema,
`remote_expr`, and `source.name`), so the rewrite changed nothing. An audit of the codebase found it was the
only such walk; the remaining descending walks are either the tokenizer implementation
itself (`dasher/_opaque.py`, which must descend), node-targeting find-then-replace passes
(`node_utils.py`, which delegate hashing to `expr.ls.tokenized`), execution-time
side-effecting transforms (`register_and_transform_*`, which correctly use `op.replace` so
opaque sub-exprs get their own pass), or non-hashing traversals (lineage, schema
validation).

When a manual walk over an expression *is* needed for hashing-adjacent work, locate nodes
and delegate the hash to the strategy's tokenizer; never re-implement opaque descent.

### Requirements for new ops

A new op that is a transparent pass-through (schema equals parent, rows unchanged) and
exists only for a side effect **may** be cache-hash-neutral. To add one:

1. The op must implement `__dasher_tokenize__` returning a tuple of its identity-bearing
   fields (so the build hash includes it).
2. The strip pass (`_remove_non_hashing_tag_nodes` or a new dedicated pass) must replace
   it with its parent before the cache hash is computed.
3. If the op needs to participate in the build hash but not the cache hash (like TeeNode),
   it must be gated behind a context variable or equivalent mechanism so the build-hash
   path includes it.

A new op that changes the logical result (filters rows, adds columns, transforms values)
must participate in **both** hashes. This is the default; no special action is needed
beyond the normal structural tokenization path.

### Identity-neutral fields

Within an op that participates in hashing, individual fields may be excluded from the
identity if they tune execution mechanics without changing the logical result. Examples:

- `BackendWriteThrough.kwargs` (`hash=False, eq=False`): tunes write mechanics
  (compression, batch size), not the rows.
- `ThreadedBackendWriteThrough.maxsize` (`hash=False, eq=False`): transport tuning.
- `TeeNode.drain`: execution-time concern that does not change the logical result.

The invariant is: if changing the field's value would change which rows appear in the
output, the field must be identity-bearing. If it only changes *how* those rows are
produced or delivered, it may be excluded.

## Alternatives considered

### Document the rule only in code comments

Scatter the invariant across docstrings on `get_expr_hash`, `_hash_expr_components`,
and `__dasher_tokenize__`.

**Rejected.** Code comments are authoritative for *how* but not for *why*. The general
principle ("every op modifies the build hash because collisions are a correctness bug")
is an architectural decision that spans multiple files and belongs in the ADR system where
contributors and agents look for cross-cutting rules.

### Enforce the invariant with a test or lint

A test that walks all `ops.Relation` subclasses and asserts each either participates in
the structural hash or is on an explicit allow-list of cache-hash-neutral ops.

**Deferred.** Worth doing, but the documentation is the prerequisite. An allow-list test
without a stated rule is just a gate; the rule tells contributors *why* the gate exists
and how to evaluate whether a new op belongs on the list.

### Merge this into ADR-0014

ADR-0014 already discusses TeeNode's hash behavior in detail.

**Rejected.** ADR-0014 is about the TeeNode/WriteThrough design. The build-hash
invariant is a project-wide rule that predates TeeNode and applies to all ops. Embedding
it in a feature-specific ADR buries a general principle under a specific design.

## Consequences

### Positive

- Contributors and agents have a single, findable rule for how ops interact with hashing.
- The distinction between build hash (must never collide) and cache hash (may collapse
  side-effect-only ops) is explicit rather than implicit.
- New cache-hash-neutral ops require conscious justification against the stated criteria,
  reducing the risk of accidental build-hash collisions.
- The identity-neutral field convention (`hash=False, eq=False` for mechanics-only
  fields) is documented alongside the hash invariant it depends on.

### Negative

- A second ADR about hashing (alongside ADR-0006) adds surface area. Mitigated by
  cross-referencing: ADR-0006 is about the read-path/hash-path split within `Read` ops;
  this ADR is about the build/cache split across all ops.
- The "deferred enforcement test" is not shipped with this ADR, so the rule is
  documentation-only until that test lands.

## References

- Build hash entry point: `get_expr_hash` in `python/xorq/common/utils/provenance_utils.py`
- Context variable toggle: `_include_tee_nodes` in `python/xorq/common/utils/dasher/_opaque.py`
- Strip passes: `_remove_non_hashing_tag_nodes`, `_remove_tee_nodes` in `python/xorq/expr/api.py`
- SQL/execution-path tag strip (preserves no tags, including `HashingTag`): `_remove_tag_nodes` in `python/xorq/expr/api.py`
- Hash component assembly: `_hash_expr_components` in `python/xorq/common/utils/dasher/_opaque.py`
- Canonical hasher and rule overrides: `HASHER = DEFAULT_HASHER.override(*_EXTRA_RULES)` in `python/xorq/common/utils/dasher/__init__.py`
- Opaque sub-expr descent (the rule that actually runs): `_xorq_opaque_to_placeholder`, `_normalize_expr_xorq`, `_normalize_scalar_udf_xorq` in `python/xorq/common/utils/dasher/_opaque.py` — these override the upstream `normalize_remote_table` / `normalize_cached_node` / `normalize_scalar_udf` in the external `xorq_dasher` package
- Opaque op set: `opaque_ops` in `python/xorq/common/utils/graph_utils.py`
- ADR-0006: `read_kwargs` hash-path/read-path split
- ADR-0014: TeeNode deferred writes (the specific design that prompted this general rule)
