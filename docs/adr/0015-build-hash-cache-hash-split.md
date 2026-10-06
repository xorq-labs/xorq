# ADR-0015: Every op modifies the build hash; cache-hash neutrality is the exception

- **Status:** Accepted
- **Date:** 2026-06-22
- **Deciders:** Dan Lovell

## Context

Xorq computes two hashes from the same expression tokenizer, applied in different contexts:

- The **build hash** (`get_expr_hash` in `provenance_utils.py`) identifies the build artifact.
  It is the key a catalog search returns when someone asks "has this pipeline been built?"
- The **cache hash** (`expr.ls.tokenized` for the modification-time strategy; the snapshot
  strategy tokenizes through its own context in `caching/strategy.py`) determines cache
  hits. It is the key a `CachedNode` checks before deciding whether to recompute.

Both hashes are produced by the same structural tokenizer in `dasher/_opaque.py`. The
difference is which ops participate; *How the split is implemented* below says how. (The
sentence that stood here credited a strip pass that is on neither hash path; see Errata.)

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
| `Tag` | Metadata annotation; schema and rows unchanged | the tokenizer's SQL step: `to_sql` runs `_remove_tag_nodes`; re-collected, but folded only under `_include_build_only_nodes` (mechanism 2) |
| `TeeNode` | Side-effect write; schema and rows unchanged | the tokenizer's SQL step: `to_sql` runs `_remove_tee_nodes`; re-collected, but folded only under `_include_build_only_nodes` (mechanism 2) |

At execution a `TeeNode` is not stripped: it is kept until the tee pass fires its write
(ADR-0014).

`HashingTag` is the counter-example: it is a `Tag` subclass whose metadata **does**
participate in the cache hash (via `__dasher_tokenize__`), because its metadata is
intended to distinguish otherwise-identical expressions.

### How the split is implemented

The build hash and cache hash share the same tokenizer. The split is achieved by two
mechanisms:

1. **Strip passes** inside the tokenizer. Both hashes tokenize through `_decompose_expr`
   (`dasher/_opaque.py`), whose structural component is the SQL that `to_sql` compiles, and
   `to_sql` first runs `_remove_tag_nodes` and `_remove_tee_nodes`. So no plain `Tag`,
   `HashingTag` or `TeeNode` reaches the SQL component of either hash (a `CacheTag` reaches
   it as the placeholder `_xorq_opaque_to_placeholder` rewrites it to). `_decompose_expr`
   then re-collects, by explicit type, `HashingTag`, `CacheTag`, plain `Tag` and `TeeNode`
   with `walk_nodes`; `_hash_expr_components` folds the first two into every hash and plain
   `Tag` and `TeeNode` only under `_include_build_only_nodes` (mechanism 2). Membership in
   a hash is that explicit re-collection, not the presence of a `__dasher_tokenize__` (see
   Errata).

   `_remove_non_hashing_tag_nodes` (strips `Tag` and `TeeNode`, keeps `HashingTag` and
   `CacheTag`) is on neither hash path. It backs `expr.ls.untagged`, which `node_utils.py`
   and `content_hash.py` tokenize for their own purposes.

   `_remove_tag_nodes` strips *all* `Tag` nodes including `HashingTag`. `to_sql` runs it,
   so it runs wherever the tokenizer runs, including while the execution transform's
   `cache` pass keys and stamps. `_transform_expr` does not apply it to the expression it
   transforms: that runs the same replacer as the `remove_tags` record of `_PASSES`, at the
   execution boundary (see the first Amendment below). The distinguishing rule is that
   `HashingTag` is folded on both hash paths and plain `Tag` on the build path only.

2. **The `_include_build_only_nodes` context variable** (`dasher/_opaque.py`) controls
   whether `_hash_expr_components` folds the build-hash-only families, `TeeNode` writer
   identity and plain `Tag` metadata, into the structural hash. `get_expr_hash` enters the
   `include_build_only_nodes()` context manager (which sets it to `True`); the cache path
   leaves it `False`. This is how both families are build-hash-bearing but
   cache-hash-neutral without two separate tokenizers.

### Opaque sub-expressions participate via tokenizer descent, not manual walks

The build-hash invariant ("every op participates in the build hash") must hold even for
ops buried inside *opaque sub-expressions* — fields the native ibis graph walk does not
traverse: `RemoteTable.remote_expr`, `CachedNode.parent`, `FlightExpr.input_expr`,
`FlightUDXF.input_expr`, and `ExprScalarUDF.computed_kwargs_expr`. These are members of the
`opaque_ops` tuple, which is derived from the `OPAQUE_SPECS` registry in `graph_utils.py`
(ADR-0016); the registry is authoritative for which fields are opaque. Two registry members
are not descent cases. `Read` has no wrapped sub-expression — its opaque content is the
`read_kwargs` path, whose hash-path/read-path split is the subject of ADR-0006. `CacheTag`
is a hash *leaf*: a pinned cache read whose `__dasher_tokenize__` is its cache key, so what
sits only under its `uncached` payload is pruned by `exclusively_pinned_leaves` and does not
fold into either hash, by design. Like `HashingTag` it survives
`_remove_non_hashing_tag_nodes`; the execution-path strip replaces it with its `parent`,
the materialized cache read.

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
`normalize_cached_node` / `normalize_scalar_udf` rules in the external `xorq_dasher` package;
for the types above, what folds into the hash is decided in-repo. Read the in-repo
`dasher/_opaque.py`, not `xorq_dasher`, to see what decides the hash.

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
side-effecting transforms (the `BOUNDARY` records of `_PASSES` in `api.py`, which use
`op.replace` so opaque sub-exprs get their own pass), or non-hashing traversals (lineage, schema
validation).

When a manual walk over an expression *is* needed for hashing-adjacent work, locate nodes
and delegate the hash to the strategy's tokenizer; never re-implement opaque descent.

### Requirements for new ops

A new op that is a transparent pass-through (schema equals parent, rows unchanged) and
exists only for a side effect **may** be cache-hash-neutral. To add one:

1. The op must implement `__dasher_tokenize__` returning a tuple of its identity-bearing
   fields (so the build hash includes it). A tokenize rule alone admits an op to neither
   hash: it must also be re-collected by `_decompose_expr` (mechanism 1; see Errata).
2. It must be dropped from the tokenizer's SQL component (today: by `_remove_tag_nodes`
   or `_remove_tee_nodes` inside `to_sql`) and not folded into the cache hash: either not
   re-collected by `_decompose_expr` at all, or re-collected and folded only under
   `_include_build_only_nodes` (requirement 3).
3. If the op needs to participate in the build hash but not the cache hash (like `TeeNode`
   and plain `Tag`), its fold in `_hash_expr_components` must be gated behind
   `_include_build_only_nodes` so only the build-hash path includes it.

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
- Context variable toggle: `_include_build_only_nodes` in `python/xorq/common/utils/dasher/_opaque.py`
- Tokenizer-side strips (run by `to_sql`): `_remove_tag_nodes`, `_remove_tee_nodes` in `python/xorq/expr/api.py`
- Execution-path tag strip: the `remove_tags` record of `_PASSES` in `python/xorq/expr/api.py` (BOUNDARY; see the Amendment)
- `ls.untagged` strip (keeps `HashingTag` and `CacheTag`; on neither hash path): `_remove_non_hashing_tag_nodes` in `python/xorq/expr/api.py`
- Tokenizer decomposition (SQL component plus re-collected identity nodes): `_decompose_expr` in `python/xorq/common/utils/dasher/_opaque.py`
- Hash component assembly: `_hash_expr_components` in `python/xorq/common/utils/dasher/_opaque.py`
- Canonical hasher and rule overrides: `HASHER = DEFAULT_HASHER.override(*_EXTRA_RULES)` in `python/xorq/common/utils/dasher/__init__.py`
- Opaque sub-expr descent (the rule that actually runs): `_xorq_opaque_to_placeholder`, `_normalize_expr_xorq`, `_normalize_scalar_udf_xorq` in `python/xorq/common/utils/dasher/_opaque.py` — these override the upstream `normalize_remote_table` / `normalize_cached_node` / `normalize_scalar_udf` in the external `xorq_dasher` package
- Opaque op set: `opaque_ops` in `python/xorq/common/utils/graph_utils.py`
- ADR-0006: `read_kwargs` hash-path/read-path split
- ADR-0014: TeeNode deferred writes (the specific design that prompted this general rule)

## Amendment: the execution-path strip stops at payload boundaries

Mechanism 1, as first written, said `_remove_tag_nodes` runs on the execution path "where
the tag metadata is irrelevant to the rows produced" (that sentence is gone; see Errata).
Rows, yes; keys, no. The execution transform
keys every `CachedNode` it meets (the `cache` pass calls `calc_key` on `CachedNode.parent`),
and that key is subject to this ADR's rule: anything feeding the cache hash keeps
`HashingTag`. The strip pass descended into `CachedNode.parent` ahead of the cache pass, so
execution keyed a parent with its `HashingTag` removed while `ls.get_key()`,
`cache_exists()` and the build metadata keyed the parent as written. The artifact landed
under one key and was looked up under another, and two expressions differing only in
hashing-tag metadata shared one entry.

The decision stands; this records the invariant it implies for the execution transform:

- **A tag strip that runs ahead of a keying pass stops at opaque payload boundaries.**
  The execution path's strip is the `remove_tags` record of `_PASSES` in
  `python/xorq/expr/api.py`, and it is `BOUNDARY`; `Traversal` in
  `python/xorq/expr/enums.py` carries the rule. It still removes every tag, `HashingTag`
  included, from the tree the current execution compiles. The function
  `_remove_tag_nodes` shares that record's replacer but always descends; `_transform_expr`
  does not apply it to the expression it transforms.
- **Each executed payload is keyed as written, then stripped by its own nested transform.**
  The owning pass keys the payload as written; the payload's own `_transform_expr` entry
  then strips it. Which payload fields re-enter is recorded on `Traversal` in
  `python/xorq/expr/enums.py`. `OPAQUE_SPECS` says which fields are opaque to descent, not
  which re-enter: `CacheTag.uncached` is opaque and never re-enters.

Alternatives rejected on the way:

- *Leave `HashingTag` in place during the strip.* No compiler has a rule for a tag; every
  hashing-tagged execution fails with `OperationNotDefinedError`.
- *Run `cache` before the strip, with the strip still descending.* The outer cache pass
  never enters a payload but the strip does, so a cache nested inside a `RemoteTable`
  payload was still keyed from a stripped parent, and an outer plain tag hid the cache
  root from provenance stamping.
- *A separate `remove_hashing_tags` pass after `cache`.* Works only if that pass is also
  `BOUNDARY`, and then an outer hashing tag hides the cache root from provenance unless
  the root check is patched too. That is this change with an extra pass.

The one pass that still rewrites a payload before it is keyed is `bind_params`. That is
accepted; the `_PASSES` header in `python/xorq/expr/api.py` records why it cannot stop at
the boundary and what it costs.

## Errata

Corrections to the descriptive text above, found by a cold read and each checked by
running the code. The decision is unchanged; these fix what the document said the code
does.

- **`opaque_ops` has seven members, not six**, and `CacheTag` was missing from this
  document entirely. `Read` and `CacheTag` are not descent cases: one has no
  sub-expression, the other is a hash leaf. The list and count in *Opaque sub-expressions
  participate via tokenizer descent* are corrected and now defer to `OPAQUE_SPECS`.
- **`_remove_non_hashing_tag_nodes` is not on the cache-hash path.** Mechanism 1 and the
  neutrality table credited it with stripping `Tag` before the cache hash. Patching it and
  counting calls shows zero during `ls.tokenized`, `ls.get_key` (snapshot and
  modification-time strategies) and `get_expr_hash`; its only caller is `ls.untagged`.
  Plain-tag neutrality comes from `to_sql` stripping tags inside the tokenizer's SQL step
  and `_decompose_expr` re-collecting only `HashingTag`, `CacheTag` and `TeeNode` (the
  last folded only under the build-only gate, in `_hash_expr_components`). Mechanism 1,
  the table, requirement 2 and the References are corrected.
- **`_transform_expr` does not apply `_remove_tag_nodes`.** The document said it runs in
  `_transform_expr`. `_remove_tag_nodes` is a descending `replace_nodes`; `to_sql` and the
  YAML compiler's `_extract_sql_queries` call it, and through `to_sql` the tokenizer calls
  it, including while the execution transform keys and stamps a cache. The execution
  transform itself runs the same replacer as a BOUNDARY pass (the Amendment above).
  Mechanism 1 and the References are corrected.
- **A plain `Tag` is neutral in the build hash too.** `get_expr_hash(t.tag("v1"))` equals
  `get_expr_hash(t)`, while a `HashingTag` changes it. The invariant as stated ("every op
  participates in the build hash") does not cover this, and the `HashingTag` docstring
  treats its parent's neutrality as designed. Requirement 1 is not the mechanism either: a
  `Tag` subclass given a `__dasher_tokenize__` is still neutral in both hashes, because
  `_decompose_expr` re-collects by an explicit type list, and plain `Tag` was on it
  nowhere. Decided: plain `Tag` is build-hash-bearing; see the second Amendment below.
- **Corrections to the previous errata round.** It placed the `TeeNode` gate in
  `_decompose_expr` (it is in `_hash_expr_components`), named `to_sql` as the only caller
  of `_remove_tag_nodes` (the YAML compiler is a second), listed `CacheTag.uncached` among
  the fields the build-hash invariant reaches (a pin prunes what is under it), and implied
  that a tokenize rule is what admits an op to a hash. Each is corrected above. The Context
  paragraph that credited the `ls.untagged` strip, and the Amendment's quotation of a
  mechanism-1 sentence that round had deleted, are retired too.

## Amendment: a plain `Tag` is build-hash-bearing

The Errata above record that a plain `Tag` was neutral in both hashes, which the invariant
("every op participates in the build hash") did not cover. The invariant stands and the
code now honours it for `Tag`:

- `Tag.__dasher_tokenize__` returns `("tag", schema, metadata)` (`python/xorq/expr/relations.py`).
  The leading literal differs from `HashingTag`'s, so a plain tag and a hashing tag with
  equal metadata do not share a token.
- `_decompose_expr` re-collects plain tags (`walk_nodes(Tag, ...)`, excluding the
  `HashingTag` and `CacheTag` subclasses, which have collections of their own) and prunes
  them under a `CacheTag` pin exactly as it prunes the other leaves. A future `Tag`
  subclass with no collection of its own is collected here as a plain tag, so the
  invariant holds for it by default.
- `_hash_expr_components` folds the plain-tag tokens only under `_include_build_only_nodes`,
  the context variable formerly named `_include_tee_nodes`, now gating both build-hash-only
  families (mechanism 2). `get_expr_hash` enters it; the cache path never does.

So a plain `Tag` is cache-hash-neutral as before (the `to_sql` strip and an unset gate) and
build-hash-bearing now. `HashingTag` and `CacheTag` behaviour is unchanged on both paths.
`test_provenance_utils.py` pins the three facts: a plain tag changes the build hash, a plain
tag leaves `ls.tokenized` and `ls.get_key()` unchanged, and a plain tag and a hashing tag
with equal metadata give different build hashes.

Consequence: every tagged expression's build hash moves. A catalog entry keyed on the
pre-change build hash of a tagged expression no longer matches a search for that
expression and must be rebuilt or re-keyed.

The alternative, exempting plain `Tag` from the invariant in this document, was rejected:
the invariant is the decision this ADR records, and two expressions that differ only in tag
metadata are different pipelines to a catalog search.
