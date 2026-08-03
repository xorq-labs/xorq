# ADR-0016: Table-driven opaque descent with registration tripwires

- **Status:** Accepted
- **Date:** 2026-08-03
- **Deciders:** dlovell, mesejo

## Context

Several op types hold sub-expressions that ibis's `__children__` protocol does
not surface: their children are `Expr`-typed args (which
`_flatten_collections` skips — it only yields `Node`s) or live in
`__config__`. Graph traversal — hashing, lineage, source discovery, structural
rewrites — must know about these "opaque" ops explicitly or it silently skips
their subtrees.

Before #2177, that knowledge was hand-written ~5 times: `gen_children_of`'s
match statement, two read-side policy variants, and `replace_nodes`'s
write-side match. Each new opaque op or descent policy meant another copy, and
the copies could drift (an op added to `opaque_ops` but missing a match arm,
or read and write sides disagreeing about an edge).

#2177 unified these behind one table. Its review retrospective then identified
the remaining gap: the table protects only *registered* ops. A future op
holding an `Expr`-typed child would silently fall through to `__children__`
— wrong hashes, lineage, and source discovery, with no failure anywhere.
Silent non-descent is strictly worse than any loud failure mode, and every new
opaque op was a fresh chance to hit it.

## Decision drivers

- Drift between the op-type list, read-side edges, and write-side edges must
  be impossible, not merely tested.
- Forgetting to register a new opaque op must fail loudly, not silently
  corrupt hashes.
- "Deliberately not descended" and "forgot to consider" must be
  distinguishable in the code.
- Traversal is a hot path; per-node overhead must stay constant-factor small.

## Decision

### One table, everything derived

`OPAQUE_SPECS` (`common/utils/graph_utils.py`) maps each opaque op type to a
frozen `OpaqueSpec` carrying everything traversal needs: `read_edges`,
`write_edges` (`None` = same as read), `forward_kwargs`, `rebind`. All other
tables are derived from it — `opaque_ops`, the read-side `OPAQUE_EDGES`, the
descent-policy overrides, and the per-type recorded-`Expr`-field sets — so
they cannot drift. Tables are `MappingProxyType`-wrapped; spec invariants
(e.g. `rebind` requires exactly one write-side edge) are validated at
construction, i.e. import time.

Both traversal sides resolve ops through a single `isinstance`-matching lookup
(`_opaque_lookup`), so read and write cannot diverge on how they classify an
op. A co-located test pins that the opaque types are mutually non-subclassing,
which is what makes the lookup order-independent.

### Fail loudly, at the right layer

- Edge fields are read unguarded: a stale name raises `AttributeError`, a
  `None` edge raises `ValueError`. Pruning an edge means removing it from a
  table, never nulling it.
- **Unregistered-op tripwire**: on the spec-less branch of both
  `gen_children_of` (read) and `replace_nodes` (write), an op holding
  `Expr`-valued args raises. This catches ops whose `Expr` payload hides
  behind `Any` annotations (the `CachedNode.parent` / `CacheTag.uncached`
  shape), which static inspection cannot see.
- **Registered-op tripwire**: on the spec branch of both paths, every
  `Expr`-valued arg must be a descent edge or listed in
  `NON_EDGE_EXPR_FIELDS` — so a new `Any`-typed `Expr`-holding field cannot
  grow on an already-registered op unnoticed.
- **Static completeness test**: every op class in `expr/relations.py` and
  `expr/udf.py` with an `Expr`-annotated field must have a spec, and each such
  field must be an edge or a recorded non-edge.

`NON_EDGE_EXPR_FIELDS` records deliberate exclusions with the reason attached
(sole entry: `FlightExpr.unbound_expr`, which executes server-side, not in the
outer graph). An exclusion is a decision with a paper trail, not an omission.

### Known blind spot

An `Expr` living only in `__config__` (the `ExprScalarUDF` shape) never
appears in `__args__` or annotations, so neither tripwire nor the completeness
test can see it. Registering such ops is enforced by review, not machinery.
This ADR is the durable record of that obligation.

## Alternatives considered

### Keep a runtime "unhandled opaque op" guard in `replace_nodes`

The pre-#2177 design raised `ValueError` when an op was in `opaque_ops` but
had no match arm.

Rejected because:
- With `opaque_ops` derived from `OPAQUE_SPECS` and both resolved by the same
  lookup, "no spec" and "not opaque" became the same condition — the guard was
  unreachable by construction, and its monkeypatch-based test had silently
  stopped testing anything.

### Static completeness test only (no runtime tripwires)

Rejected because:
- Two of the six edge-bearing ops annotate their `Expr` fields as `Any`;
  static inspection cannot see them. Only an instance-level check catches that
  shape, and only at traversal time.

### Declaration-site registration

The spec lives on the op class itself (a class attribute or
`__init_subclass__` hook on a marker base), and `graph_utils` collects rather
than maintains the table. Locality would make the gap unrepresentable: you
could not define an opaque op without confronting the traversal question in
the same diff.

Deferred because:
- It inverts the dependency between op modules and `graph_utils` and is best
  designed together with XOR-363, which is already reshaping descent policies.
  The tripwires provide the safety net until then.

## Consequences

### Positive

- Type list, read table, write table, and recorded-field sets cannot drift —
  they are the same table.
- Forgetting to register an opaque op (or a new `Expr` field on one) fails at
  first traversal with an actionable message, instead of silently producing
  wrong hashes/lineage.
- Descent policies are data (`{**OPAQUE_EDGES, CacheTag: ()}`), not function
  bodies; XOR-363's flight-leaf policy was added as one line.

### Negative

- ~25–30% constant-factor overhead on pure traversal microbenchmarks (no
  measurable wall-clock movement in real suites; traversal is never the
  dominant cost). A sound per-type memo is impossible because `Any`-typed
  fields make `Expr`-bearing an instance property, not a type property.
- The `__config__` blind spot remains review-enforced.
- External code defining `Expr`-bearing ops without specs now hard-fails
  instead of silently mis-hashing — intended, but it is a behavior change for
  any such (unsupported) usage.

## References

- #2177 — the table refactor (`OPAQUE_SPECS`, derived tables, policies)
- #2196 — the tripwires, completeness test, and `NON_EDGE_EXPR_FIELDS`
- XOR-363 — compact/expandable lineage; prospective home of declaration-site
  registration
- `CONTRIBUTING.md` § "Removing or changing behavior" — the process
  conventions distilled from the same review retrospective
