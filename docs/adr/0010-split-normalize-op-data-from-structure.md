# ADR-0010: Split data-dependent tokens out of expression normalization

- **Status:** Amended (compute_expr_token / expr_metadata not shipped — see below)
- **Date:** 2026-05-19
- **Deciders:** Dan, Pierre

> **Implementation status (2026-05-20):** The cheap-recompute protocol
> described in §"`compute_expr_token`" and §"References" was not shipped.
> ``expr_metadata`` was never added to ``dasher/`` and the
> ``_normalize_expr_xorq_impl`` Expr-rule return shape was never converted
> to the documented ``("ibis.Expr.v3", structural_hash, *slot_hashes)``
> tuple-of-hex-strings format.  The half-written ``dasher/_recompute.py``
> file (compute_expr_token only) was deleted as dead code; with no
> producer of the v3 shape it had no callers.  This ADR is retained for
> historical context; treat the §"`compute_expr_token`" and
> §"Metadata schema" sections as describing a design that was deferred,
> not the current API.

## Context

`dask.base.tokenize(expr)` drives xorq's caching — if the token changes, the
cache misses.  Today `normalize_op` (`dask_normalize_expr.py:713-740`) tangles
data-dependent values (table contents) with structural values (SQL shape,
schemas, UDFs) into a single flat tuple:

1. `opaque_node_replacer` replaces opaque nodes (RemoteTable, Read, …) with
   `UnboundTable(name=dask.base.tokenize(node))`, embedding content hashes as
   table names in the generated SQL.
2. `op.find(ir.DatabaseTable)` re-finds the same data nodes and includes their
   normalizations again in the output tuple.

Because the SQL string itself contains content hashes, there is no way to
predict the token for a new table without re-running the full pipeline:
expression compilation, `into_backend`, `op.replace`, SQL generation, and
recursive normalization of every node.

## Decision drivers

- Substituting one table in a multi-table expression should cost O(size of new
  table), not O(full normalization pipeline).
- The structural part of a token (SQL shape, join predicates, filters, schemas)
  must be stable across data changes so it can be computed once and reused.
- No string substitution in SQL strings — the unbound expression should
  naturally produce data-free SQL.
- No backward compatibility required — a cache break is acceptable.
- Token recomputation must be possible without importing xorq, given a
  serialized metadata artifact and the new data dep hash.

## Decision

Split `normalize_op`'s implementation into a leaf-collection step and a
structural-normalization step, returning a `(data_deps, structural)` tuple.
The old `normalize_op(op, compiler)` signature is preserved as a thin wrapper
delegating to the new `normalize_op_split` so existing callers keep working.

### `normalize_expr` return shape

The registered `dask.base.normalize_token` handler for `ibis.expr.types.Expr`
changes its return value from a flat normalized tuple to:

```python
(tuple(dask.base.tokenize(d) for d in data_deps), dask.base.tokenize(structural))
```

— a `(tuple_of_hex_slot_hashes, hex_structural_hash)` pair.  Pre-hashing
each component to its hex string lets external callers reproduce the final
token from a serialized metadata artifact using only `hashlib` (see
`compute_expr_token` below).

### `normalize_op_split`

A new function that exposes the split for callers that need to inspect or
substitute individual data deps:

```python
def normalize_op_split(expr):
    """
    Returns (leaf_dts, data_deps, structural).

    leaf_dts    — data leaf ops in walk order: plain DatabaseTable,
                  InMemoryTable, and Read.  Reached *through* opaque
                  sub-expressions (RemoteTable.remote_expr, CachedNode.parent,
                  FlightExpr/UDXF.input_expr, ExprScalarUDF.computed_kwargs_expr)
                  so cross-engine data dependencies and ML-pipeline training
                  inputs appear as leaves.
    data_deps   — tuple of normalized tokens, one per leaf in leaf_dts.
    structural  — normalized token for the data-free expression shape.
                  Opaque sub-expressions are placeholdered by their *own*
                  structural token, computed with their own backend's
                  compiler (so backend-specific ops like DuckDB's ArrayFilter
                  still tokenize correctly inside a cross-engine RemoteTable).
    """
```

`normalize_expr` calls this and pre-hashes each component to a hex string
before returning.

### Mechanism

```python
from xorq.common.utils.graph_utils import walk_nodes

LEAF_TYPES = (ir.DatabaseTable, ir.InMemoryTable, rel.Read)

def is_data_leaf(node):
    # exclude DatabaseTable subclasses (RemoteTable, CachedNode, FlightExpr/UDXF)
    return (
        type(node) is ir.DatabaseTable
        or isinstance(node, ir.InMemoryTable)
        or isinstance(node, rel.Read)
    )

def normalize_op_split(op_or_expr, compiler=None):
    op = op_or_expr.op() if hasattr(op_or_expr, "op") else op_or_expr
    if compiler is None and hasattr(op_or_expr, "op"):
        compiler = get_compiler(op_or_expr)

    # 1. Walk the graph crossing opaque boundaries; collect data leaves.
    leaf_dts = tuple(n for n in walk_nodes(LEAF_TYPES, op) if is_data_leaf(n))

    # 2. Per-leaf data tokens.  InMemoryTable has no registered
    #    normalize_token handler so route through normalize_inmemorytable.
    data_deps = tuple(_normalize_data_leaf(dt) for dt in leaf_dts)

    # 3. Structural side: single-pass rewrite that strips data leaves and
    #    replaces opaque sub-expressions with their *own structural* token
    #    (computed with their own compiler — see opaque_node_replacer).
    structural = _normalize_structural(op, compiler=compiler)
    return leaf_dts, data_deps, structural
```

### Why the xorq `walk_nodes` here

A cross-engine expression like
`xo.deferred_read_parquet(path).into_backend(xo.connect())` wraps a `Read` in
a `RemoteTable.remote_expr`.  ibis's built-in `op.find` does **not** descend
into `Expr`-typed fields, so the inner `Read` would never appear in the leaf
list — swapping the underlying file would not invalidate the cache.
`walk_nodes` from `graph_utils.py` uses `gen_children_of`, which knows about
the opaque fields (`remote_expr`, `parent`, `input_expr`,
`computed_kwargs_expr`) and reaches the inner leaves.  Same for an
`InMemoryTable` inside an `ExprScalarUDF.computed_kwargs_expr` (the
training-data dependency of `FittedPipeline.predict`).

### Why opaque sub-expressions are tokenized with their own compiler

After the outer rewrite has replaced inner `DatabaseTable`s with
`UnboundTable`, calling `dask.base.tokenize(modified_remote_expr)` would
fall back to the default backend's compiler — which fails on
backend-specific ops (DuckDB's `ArrayFilter` cannot be compiled by the
DataFusion default compiler).  `opaque_node_replacer` therefore takes the
structural-only token of the *original* sub-expression using the
sub-expression's own compiler (`_opaque_structural_name`,
`dask_normalize_expr.py:711`).

This contains the recursion correctly:  the inner sub-expression's data
leaves still flow into the *outer* `data_deps` (already collected in step 1
by `walk_nodes`), and only its structural shape contributes to the outer
`structural` hash via the opaque-node placeholder name.

### Why `type(n) is ir.DatabaseTable`, not `isinstance`, for the
DatabaseTable check

RemoteTable, CachedNode, FlightExpr, FlightUDXF are all `DatabaseTable`
subclasses, but they carry structural information (inner sub-expressions,
cache metadata) that must stay in the expression tree to drive the
opaque-node placeholdering.  The identity check ensures we only treat
plain `DatabaseTable` instances as data leaves.  `InMemoryTable` and `Read`
use `isinstance` because they don't have similarly-confusing subclasses.

### Cheap substitution (with xorq)

Call `normalize_op_split` (`dask_normalize_expr.py:844`), look up the target
slot by index (not name — names may collide across backends or memtables),
replace its entry in `data_deps`, and `tokenize` the result.

**Cost**: one `normalize_memory_databasetable` call (serialize + hash the new
table's Arrow batches).

**Skipped**: `register`, `into_backend`, expression compilation, `op.replace`,
SQL generation, normalization of all other nodes.

### Serializable metadata for token computation without xorq

The split produces components that reduce to hex strings.  These can be
serialized as a lightweight metadata artifact and used to compute expression
tokens without importing xorq — only `hashlib` is needed.

**Metadata schema (version 2):**

```json
{
  "version": 2,
  "structural_hash": "38317617c8a70d3a...",
  "slots": [
    {"index": 0, "name": "awards_players", "hash": "a5565158c996e82e..."},
    {"index": 1, "name": "batting",        "hash": "94ac8427b43abdf2..."}
  ]
}
```

`structural_hash` is `dask.base.tokenize(structural)`.  Each slot `hash` is
`dask.base.tokenize(data_dep)` for the corresponding leaf table.  Both are
32-character MD5 hex strings.

`index` is the unambiguous slot key.  `name` is a human-readable hint and
may collide across slots — two `DatabaseTable` ops can share a name across
backends, two `InMemoryTable` ops can both default to `"ibis_pandas_…"`,
etc.  Cheap-substitution callers should look up by `index`, optionally
disambiguating by content of `leaf_dts[index]` if they want to verify the
slot identity before substituting.

Metadata is produced by `expr_metadata` (`dask_normalize_expr.py:930`).
Tokens are recomputed from metadata by `compute_expr_token`
(`dask_normalize_expr.py:968`), which only needs `hashlib`.  This works
because `dask.base.tokenize(expr)` reduces to
`md5(str(_normalize_seq_func((handler_return,))))`, and when `handler_return`
is a tuple of identity-normalized hex strings, the whole pipeline collapses
to a single `md5` over the literal preimage.

### Scope: what counts as a data leaf vs. a structural carrier

`DatabaseTable`, `InMemoryTable`, and `Read` are all treated as data leaves
and contribute to `data_deps` (including when reached through opaque
sub-expressions).  Other opaque node types (`RemoteTable`, `CachedNode`,
`FlightExpr`, `FlightUDXF`) are treated as structural carriers — they appear
in `structural` placeholdered by their inner sub-expression's structural
token.

`ExprScalarUDF.computed_kwargs_expr` leaves are reached by `walk_nodes`
(so an `InMemoryTable` used as training data inside `FittedPipeline.predict`
appears in `data_deps`), and `_normalize_computed_kwargs_expr` is
*data-free* — `InMemoryTable` is keyed by schema only, `Read` strips
`hash_path`/`read_path` from its key fields, and `CachedNode` is keyed by
schema + cache class.  The data identity for those leaves flows into outer
`data_deps`; the structural component reuses across data swaps.

## Alternatives considered

### Slot-based string substitution in SQL

Replace opaque nodes with positional placeholder names (`__slot_0__`,
`__slot_1__`) in the SQL string, then do string substitution when
recomputing tokens.

Rejected because:
- Requires maintaining a mapping between slot indices and SQL string positions.
- The hex table names embedded in SQL by `opaque_node_replacer` form a
  dependency chain (outer SQL references hashes of inner subtrees), making
  substitution error-prone.
- Unbinding the leaves is simpler and produces the same result without
  touching the SQL string at all.

### Use ibis's built-in `expr.unbind()`

Call `.unbind()` to strip backend bindings, which converts `DatabaseTable`
to `UnboundTable`.

Rejected because:
- `.unbind()` replaces *all* `DatabaseTable` subclasses, including
  `RemoteTable` — this discards the inner expression structure (filters, inner
  SQL), losing structural information that should be preserved.
- The targeted `type(n) is ir.DatabaseTable` replacement via `replace_nodes`
  unbinds only the data-holding leaves.

## Consequences

### Positive

- Substituting a table is O(new table size) instead of O(full pipeline).
- The structural token is a reusable, cacheable artifact — compute once per
  expression shape, reuse across any number of data variations.
- Removes the current redundancy where data-dependent values appear both in
  SQL table names and in the normalization tuple.
- Token recomputation from serialized metadata requires only `hashlib` — no
  xorq or dask import needed.

### Negative

- Cache break: token computation changes, so all existing cached results miss
  on first run.  No backward-compatibility shim (see Decision Drivers).
- `walk_nodes` traversal order determines `data_deps` indices.  Callers
  should look up slots by `index`, not by `name` (which may collide).

## References

- `dask_normalize_expr.py:713-740` — current `normalize_op`
- `dask_normalize_expr.py:646-703` — `opaque_node_replacer`
- `dask_normalize_expr.py:83-102` — `normalize_memory_databasetable`
- `graph_utils.py:73-93` — `walk_nodes`
- `graph_utils.py:96-138` — `replace_nodes`
- `graph_utils.py:32-55` — `gen_children_of`
- `scripts/2026-04-27-pierre-sync.py` — example expression used for validation

## Amendment — 2026-05-19: dasher migration (v2 → v3)

PR #1951 replaced `dask.base.tokenize` with `xorq-dasher`, changing the
hash algorithm and encoding underneath this ADR's decomposition. The
decision (structural + per-slot data hashes, cheap substitution, minimal-env
recomputation) is unchanged; only the mechanism moved.

### What changed

| | v2 (dask) | v3 (dasher) |
|---|---|---|
| Hash algorithm | MD5 (32-char hex) | xxh128 (32-char hex) |
| Encoding | `str(normalized_tuple)` | Binary `_encode` (type-tagged, struct-packed) |
| Recomputation dependency | `hashlib` only | `xxhash` + `struct` |
| Normalizer return | `(tuple_of_slot_hashes, structural_hash)` | `("ibis.Expr.v3", structural_hash, *slot_hashes)` |
| Metadata version | 2 | 3 |
| Slot metadata | `{index, name, hash}` | `{index, kind, name, hash}` |

### `_normalize_expr_xorq_impl` return shape

The Expr normalizer now hashes data leaves inline and returns an
all-strings tuple:

```python
structural_hash = hasher.tokenize("ibis.Expr.structural", sql, udfs)
slot_hashes = tuple(
    [hasher.tokenize(r) for r in reads]
    + [hasher.tokenize(dt) for dt in dts]
    + [hasher.tokenize(normalize_inmemorytable(m)) for m in mems]
)
return ("ibis.Expr.v3", structural_hash, *slot_hashes)
```

Because every element is a hex string (a primitive), `_normalize_recursive`
passes through without further recursion, making the normalized form
directly decomposable.

### `compute_expr_token`

Lives in `dasher/_recompute.py` with a vendored copy of `_encode`/`_write`
so it imports only `xxhash` and `struct`:

```python
def compute_expr_token(structural_hash, slot_hashes):
    inner = ("ibis.Expr.v3", structural_hash) + tuple(slot_hashes)
    return xxhash.xxh128(_encode((inner,))).hexdigest()
```

### Why `xxhash` is acceptable as a minimal-env dependency

The v2 formula needed only `hashlib` (stdlib). The v3 formula needs
`xxhash`, a single-file C extension with no transitive dependencies and
wheels for every platform. It is already a required dependency of
`xorq-dasher` and thus present in any environment that produced the
metadata in the first place.

### References

- `dasher/_opaque.py` — `_decompose_expr`, `_normalize_expr_xorq_impl`, `expr_metadata`
- `dasher/_recompute.py` — `compute_expr_token`, vendored `_encode`/`_write`
- `dasher/__init__.py` — public exports, `with_caches` wrapper
- `tests/test_dasher.py` — round-trip, structural stability, slot invalidation, minimal-env tests
