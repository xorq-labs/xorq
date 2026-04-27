# ADR-0009: Split data-dependent tokens out of expression normalization

- **Status:** Proposed
- **Date:** 2026-04-27
- **Deciders:** Dan, Pierre

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

Replace `normalize_op` with a new implementation that unbinds the expression
of its leaf data tables, normalizes the unbound (structural) expression
separately, and returns a `(data_deps, structural)` tuple.  The current
`normalize_op` and `normalize_expr` are replaced, not supplemented — the old
code path is removed.

### `normalize_expr` return shape

The registered `dask.base.normalize_token` handler for `ibis.expr.types.Expr`
changes its return value from a flat normalized tuple to:

```python
(data_deps, structural)
```

where `data_deps` is a tuple of per-leaf-table normalized tokens and
`structural` is the normalized token for the query shape.
`dask.base.tokenize(expr)` then hashes this pair.

### `normalize_op_split`

A new function that exposes the split for callers that need to inspect or
substitute individual data deps:

```python
def normalize_op_split(expr):
    """
    Returns (leaf_dts, data_deps, structural).

    leaf_dts    — the original DatabaseTable ops, in walk order
    data_deps   — tuple of their normalized tokens
    structural  — normalized token for the query shape (data-free)
    """
```

`normalize_expr` delegates to this and returns `(data_deps, structural)`.

### Mechanism

The key insight is that replacing leaf `DatabaseTable` nodes with
`UnboundTable` (preserving name and schema) makes the entire normalization
pipeline data-free — `normalize_memory_databasetable` is only reachable
through the `normalize_databasetable` dispatch, which is registered for
`ir.DatabaseTable`, not `ir.UnboundTable`.

```python
from xorq.common.utils.graph_utils import walk_nodes, replace_nodes

def normalize_op_split(expr):
    # 1. Find leaf data nodes via the xorq-aware traversal
    leaf_dts = [
        n for n in walk_nodes((ir.DatabaseTable,), expr)
        if type(n) is ir.DatabaseTable
    ]

    # 2. Compute data-dependent normalizations
    data_deps = tuple(dask.base.normalize_token(dt) for dt in leaf_dts)

    # 3. Replace leaves with UnboundTable (preserves name + schema)
    def unbind_leaf_dts(node, kwargs):
        if type(node) is ir.DatabaseTable:
            return ir.UnboundTable(name=node.name, schema=node.schema)
        return node.__recreate__(kwargs) if kwargs else node

    replaced_op = replace_nodes(unbind_leaf_dts, expr)

    # 4. normalize_op on the replaced tree is purely structural
    structural = normalize_op(replaced_op)

    return leaf_dts, data_deps, structural
```

### Why `replace_nodes`, not `op.replace`

ibis's built-in `op.replace` does not traverse into xorq-specific opaque
fields (`remote_expr`, `parent`, `input_expr`, `computed_kwargs_expr`).
`replace_nodes` (`graph_utils.py:96`) uses `gen_children_of` which handles
all of these — so a `DatabaseTable` inside a `RemoteTable.remote_expr` is
reached and replaced.

### Why `type(n) is ir.DatabaseTable`, not `isinstance`

RemoteTable, CachedNode, Read, FlightExpr, etc. are all `DatabaseTable`
subclasses, but they carry structural information (filters in `remote_expr`,
cache metadata) that must stay in the expression tree.  The identity check
ensures we only unbind actual data-holding leaves.

### Cheap substitution (with xorq)

```python
leaf_dts, data_deps, structural = normalize_op_split(expr)

# Substitute batting with a new table:
batting_idx = next(i for i, dt in enumerate(leaf_dts) if dt.name == "batting")
new_data_deps = list(data_deps)
new_data_deps[batting_idx] = dask.base.normalize_token(new_batting_dt)
new_token = dask.base.tokenize(tuple(new_data_deps), structural)
```

**Cost**: one `normalize_memory_databasetable` call (serialize + hash the new
table's Arrow batches).

**Skipped**: `register`, `into_backend`, expression compilation, `op.replace`,
SQL generation, normalization of all other nodes.

### Serializable metadata for token computation without xorq

The split produces components that reduce to hex strings.  These can be
serialized as a lightweight metadata artifact and used to compute expression
tokens without importing xorq — only `hashlib` is needed.

**Metadata schema:**

```json
{
  "version": 1,
  "structural_hash": "38317617c8a70d3a...",
  "slots": [
    {"name": "awards_players", "hash": "a5565158c996e82e..."},
    {"name": "batting",        "hash": "94ac8427b43abdf2..."}
  ]
}
```

`structural_hash` is `dask.base.tokenize(structural)`.  Each slot `hash` is
`dask.base.tokenize(data_dep)` for the corresponding leaf table.  Both are
32-character MD5 hex strings.

**Producing the metadata (requires xorq):**

```python
leaf_dts, data_deps, structural = normalize_op_split(expr)
metadata = {
    "version": 1,
    "structural_hash": dask.base.tokenize(structural),
    "slots": [
        {"name": dt.name, "hash": dask.base.tokenize(dep)}
        for dt, dep in zip(leaf_dts, data_deps)
    ],
}
```

**Computing a token from metadata (no xorq, no dask):**

```python
import hashlib

def compute_expr_token(data_dep_hashes, structural_hash):
    preimage = str((tuple(data_dep_hashes), structural_hash))
    return hashlib.md5(preimage.encode(), usedforsecurity=False).hexdigest()

slot_hashes = [s["hash"] for s in metadata["slots"]]
slot_hashes[batting_idx] = known_new_batting_hash
token = compute_expr_token(slot_hashes, metadata["structural_hash"])
```

This works because `dask.base.tokenize(*args)` is
`md5(str(normalize(*args)))`, and when the args are already hex strings
(which are identity-dispatched by dask's normalizer), the final hash is
fully determined by `str((tuple(hex_strings), hex_string))`.

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
  should look up slots by table name, not by hardcoded index.

### Scope

This addresses `DatabaseTable` leaves (in-memory tables registered via
`con.register`).  `Read` nodes (file-based) and other opaque node types
remain in the structural part via `opaque_node_replacer`.  The same pattern
can extend to those node types later if needed.

## References

- `dask_normalize_expr.py:713-740` — current `normalize_op`
- `dask_normalize_expr.py:646-703` — `opaque_node_replacer`
- `dask_normalize_expr.py:83-102` — `normalize_memory_databasetable`
- `graph_utils.py:73-93` — `walk_nodes`
- `graph_utils.py:96-138` — `replace_nodes`
- `graph_utils.py:32-55` — `gen_children_of`
- `scripts/2026-04-27-pierre-sync.py` — example expression used for validation
