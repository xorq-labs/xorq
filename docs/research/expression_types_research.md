# Expression Types Research

## Sources

### ExprKind enum
- File: `python/xorq/ibis_yaml/enums.py:24-27`
- Three values: `Source`, `Expr`, `UnboundExpr`

### ExprMetadata
- File: `python/xorq/vendor/ibis/expr/types/core.py:681-732`
- Determines kind via match on `_unbound_node` and `_is_source`
- Source detection: root (unwrapped of Tags) is one of `DatabaseTable`, `Read`, `InMemoryTable`, `CachedNode`
- UnboundExpr detection: presence of `UnboundTable` anywhere in graph
- Expr: everything else (any transformations on bound data)
- Properties: `kind`, `schema_in` (only for UnboundExpr), `schema_out`, `to_dict()`

### LETSQLAccessor (.ls)
- File: `python/xorq/vendor/ibis/expr/types/core.py:735-947`
- Key properties: `kind`, `unwrapped`, `cached_nodes`, `backends`, `is_multiengine`, `is_cached`, `has_cached`, `cache`, `uncached`, `tokenized`, `tags`, `pipelines`
- Key methods: `get_key()`, `exists()`, `get_cache_path()`

### Xorq-specific Relation nodes
- File: `python/xorq/expr/relations.py`
- `Tag(Relation)` - metadata wrapper, stripped before hashing
- `HashingTag(Tag)` - metadata that contributes to hash
- `DatabaseTableView(DatabaseTable)` - base for special tables
- `CachedNode(DatabaseTableView)` - materialization point, has `parent` and `cache`
- `RemoteTable(DatabaseTableView)` - cross-backend transfer, has `remote_expr`
- `FlightExpr(DatabaseTableView)` - Flight-based unbound expr execution
- `FlightUDXF(DatabaseTableView)` - Flight-based UDXF execution
- `Read(DatabaseTable)` - deferred file I/O

### Ibis Relation operations
- File: `python/xorq/vendor/ibis/expr/operations/relations.py`
- Physical: `UnboundTable`, `DatabaseTable`, `InMemoryTable`
- Transform: `Project`, `Filter`, `Sort`, `Limit`, `Aggregate`, `DropColumns`
- Set: `Union`, `Intersection`, `Difference`
- Join: `JoinChain`
- Special: `View`, `SQLStringView`, `SQLQueryResult`, `DummyTable`, `FillNull`, `DropNull`, `Sample`, `Distinct`, `TableUnnest`

### Tests
- File: `python/xorq/expr/tests/test_relations.py`
- Covers: ExprKind detection for all three kinds, .ls accessor, .unwrapped behavior

### Key Table methods
- `filter()`, `select()`, `order_by()`, `aggregate()`, `join()`, `limit()` - standard transforms
- `cache()` - creates CachedNode
- `into_backend()` - creates RemoteTable
- `unbind()` - replaces DatabaseTable with UnboundTable
- `pipe()` - functional composition for UDXFs

## How expression kinds map to the graph

```
Source:       [InMemoryTable] or [DatabaseTable] or [Read] or [CachedNode]
                 (optionally wrapped in Tag/HashingTag)

Expr:         [Source] → [Filter] → [Project] → [Aggregate] → ...
                 (any chain of transforms on bound data)

UnboundExpr:  [UnboundTable] → [Filter] → [Project] → ...
                 (any expression containing at least one UnboundTable)
```

## How special nodes relate to ExprKind

- CachedNode is a **Source** (it's a materialization boundary)
- RemoteTable is a **Source** (it's a DatabaseTableView, detected as DatabaseTable)
- Read is a **Source** (explicitly in source_nodes tuple)
- Tag/HashingTag wrapping a Source → still **Source** (unwrapped before check)
- FlightExpr/FlightUDXF extend DatabaseTableView → treated as **Source**

## Practical flow

1. User creates source: `xo.memtable(...)`, `con.table(...)`, `xo.deferred_read_parquet(...)`
2. User transforms: `.filter()`, `.select()`, `.join()` → becomes Expr
3. User caches: `.cache(cache=...)` → CachedNode wraps result, becomes Source again
4. User moves backend: `.into_backend(con)` → RemoteTable, becomes Source
5. User applies UDXF: `.pipe(flight_udxf(...))` → FlightUDXF node
6. User unbinds: `.unbind()` → becomes UnboundExpr (template)
