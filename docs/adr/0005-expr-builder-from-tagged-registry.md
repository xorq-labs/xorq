# ADR-0005: ExprBuilder — registry-driven domain object recovery from tagged expressions

- **Status:** Accepted
- **Date:** 2026-04-09
- **Context area:** `python/xorq/expr/builders/`, `python/xorq/vendor/ibis/expr/types/core.py`, `python/xorq/expr/ml/pipeline_lib.py`

## Context

xorq expressions can carry domain-specific metadata in `Tag` nodes — BSL semantic models attach query structure, ML pipelines attach step definitions and fitted parameters. Before this work, the metadata was write-only: it was stamped onto expressions during construction but there was no systematic way to recover the original domain object (e.g., a `SemanticModel` or `FittedPipeline`) from a cataloged expression.

This matters for two workflows:

1. **Catalog round-trip.** A user catalogs a BSL query expression. Later, they load it and want to issue a *different* query on the same semantic model. Without recovery, they must reconstruct the model from scratch.
2. **ML pipeline rebinding.** A user catalogs a prediction expression. Later, they load it and want to run the same fitted pipeline on new production data. Without recovery, they must re-fit the pipeline manually.

## Decision

### TagHandler registry

Introduce a `TagHandler` registry in `python/xorq/expr/builders/__init__.py`. Each handler is an attrs `@frozen` class with two optional callbacks:

- `extract_metadata(tag_node) -> dict` — produces sidecar metadata for the catalog (dimensions, measures, pipeline steps, etc.)
- `from_tagged(tag_node) -> object` — recovers the live domain object from the tag

Handlers are registered by tag name (the string in `tag_node.metadata["tag"]`). Built-in handlers for BSL and ML pipeline tags are registered at module init time. Third-party packages register via the `"xorq.from_tagged"` entry-point group.

### ExprKind.ExprBuilder

A new `ExprKind.ExprBuilder` variant identifies expressions whose outermost tag matches a registered handler.

`ExprKind` describes the **outermost structural layer** of an expression, not a whole-graph property. The outermost op node (walking through any Tag/HashingTag chain) determines the kind:

- If a catalog composition tag (`HashingTag` with `CatalogTag` metadata) is outermost → `Composed`
- If a builder tag (registered in the handler registry) is outermost → `ExprBuilder`
- Unrecognized tags are unwrapped (they're decorative and don't affect kind)
- If the unwrapped root is a source node → `Source`
- Otherwise → `Expr`

The one exception is `UnboundExpr`, which requires a whole-graph walk because `UnboundTable` is always a leaf node, never the outermost node. It's checked first as a special case — it's a constraint ("needs binding"), not a structural kind.

There are no priority conflicts: the outermost tag is structurally unambiguous. A composed expression with builder tags inside gets `Composed` because the catalog tag is outermost.

### ExprTraits — whole-graph boolean properties

For consumers that need to know "does X exist anywhere in this graph?" (not just the outermost layer), `ExprTraits` provides cached boolean flags:

```python
@frozen
class ExprTraits:
    has_unbound: bool     # UnboundTable exists somewhere
    has_composition: bool  # catalog HashingTag nodes exist
    has_builders: bool     # builder tags exist
    is_source: bool        # root (after unwrapping tags) is a source node
```

Exposed as `expr.ls.expr_traits` (cached property). The graph walk happens once. `kind` and `expr_traits` are complementary:

- `kind` — "what is this expression?" (outermost layer, cheap)
- `expr_traits` — "what does this expression contain?" (whole-graph, cached)

### ExprMetadata.builders

`ExprMetadata` gains a `builders` tuple field. During `from_expr`, `_extract_builders` walks the expression's tags, calls `extract_builder_metadata` for each registered tag name, and collects the results. This metadata is persisted in the sidecar YAML alongside `kind`, `schema_out`, etc.

### ML pipeline recovery via `FittedPipeline.from_tag_node`

ML pipeline recovery is split into two classmethods:

- `FittedPipeline.from_tag_node(tag_node)` — the core method, used by the builder registry. Receives the specific pipeline tag node (which carries `ALL_STEPS` metadata), reads features/target from it, finds the training source by graph structure (innermost step tag's parent), and replays `pipeline.fit()`. Fit is a deferred operation — it builds the expression graph without executing sklearn training.
- `FittedPipeline.from_expr(expr)` — convenience method that mirrors `expr.ls.pipeline`. Takes the outermost pipeline tag and delegates to `from_tag_node`.

The `ls.builder` property on `LETSQLAccessor` dispatches through the registry, calling `_resolve_builder_from_tag` which walks tags outermost-first and returns the first handler match.

### Training data provenance

`Pipeline.fit()` computes a `training_hash` (via `make_name`) before fitting and stores it on `FittedPipeline`. This hash is included in the `ALL_STEPS` tag metadata as inert provenance — it is not used at recovery time but is available for optional validation ("is this the same training data I fitted on?").

Recovery finds the training source by graph structure: the innermost ML-related tag (FittedStep or FittedPipeline) in the expression graph has a parent chain that reaches the training data. This is structurally guaranteed by how `Pipeline.fit()` builds the expression — it applies steps sequentially on the training expression, so the training data is always the deepest source in the pipeline subgraph.

### Builtin key protection

Builtin tag keys (those registered by `_register_builtins`) are protected: `register_tag_handler` and `_discover_from_tagged` both reject attempts to overwrite them. The protected set is derived automatically by snapshotting the registry keys after `_register_builtins` runs — there is no separate enum or manifest to maintain.

## Alternatives considered

### Separate TRAINING tag for ML pipeline recovery

An early iteration added a `FittedPipelineTagKey.TRAINING` tag stamped onto the raw training data in `Pipeline.fit()`. `from_expr` walked the graph to find this tag, read features/target from it, and used `training_tag.parent.to_expr()` as the training source.

Rejected because the TRAINING tag was redundant:
- Features and target are already stored per-step in `ALL_STEPS` metadata on every predict/transform tag.
- The training source's position in the graph is structurally determined — it is always reachable as the innermost step tag's parent.
- The extra tag changed `walk_nodes` traversal order, requiring test relaxations that masked whether ordering changes were real regressions.

### BuiltinTagKey enum for protected keys

An early iteration used a `StrEnum` subclass to declare protected builtin tag keys. Rejected because:
- It duplicated values already in `FittedPipelineTagKey`, creating two sources of truth that could drift.
- No enum member was ever referenced — the enum was immediately flattened into a `frozenset`.
- It required a `strenum` compatibility shim for Python < 3.11.

Deriving the protected set from what `_register_builtins` actually registers is simpler and cannot drift.

### Priority-based whole-graph kind detection

An early iteration walked the entire expression graph for all kind signals (UnboundTable, HashingTag, builder tags, source nodes), then used a priority order to pick one kind. Rejected because:
- Inputs came from inconsistent scopes (3 whole-graph walks + 1 root-only check).
- Priority order was policy, not structure — different consumers wanted different priorities.
- Adding new kinds required changing every case arm in a match statement.
- Kind depended on which handlers were registered (runtime state), not expression structure.

Making kind outermost-only eliminates the priority conflict (the outermost tag is structurally unambiguous) and reduces kind detection from 4 graph walks to a tag-chain traversal. `ExprTraits` provides whole-graph answers for consumers that need them.

### Hash-based training source lookup

An alternative to structural graph walking: store the training data's content hash in `ALL_STEPS` and find the matching node at recovery time by tokenizing every node during graph walk. Rejected for recovery because:
- O(n * m) tokenization cost vs. O(n) structural walk.
- Dask tokenization stability across versions is not guaranteed.
- The structural position is already reliable.

The hash is kept as inert provenance metadata for optional validation use cases.

## Consequences

### Positive

- **Catalog entries become actionable.** `entry.expr.ls.builder` recovers the domain object, enabling new queries (BSL) or predictions on new data (ML) without reconstruction.
- **Third-party extensibility.** Any package can register a `TagHandler` via entry points — no changes to xorq core required.
- **Sidecar richness.** Builder metadata (dimensions, measures, pipeline steps) is available from the sidecar without fetching the expression archive.
- **Clean kind/traits split.** `kind` is a cheap outermost-only check; `expr_traits` provides cached whole-graph booleans. No priority conflicts, no lossy reduction.

### Negative

- **Recovery replays fit.** ML pipeline recovery calls `pipeline.fit()` on the training source. Fit is deferred (no sklearn execution), but the resulting `FittedPipeline` is a new object, not the original. Non-deterministic estimators with unfixed `random_state` would produce different fitted models if the cache is invalidated.
- **First-match-wins dispatch.** `_resolve_builder_from_tag` returns the first handler match during outermost-first graph walk. For expressions with multiple tag types (e.g., BSL wrapped in an ML pipeline), the outermost tag wins. This is undocumented beyond the docstring and could surprise users.
- **Module-global registry.** The handler registry is a module-level dict. Tests must save/restore state via `_reset_registry()`. An instance-based registry would be cleaner but is deferred unless test flakiness warrants it.
