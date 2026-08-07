# ADR-0016: Rebuildable protocol for recovered builder objects

- **Status:** Accepted
- **Date:** 2026-06-22
- **Deciders:** dlovell

## Context

ADR-0005 introduced the `TagHandler` registry: `from_tag_node(tag_node) -> object`
recovers a live domain object (e.g. `FittedPipeline`, `ExprComposer`) from an
expression's tags. That recovery half is fully general — a handler may return any
object, and third parties register any tag via the `xorq.from_tag_node` entry-point
group.

The *consumption* half was not general. `catalog replay --rebuild` needs every
recovered builder to re-emit its expression under a target catalog, but the recovered
objects expose differently-shaped rebuild APIs:

- `FittedPipeline.reemit(tag_node, rebuild_subexpr)` — recurses through its own
  training/predict subtrees via the `rebuild_subexpr` closure.
- `ExprComposer.with_inputs_translated(remap, to_catalog).expr` — translates a
  self-contained recipe's catalog inputs and re-emits.

`get_rebuild_dispatch` reconciled these by duck-typing: it sniffed for a `reemit`
method, then for `with_inputs_translated` + an `expr` attribute, normalizing both to a
single `dispatch(rebuild_subexpr, remap, to_catalog) -> Expr` closure. Anything that
matched neither shape fell silently through to `None` — so a new third-party builder
was *recoverable* but not *rebuildable*, with no named contract to implement.

Reasonable people could disagree on whether to unify the two builders' construction
APIs wholesale. A prototype against both classes showed they should not: their
"create a new expr" verbs are genuinely different (`FittedPipeline` takes new input
data plus an output-mode selector; `ExprComposer` is parameter-free recipe replay).
Only the *rebuild* operation is common.

## Decision drivers

- Replace duck-typing with an explicit, discoverable contract.
- Give third-party builders a single method to implement for `--rebuild` support.
- Do not force-unify the divergent "build a new expr" APIs (predict/transform vs.
  recipe replay) — that abstraction leaks.
- No change to ADR-0005's recovery registry or `from_tag_node` semantics.

## Decision

### `Rebuildable` protocol

Add a `runtime_checkable` `Rebuildable` protocol in
`python/xorq/expr/builders/__init__.py`:

```python
@runtime_checkable
class Rebuildable(Protocol):
    def rebuild(
        self, tag_node, rebuild_subexpr, remap, to_catalog
    ) -> Expr: ...
```

The four arguments are the full context available at the dispatch site. Each
implementation uses the ones relevant to its rebuild model and ignores the rest:

- input-driven builders (`FittedPipeline`) use `tag_node` + `rebuild_subexpr` and
  ignore `remap`/`to_catalog`;
- recipe builders (`ExprComposer`) use `remap` + `to_catalog` and ignore
  `tag_node`/`rebuild_subexpr`.

### `get_rebuild_dispatch` collapses to one isinstance check

The two domain-object duck-typing branches in `get_rebuild_dispatch` are replaced by a
single `isinstance(builder, Rebuildable)` check
(`python/xorq/expr/builders/__init__.py`). The handler-level `reemit` path (path 1,
for builders whose `from_tag_node` does not carry the full recipe) is unchanged.

### The two builders implement `rebuild`

- `FittedPipeline.reemit` is renamed to `rebuild` with the widened signature; the body
  is unchanged (`python/xorq/expr/ml/pipeline_lib.py`).
- `ExprComposer` gains a `rebuild` method delegating to
  `with_inputs_translated(remap, to_catalog).expr`
  (`python/xorq/catalog/composer.py`). `with_inputs_translated` stays — the
  `ExprKind.Composed` path in `replay.py` calls it directly.

## Alternatives considered

### Unify the "build a new expr" API too (`build(**params)`)

Add a `build(**params) -> Expr` member alongside `rebuild`.

Rejected because the two builders have incompatible construction models:
`ExprComposer` is parameter-free (vary the expr by constructing a different composer),
while `FittedPipeline.build` requires a new-data expr plus a method selector
(predict / transform / predict_proba / decision_function). A shared `**params`
signature carries no real contract; it only type-checks by being a catch-all. "Create
a new expr" stays domain-specific.

### Keep duck-typing, just document it

Leave `get_rebuild_dispatch` sniffing for `reemit` / `with_inputs_translated`.

Rejected because it leaves third-party builders with no discoverable contract and
silently downgrades unmatched builders to non-rebuildable.

### Put metadata introspection (`params()`) on the protocol

Rejected as redundant with the handler's existing `extract_metadata(tag_node)`
(ADR-0005), which already produces sidecar metadata from the tag.

## Consequences

### Positive

- `get_rebuild_dispatch` drops from a three-way duck-typing sniff to one
  `isinstance` check plus the handler path.
- Third-party builders get a single, discoverable method (`rebuild`) to implement for
  `catalog replay --rebuild` support.
- ADR-0005's recovery registry is untouched; this is purely additive on the
  consumption side.

### Negative

- `FittedPipeline.rebuild` accepts `remap`/`to_catalog` it ignores, and
  `ExprComposer.rebuild` accepts `tag_node`/`rebuild_subexpr` it ignores. The unified
  signature trades a little per-implementation noise for one dispatch contract.
- Renaming `FittedPipeline.reemit` to `rebuild` is a breaking change for any external
  caller that invoked `reemit` directly (in-tree there were none beyond the dispatch).

## References

- ADR-0005: ExprBuilder — registry-driven domain object recovery from tagged expressions
- `python/xorq/expr/builders/__init__.py` — `Rebuildable`, `get_rebuild_dispatch`
- `python/xorq/catalog/replay.py` — rebuild driver
