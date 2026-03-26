# xorq

## Package management
- Always use `uv` (not `pip`). Use `uv pip install <pkg>` for ad-hoc installs.

## Running tests
- Always: `python -m pytest <path> -x -q 2>&1 | tail -30`
- Skip slow tests during development: add `-m "not slow"`
- Target specific test files, not the full suite
- Check for collection errors after porting: `python -m pytest --co -q`
- Do NOT defer imports inside test functions or fixtures — all imports go at module level

## Linting
- Run `pre-commit run --files <file>` to fix lint/format issues before committing

## Commit conventions
- Format: `type(scope): subject` (feat, fix, perf, chore, ref)
- Always include body explaining *why*
- Always include `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`

## Catalog: sidecar over zip
- `CatalogEntry` metadata (kind, schema, backends, composed_from) lives in a git-tracked sidecar YAML — always prefer `entry.metadata`, `entry.kind`, `entry.columns`, etc. over `entry.expr`
- Only access `entry.expr` / `entry.lazy_expr` when you need the deserialized expression (execution, RemoteTable, graph walking)
- `entry.expr` raises `ContentNotAvailableError` when annex content is not local; sidecar properties always work
- To extend the sidecar: update `CatalogAddition.metadata`, expose on `CatalogEntry`, optionally extend `ExprMetadata` — see ADR-0003

## Known gotchas
- DuckDB connections hang when reused — avoid reusing them in scripts/tests
- CLI startup latency matters — keep heavy imports deferred (inside command functions)

## Project layout
- Source: `python/xorq/`
- Tests co-located: `python/xorq/<module>/tests/`
- Related repo: `/home/dan/repos/github/xorq-catalog/`
- Sibling project is being ported into `python/xorq/catalog/`

## Benchmarking / profiling
- Throwaway scripts go to `/tmp/bench_*.py`
- Existing benchmark suite: `python/xorq/ibis_yaml/tests/test_benchmark.py`
