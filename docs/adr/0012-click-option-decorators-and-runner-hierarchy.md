# ADR-0012: Shared Click option decorators and runner class hierarchy for CLI parity

- **Status:** Accepted
- **Date:** 2026-05-18
- **Deciders:** Dan Lovell

## Context

The xorq CLI has three command surfaces — `xorq`, `xorq uv`, and `xorq catalog` — that share conceptual options (output format, cache directory, limit, params, unbind targets, serve host/port). Before this work, each surface copy-pasted its own `@click.option(...)` blocks, leading to:

- Drift in flag names, help text, and defaults across surfaces (e.g. `catalog run` lacked `--cache-dir`; `uv run` lacked `--limit` and `--params`).
- No `xorq uv run-cached` or `xorq uv run-unbound` commands, leaving gaps in the `uv` surface relative to the top-level CLI.
- Growing maintenance burden: adding a shared option required copy-editing N commands.

The `xorq uv` commands delegate to the top-level CLI via `uv tool run xorq <subcommand>` inside an isolated environment. This subprocess-forwarding architecture means the `uv` layer must construct argument lists, not call Python functions — which constrains how options are validated and how file handles are passed.

## Decision drivers

- Identical flags for the same concept across all three CLI surfaces.
- New `xorq uv run-cached` and `xorq uv run-unbound` commands with minimal new code.
- No circular imports between `cli.py`, `catalog/cli.py`, and the shared option module.
- `packager.py` stays decoupled from Click.

## Decision

### Shared option decorators in `cli_options.py`

Extract 14 reusable decorators (`output_options`, `limit_option`, `params_options`, `cache_dir_option`, `cache_strategy_options`, `unbind_options`, `serve_options`, `fuse_option`, `rename_params_option`, `code_option`, `sync_option`, `env_options`, `gcs_option`, `json_option`) into `python/xorq/cli_options.py`. A companion `python/xorq/cli_constants.py` holds `OutputFormats` and default values, breaking the circular import between `cli_options` and the CLI modules that define commands (`python/xorq/cli.py`, `python/xorq/catalog/cli.py`).

Each decorator is a thin wrapper around `click.option` calls. Decorators apply bottom-up so `--help` reads naturally.

### `run-unbound` special case for `@output_options`

`run-unbound` has different `--output-path` semantics: default is stdout for arrow format, `/dev/null` otherwise. Rather than parameterizing `@output_options` with conditional defaults (which would push runtime logic into a decorator), `output_options` accepts an optional `output_path_help` kwarg. Both the top-level `run-unbound` and the `uv run-unbound` command pass custom help text via `@output_options(output_path_help=...)` while reusing the same decorator. The actual output-path arbitration (stdout for arrow, `/dev/null` otherwise) lives in `run_unbound_command`, not in the decorator.

### `_BasePackagedRunner` with template method

The plan considered and rejected inheritance for the runner classes due to `@frozen` subclassing constraints in attrs. During implementation, a cleaner design emerged: a `_BasePackagedRunner` frozen base class in `python/xorq/ibis_yaml/packager.py` that owns the shared fields (`build_path`, `cache_dir`, `output_path`, `output_format`, `limit`, `python_version`, `_bundle`), `__attrs_post_init__` validation via `_validate_build_path`, and a template-method `_run` that calls `_subcommand()` and `_extra_args()` — both overridden by each concrete class.

This works because the `@frozen` constraint that blocked earlier inheritance designs only applies when the subclass tries to add `__attrs_post_init__` or `cached_property` overrides that mutate state. Here the base owns the post-init and the `cached_property`; subclasses only override plain methods (`_subcommand`, `_extra_args`) and declare additional fields. The `object.__setattr__` dance happens once in the base.

The three concrete classes are:

| Class | Subcommand | Extra fields |
|---|---|---|
| `PackagedRunner` | `run` | `raw_params` |
| `PackagedCachedRunner` | `run-cached` | `cache_type`, `ttl`, `raw_params` |
| `PackagedUnboundRunner` | `run-unbound` | `to_unbind_hash`, `to_unbind_tag`, `typ`, `batch_size`, `instream` |

### `validate_params_early` raises `ValueError`, not `click.BadParameter`

Param validation lives in `packager.py` as `validate_params_early(build_path, raw_params)`. It reads `expr_metadata.json` from the build directory and checks supplied param names against declared ones. It raises `ValueError` to keep the packager module decoupled from Click. The CLI layer wraps calls in `try/except ValueError` → `click.BadParameter`.

This boundary means future validation added to the packager should also raise domain exceptions (`ValueError`, `FileNotFoundError`), not Click exceptions. Click coupling belongs exclusively in `cli.py` and `catalog/cli.py`.

### `--params` excluded from `run-unbound`

`run-unbound` intentionally does not accept `--params`. The unbound node is replaced with piped data at execution time; expression parameters are baked in at build time via `into_expr`. Parameterization and unbinding operate at different lifecycle stages and combining them in a single command would require the `uv` subprocess to both parameterize and pipe — an interaction that has no existing test coverage or defined semantics. Users who need both must parameterize at build time.

### `click.Path` for `--instream` in `uv run-unbound`

The top-level `run_unbound_command` in `cli.py` uses `click.File("rb")` for `--instream`, receiving a file-like object. The `uv` variant cannot forward a file object across a subprocess boundary, so it uses `click.Path(exists=True)` instead, receiving a path string that is appended to the subprocess args. When `--instream` is omitted, `uv_tool_run` is called with `capture_output=False`, so the subprocess inherits the parent's stdin — the inner `xorq run-unbound` defaults `--instream` to `"-"` (stdin) and reads from the inherited fd.

## Alternatives considered

### Standalone runner classes with full duplication

The initial plan accepted ~15 lines of duplicated fields and `__attrs_post_init__` per runner class, reasoning that inheritance in `@frozen` attrs classes is awkward. Implementation revealed that a base class with template methods avoids the duplication cleanly — the `@frozen` subclassing constraint only blocks subclasses that override `__attrs_post_init__` or add `cached_property`, which the template-method design avoids.

Rejected because:
- Three copies of `__attrs_post_init__` and `_validate_build_path` means a validation bug fix must be applied in three places. The template-method design keeps both in the base class, so fixes land once.

### Composite `@catalog_compose_options` decorator

Early plan versions bundled `--fuse`, `--rename-params`, and `-c/--code` into a single `@catalog_compose_options` decorator. This was split into three individual decorators (`fuse_option`, `rename_params_option`, `code_option`) because the three options don't always co-occur — some catalog commands use `--fuse` without `--code` — and a composite decorator would force commands to accept unused parameters.

Rejected because:
- `catalog compose` uses all three, but `catalog run` uses only `--fuse`. A composite decorator would force `run` to accept `--code` and `--rename-params` silently, polluting `--help` and creating dead parameter paths that could confuse users or mask bugs.

### Extending `PackagedRunner` with a `command` field

The initial plan proposed adding an optional `command` field (default `"run"`, alternative `"run-cached"`) to `PackagedRunner` instead of creating new classes. This would conflate the runner's identity with a runtime parameter, requiring conditionals in `_run` to decide which extra args to append.

Rejected because:
- `PackagedUnboundRunner` needs `instream` and `to_unbind_hash`; `PackagedCachedRunner` needs `cache_type` and `ttl`; plain `PackagedRunner` needs neither. A single class with a `command` field would carry all these fields on every instance — attrs can't enforce that `cache_type` is set when `command="run-cached"` and absent otherwise, so validation shifts from the type system to runtime conditionals in `_run`.

## Consequences

### Positive

- Adding a shared option to all surfaces is a one-line import change per command, not a copy-paste-edit across N commands.
- `xorq uv run-cached` and `xorq uv run-unbound` exist, closing the parity gap with the top-level CLI.
- The `_BasePackagedRunner` template method means adding a fourth runner variant (e.g. `PackagedServeRunner`) requires only `_subcommand`, `_extra_args`, and variant-specific fields.
- `validate_params_early` catches invalid `--params` before subprocess launch, giving clear errors instead of noisy `uv tool run` failures.

### Negative

- `cli_options.py` is a new module that every CLI contributor must know exists. If someone adds a `@click.option` directly to a command instead of using or extending a shared decorator, drift resumes.
- `_BasePackagedRunner._run` is a `@functools.cached_property` that performs a side-effectful subprocess call. This is an existing antipattern copied from the original `PackagedRunner`; the base class now entrenches it across all three variants. Accepted as debt: the pattern works correctly today and changing it would widen the scope of this refactor without fixing a bug. A plain method with a `_result` sentinel would be cleaner but is not blocking.
- The `output_options(output_path_help=...)` parameterization is ad-hoc. If a second command needs different defaults (not just help text), the decorator will need further generalization or that command will need to opt out entirely.
- `cli.py`, `cli_constants.py`, and `cli_options.py` are three top-level files sharing a `cli_` prefix — a package expressed as a naming convention. Accepted as debt: converting to a `cli/` package would touch every import site and is better done as a standalone refactor than buried in this PR. The prefix convention doesn't scale, but three files is manageable for now.

## References

- PR [#1962](https://github.com/xorq-labs/xorq/pull/1962) — shared decorators, parity gaps, and runner infrastructure (Phases 1–2)
- PR [#1966](https://github.com/xorq-labs/xorq/pull/1966) — `xorq uv run-cached` and `xorq uv run-unbound` (Phase 3)
- [ADR-0008](0008-wheel-based-packaging-pipeline.md) — wheel pipeline that `PackagedRunner` and its siblings execute within
