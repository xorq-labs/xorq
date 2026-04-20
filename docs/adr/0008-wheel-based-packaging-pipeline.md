# ADR-0008: Wheel-based uv packaging and execution pipeline

- **Status:** Accepted, supersedes [ADR-0004](0004-uv-as-sole-packaging-and-execution-runtime.md)
- **Date:** 2026-04-20
- **Context area:** `python/xorq/ibis_yaml/packager.py`

> **Supersedes [ADR-0004](0004-uv-as-sole-packaging-and-execution-runtime.md).** ADR-0004 chose `uv build --sdist` for packaging; this ADR replaces that with `uv build --wheel` and restructures the archive contract around a wheel + `requirements.txt` sidecar pair. The rest of the uv-as-sole-runtime choices from ADR-0004 (single tool for lock, export, isolated run, Python version) carry over and are restated here for completeness.

## Context

The wheel pipeline (`WheelPackager → PackagedBuilder → PackagedRunner`) needs to build wheels, resolve and lock dependencies, export pinned requirements, and execute xorq commands in isolated environments. Each of these steps has a traditional Python tooling equivalent:

| Step | Traditional tools | uv equivalent |
|------|------------------|---------------|
| Build wheel | `python -m build --wheel` (pip-based build isolation) | `uv build --wheel` |
| Lock dependencies | `pip-compile` (pip-tools) or `pip freeze` | `uv lock` |
| Export pinned requirements | `pip-compile --output-file` | `uv export --locked` |
| Isolated execution | `python -m venv` + `pip install` + direct invocation | `uv tool run` |
| Python version selection | `pyenv` / manual management | `--python` flag |

Using multiple tools creates integration seams: pip's resolver may disagree with pip-compile's, a venv created by one tool may be populated differently by another, and Python version management is a separate concern requiring its own toolchain. Each seam is a potential source of non-reproducibility.

## Decision

Use `uv` as the single tool for every packaging and execution step in the wheel pipeline. No step shells out to pip, pip-tools, venv, pyenv, or `python -m build`.

### Build: `uv build --wheel`

`WheelPackager._build_wheel` invokes `uv build --wheel --python <version> --out-dir <tmpdir> <project>`. This replaces `python -m build --wheel`, which delegates to pip for build dependency installation. uv creates the build isolation environment using hardlinks/reflinks and its own resolver, avoiding pip's bootstrap cost and resolver inconsistencies.

The output is a standard PEP 427 wheel — the same backend (hatchling) runs in both cases. The difference is entirely in the frontend: how build dependencies are fetched and how the isolation environment is managed.

#### Why wheel instead of sdist

An earlier version of this pipeline built sdists (`uv build --sdist`), then passed them to `uv tool run --with <sdist.zip>`. This required uv to internally build a wheel from the sdist before installation, adding ~0.8s on cold runs. Additionally, the sdist was produced as a `.tar.gz` that had to be converted to `.zip` for downstream consumption. Building a wheel directly eliminates both conversions, saving ~1.7s (45%) on cold builds. On cached runs the difference vanishes since uv caches the built wheel.

### Lock and export: `uv lock` + `uv export`

`WheelPackager.__attrs_post_init__` validates that either `uv.lock` or `requirements.txt` exists in the project directory, raising `FileNotFoundError` if neither is found. Lock creation is a precondition — the user must run `uv lock` before invoking the packager. `WheelPackager._write_requirements_path` calls `uv_export_requirements`, which runs `uv export --locked --no-dev --no-emit-project --no-header --no-annotate` to produce a pinned `requirements.txt`.

This replaces the `pip-compile` / `pip freeze` workflow. The `--locked` flag asserts that `uv.lock` is up-to-date with `pyproject.toml`, erroring if not — unlike `--frozen`, which would skip this validation. The lock file is the single source of truth for resolved versions; the export is a deterministic projection of it. Because both lock and export use uv's resolver, there is no resolver mismatch between the lock step and the install step.

#### Sync-check is byte-exact

When both `uv.lock` and `requirements.txt` are present in the project, `WheelPackager._write_requirements_path` re-runs `uv export` and compares its output byte-for-byte against the existing `requirements.txt`. Any divergence — including whitespace, ordering, or format changes — raises `RuntimeError`. **This is intentional** and it *will* trip:

- Whenever `uv.lock` changes without re-exporting `requirements.txt`.
- Whenever the running `uv` is a different version than the one that last produced `requirements.txt`, if that version changed `uv export`'s output format.

The strict check prevents silently running with stale requirements. The resolution is mechanical: delete the in-tree `requirements.txt` (the packager will regenerate it) or re-run `uv export` manually. Downstream tooling that uses templates or vendored projects should either re-export before building or omit `requirements.txt` entirely and let the packager regenerate it from `uv.lock`.

#### `uv.lock` determines `requirements.txt`

`requirements.txt` is not an independent artifact — it is derived from `uv.lock` via `uv export` and must match exactly. The wheel and `requirements.txt` are stored as sidecar files: both are copied into the build directory inline within `PackagedBuilder._build` (after the `xorq build` invocation completes) and validated by `PackagedRunner` at init time. This ensures the build directory is self-contained and reproducible.

#### Hash-pinned requirements

`uv export` emits hashes by default (the `--no-hashes` flag is not passed). The exported `requirements.txt` contains lines like:

```
numpy==1.26.4 --hash=sha256:abcdef...
```

This means pip or uv will verify the integrity of every downloaded package against the hash recorded at lock time. Hash-pinning provides supply-chain integrity: a compromised or tampered package on PyPI will fail hash verification rather than being silently installed.

### Isolated execution: `uv tool run`

`PackagedBuilder` and `PackagedRunner` invoke xorq CLI commands via `uv tool run --isolated --with <wheel> --with-requirements <requirements.txt> xorq build|run`. This creates a temporary, isolated environment with the wheel installed alongside its pinned dependencies, runs the command, and discards the environment.

This replaces the traditional pattern of creating a venv, pip-installing the package and its dependencies, invoking the command, and cleaning up. `uv tool run` collapses these four steps into one, and the `--isolated` flag guarantees no leakage from the user's global or project environment.

#### How `--with` and `--with-requirements` resolve together

`--with <wheel>` installs the project itself (the wheel) as a package. `--with-requirements <requirements.txt>` installs the project's pinned transitive dependencies. uv resolves both together in a single environment:

1. The requirements from `--with-requirements` are installed first as exact-version pins (with hash verification). These satisfy the transitive dependency graph.
2. The wheel from `--with` is installed directly — no build step needed since it is already a wheel. Its declared dependencies in `pyproject.toml` (e.g., `numpy>=1.20`) are already satisfied by the pinned versions from step 1, so no additional resolution occurs.

If the wheel declares a dependency that is *not* in `requirements.txt`, uv will resolve and install it — but this indicates a bug: the lock file is out of sync with `pyproject.toml`. The pipeline prevents this by always deriving `requirements.txt` from `uv.lock`, which is generated from the project's `pyproject.toml`.

The `--isolated` flag ensures this resolution happens in a clean environment with no pre-existing packages. Without it, packages from the user's tool environment could leak in and mask missing dependencies.

### Build directory and catalog entries

`PackagedBuilder` produces a build directory containing serialized expression files (`expr.yaml`, `expr_metadata.json`, `build_metadata.json`, `profiles.yaml`) plus the wheel and `requirements.txt` as sidecar files. `REQUIRED_ARCHIVE_NAMES` in `enums.py` enforces that the serialized files and `requirements.txt` are present when the directory is zipped into a catalog entry. Wheel presence is validated separately by `test_zip` in `zip_utils.py`, which asserts that exactly one `.whl` file exists in the archive. Together, these checks make it impossible to catalog an incomplete build.

`_ensure_wheel_artifacts` checks for the wheel and requirements sidecars in a build directory and builds them from a project's `pyproject.toml` if missing. The project is located by an explicit `project_path` kwarg when passed, and otherwise by walking upward from the current working directory — the explicit form is required when the caller's cwd is not inside the project (e.g. a Jupyter kernel started from `/tmp`). `catalog.add()` accepts `project_path` and threads it through.

The call has two owners, one per flow:

- `Catalog._add_build_dir` invokes it before zipping any build directory (whether produced by `PackagedBuilder`, by direct `build_expr`, or supplied by the user) into a catalog entry.
- `build_expr_context_zip` invokes it before zipping a build directory for standalone use (Flight exchange, `catalog.push`).

Together these ensure that every build archive contains the artifacts needed for isolated execution via `PackagedRunner`.

### Python version threading: `--python`

The `--python <version>` flag is passed to both `uv build` and `uv tool run`, ensuring the same Python version is used for building and execution.

#### How the Python version specifier is derived

`_read_requires_python` reads the `requires-python` specifier from `pyproject.toml` (e.g., `>=3.10`), a directory containing one, or a `.whl` file's `METADATA`. The specifier is intersected with `PYTHON_VERSION_CAP = SpecifierSet("<3.14")` to produce a bounded range (e.g., `>=3.10,<3.14`). This specifier string — not a concrete version — is passed to uv via the `--python` flag, and uv resolves it to the highest installed or downloadable interpreter that satisfies the constraint.

The version specifier is threaded through the entire pipeline: `WheelPackager` passes it to `uv build`, and `PackagedBuilder`/`PackagedRunner` pass it to `uv tool run`. If `python_version` is explicitly provided to the constructor, it overrides the auto-detected value.

uv discovers or downloads the requested Python version automatically, so no external version manager (pyenv, system alternatives) is needed.

### Nix shell compatibility

A shared `_nix_env()` helper detects `in_nix_shell()` and overrides `LD_LIBRARY_PATH` from `UV_TOOL_RUN_LD_LIBRARY_PATH`. Every uv invocation in the pipeline — `uv build`, `uv export`, and `uv tool run` — passes this env through, so the workaround applies pipeline-wide rather than only at execution time. This is necessary because Nix's isolated library paths conflict with uv's downloaded Python interpreters. This workaround is specific to the uv execution model — a traditional venv approach would face the same Nix friction but require a different workaround.

## Rationale

### Why a single tool over best-of-breed per step?

Each integration seam between tools is a source of non-reproducibility and a maintenance burden. When pip resolves differently from pip-compile, or when a venv created by `python -m venv` behaves differently from one populated by pip, debugging crosses tool boundaries. A single tool with a unified resolver, cache, and environment model eliminates these seams.

### Why uv specifically?

1. **Speed.** uv resolves and installs dependencies 10-50x faster than pip, which directly impacts `WheelPackager` and `uv tool run` latency. For a pipeline that builds, locks, exports, and executes, the cumulative speedup is significant.

2. **No bootstrap problem.** `python -m build` requires `build` to be pre-installed. `pip-compile` requires `pip-tools`. uv is a single static binary with no Python dependency — it *is* the build frontend, resolver, and environment manager.

3. **Deterministic resolution.** uv's resolver produces identical results across platforms and runs given the same inputs. Combined with `uv.lock`, the entire dependency graph is reproducible.

4. **Global cache.** uv caches packages globally, so repeated builds (CI, multiple worktrees, packager re-runs) do not re-download dependencies.

5. **`uv tool run` as a primitive.** The ability to run a command in an ephemeral, isolated environment with specific dependencies — without creating a persistent venv — is the key enabler for `PackagedBuilder` and `PackagedRunner`. pip has no equivalent single command.

### Why sidecar files instead of embedding in the archive?

Wheels have a standardized internal structure (`.dist-info/`, package directories) that does not accommodate arbitrary files like `uv.lock` or `requirements.txt` at the top level. Rather than fighting the format, the pipeline stores `requirements.txt` alongside the wheel as a sidecar file in the build directory. Both files are then zipped together into the catalog entry.

This approach is simpler than the previous sdist-based design, which embedded `uv.lock` and `requirements.txt` inside the sdist zip using `append_toplevel` / `replace_toplevel` operations. The sidecar approach eliminates archive manipulation entirely — files are just copied.

## Consequences

### Positive

- **Single dependency.** The pipeline requires only `uv` and `git` at the system level. No pip, pip-tools, pyenv, or `build` package needed.
- **Reproducible builds.** The same `uv.lock` produces the same `requirements.txt` produces the same execution environment, with no resolver drift between steps.
- **Fast iteration.** uv's speed plus direct wheel builds make the full package → build → run cycle fast enough for interactive development, not just CI.
- **Hermetic execution.** `uv tool run --isolated` guarantees that `PackagedBuilder` and `PackagedRunner` see only the declared dependencies, catching missing-dependency bugs early.

### Negative

- **uv is a hard dependency.** If uv is unavailable (e.g., a restricted environment that only allows pip), the entire pipeline is inoperable. There is no fallback to traditional tools.
- **uv-specific lock format.** `uv.lock` is not consumable by pip-tools or poetry. Users who need to integrate with non-uv workflows must use the exported `requirements.txt`, which preserves version pins and hashes but loses uv-specific lock metadata (source URLs, resolution markers).
- **Version coupling.** The pipeline implicitly depends on uv's CLI interface (`uv build`, `uv lock`, `uv export`, `uv tool run`). A breaking change in uv's CLI would require updating the pipeline. This risk is mitigated by uv's stability guarantees and by pinning the uv version in CI.
- **Nix friction.** The `LD_LIBRARY_PATH` workaround for Nix shells is fragile and specific to how uv manages Python interpreters. Changes to either uv's interpreter discovery or Nix's library isolation could break this workaround.
- **Per-invocation latency of `uv tool run`.** Each `uv tool run --isolated` invocation pays ~0.5s of overhead for environment resolution and Python interpreter startup, even when the tool cache is warm. This cost is additive with the inner `xorq run` startup (~0.55s for CLI framework + `load_expr` import chain), giving a total fixed overhead of ~1.2s before any expression evaluation begins. For interactive workflows where `xorq uv-run` is called repeatedly, this overhead dominates wall time for fast expressions. A potential mitigation is switching to `uv tool install` with a persistent, cached tool environment — trading hermeticity (the environment persists between runs) for lower per-invocation cost.
