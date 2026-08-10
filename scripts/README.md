# scripts/

Repository tooling that is not part of the shipped package. `pyproject.toml`
builds the wheel and sdist from `python/` only, so nothing here reaches an
installed `xorq`. Code that users import belongs under `python/xorq/`; code that
only maintains this repository belongs here.

Two kinds of thing live in this directory, and the difference matters when one
of them breaks:

| Script | Kind | Invoked by |
| --- | --- | --- |
| `adr_check.py` | guard | `ci-adr.yml` on every pull request, and by hand |
| `adr_index.py` | workflow | by hand, to read the ADRs as a list |
| `adr_new.py` | workflow | by hand, when starting an ADR |
| `adr_rename.py` | workflow | by hand, once the pull request exists |
| `canonical_digest_xver_probe.py` | probe | by hand, when investigating digest stability |

A **guard** fails CI, so it is load-bearing: it needs tests, and a change to it
should assume someone's pull request depends on the answer. A **workflow** script
is run deliberately by a person who can read the error and retry. A **probe** is
a diagnostic kept because rebuilding it costs more than storing it; its docstring
holds the recipe, and nothing runs it unless you do.

State the kind in the module docstring, along with the usage, so the next reader
does not have to grep CI to find out.

## Conventions

Anything CI invokes stays **stdlib-only**. `ci-adr.yml` runs `adr_check.py` with
the runner's bare `python3` — no `setup-python`, no `uv sync`, nothing to keep in
step with the lockfile. A probe may depend on whatever it needs, since a human
builds the environment first.

Scripts are run **directly**, as `python3 scripts/<name>.py`, and the
documentation cites them that way. This is deliberate rather than an oversight:
`just` is not a declared dependency of this project — `CONTRIBUTING.md` asks you
to install it by hand alongside the backend test data — so routing ADR authoring
through a recipe would put a tool install in front of the lightest contribution
there is. Add a `just` recipe when a script needs real wrapping, not to alias it.

This directory is **not a package**: no `__init__.py`, and no import from
`xorq`. Python puts a script's own directory on `sys.path`, so a sibling import
works and is preferable to copying a definition — `adr_rename.py` takes
`FILENAME_RE` from `adr_check.py` that way, which is what stops the two
disagreeing about what a valid ADR filename looks like.

## Tests

Tests live in `scripts/tests/` and run in the `adr-tests` job of `ci-adr.yml`:

```sh
uv run --no-sync pytest scripts/tests
```

They need only `pytest` — the tests reach the scripts by path, so the job runs
`uv run --isolated --no-project --with pytest`, and the guard job stays
install-free. A guard without tests is a guard that can stop firing silently,
which looks exactly like a repository with no problems.

`ruff` and `codespell` cover this directory; see `.pre-commit-config.yaml` and
`ci-lint.yml`.
