# .envrcs/

direnv configuration split into composable fragments, sourced from the root `.envrc`.

## Tracked (templates & helpers)

| File | Purpose |
|---|---|
| `.envrc.sops` | Defines `use_sops` and `use_sops_if_exists` for decrypting secrets |
| `.envrc.nix-config` | Bootstraps nix-direnv and sets `NIX_CONF_DIR` |
| `.envrc.secrets.template` | Shows which sops-encrypted files the secrets layer expects |
| `.envrc.user.template` | Default user env: sources `.envrc.user.uv`, loads optional dotenvs |
| `.envrc.user.flake` | User env variant: nix flake (immutable install) |
| `.envrc.user.editable` | User env variant: nix flake with editable install |
| `.envrc.user.uv` | User env variant: uv sync + venv activation |

## Gitignored (local / secret)

These are created locally by copying templates or by other tooling. Never commit them.

| File | Created from |
|---|---|
| `.envrc.secrets` | `.envrc.secrets.template` |
| `.envrc.user` | Copy one of `.envrc.user.{template,flake,editable}` |
| `.env.secrets.*` | sops-encrypted secret bundles |
| `.env.catalog.*` | Catalog backend connection config |
| `.env.user.*` | Per-developer dotenv overrides |

## Path conventions

- **`$direnv_root`** — exported by the root `.envrc`; points to the repo root. Use it for paths that must survive worktree copies (e.g. `$direnv_root/.venv`).
- **`$this_dir`** — set locally via `this_dir=$(dirname "$(realpath "${BASH_SOURCE[0]}")")` at the top of a fragment. Use it for paths relative to `.envrcs/` itself (e.g. `$this_dir/.env.secrets.concatenated.sops-encrypted`).

Prefer these over bare relative paths like `../.venv` or `.env.foo`, which break when the evaluation directory isn't what you expect.

## How it fits together

```
.envrc (repo root)
├── watch_file pyproject.toml
├── export direnv_root
├── source_env_if_exists .envrcs/.envrc.secrets
│   └── .envrc.sops → use_sops on encrypted .env files
└── source_env_if_exists .envrcs/.envrc.user
    └── one of: .envrc.user.{template,flake,editable}
```

To get started, copy the templates:

```sh
cp .envrcs/.envrc.secrets.template .envrcs/.envrc.secrets
cp .envrcs/.envrc.user.template .envrcs/.envrc.user
direnv allow
```

## Git worktrees

A new worktree has none of the gitignored files above, so direnv there loads no secrets and no user env until you copy them from the main checkout. It has no `.gitignore` either, since that file is untracked: copy it first, or those secrets show up as untracked files `git add` will stage. It also lacks the test data in `ci/ibis-testing-data`: run `just download-data` in the worktree, or symlink the main checkout's copy.

Maintainers use the worktree helpers in the [devcontainer repo](https://github.com/xorq-labs/devcontainer) (`new-worktree`, `setup-worktree`), which do all of this; see its README. They are optional and Linux-only.
