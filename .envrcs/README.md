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

### Quick start

Check out an existing branch in a fully configured worktree:

```sh
dev/new-worktree feat/my-feature
# creates ../xorq-feat-my-feature, copies envrc files, symlinks .gitignore, runs direnv allow
```

### Manual setup

If you already have a worktree, run `setup-worktree` inside it:

```sh
git worktree add ../xorq-pr-123 pr-123-branch
cd ../xorq-pr-123
dev/setup-worktree
direnv allow
```

### What the scripts do

- **`new-worktree <branch>`** — checks out an existing branch into a sibling worktree at `../xorq-<branch>` (slashes become dashes), runs `setup-worktree`, then `direnv allow`. Prints the worktree path on stdout.
- **`setup-worktree`** — copies `.envrc.secrets`, `.envrc.user`, and `.env.*` files from the main worktree's `.envrcs/`. Symlinks `.gitignore`. Writes all created files to `.envrcs/.worktree-manifest`.
- **`cleanup-worktree`** — removes everything listed in `.envrcs/.worktree-manifest`, then runs `git worktree remove`.

### Clean removal

```sh
cd ../xorq-feat-my-feature
dev/cleanup-worktree
```
