# .envrcs/

direnv configuration split into composable fragments, sourced from the root `.envrc`.

The fragments follow [xorq-labs/xorq-envrcs](https://github.com/xorq-labs/xorq-envrcs) at 9e7f67d, whose `.envrcs/README.md` explains the reasoning behind each one. Every fragment matches it except for comments, apart from xorq's additions: `.envrc.nix-config` sets xorq's nix config, `.envrc.user.uv` installs shell completion, `.envrc.user.editable` has no upstream counterpart, and the root `.envrc` keeps its `pyproject.toml` watch (the flake devShells are built from it). The templates are xorq's own.

## Getting started

```sh
direnv allow
```

The first load rebuilds `.gitignore` and creates `.envrc.secrets` and `.envrc.user` from their templates (below). `.env.local` is not created for you; if you need it:

```sh
cp .envrcs/.env.local.template .envrcs/.env.local
```

**If you already have an `.envrc.user`**, it is kept as it is, so it lacks lines the template has gained: `dotenv_if_exists .env.local`, and the commented `source_env .envrc.user.kenn` and `source_env .envrc.kata`. Copy the ones you want from `.envrc.user.template`; without the first, `.env.local` is never read.

**If you already have a `.gitignore`**, the first load compares it with the rebuild. If it has rules the template lacks, it is moved to `.gitignore.orig` and the load says so, once. Until you move the rules you want into `.gitignore.local`, they are not applied, and files they hid show up in `git status` and `git add -A`:

```sh
$EDITOR .gitignore.orig .gitignore.local   # copy the rules you want into .gitignore.local
rm .gitignore.orig
direnv reload
```

## Tracked (templates & helpers)

| File | Purpose |
|---|---|
| `.envrc.gitignore` | Rebuilds the root `.gitignore` from `.gitignore.template` + `.gitignore.local` on every load |
| `.envrc.from-template` | Creates `.envrc.secrets` and `.envrc.user` from their templates when missing; never overwrites, skips linked worktrees |
| `.envrc.sops` | Defines `use_sops` and `use_sops_if_exists` for decrypting secrets |
| `.envrc.nix-config` | Bootstraps nix-direnv and sets `NIX_CONF_DIR` |
| `.envrc.secrets.template` | Shows which sops-encrypted files the secrets layer expects |
| `.envrc.user.template` | Default user env: sources `.envrc.user.uv`, loads optional dotenvs |
| `.envrc.user.flake` | User env variant: nix flake (immutable install) |
| `.envrc.user.editable` | User env variant: nix flake with editable install |
| `.envrc.user.uv` | User env variant: uv sync + venv activation |
| `.envrc.user.kenn` | Optional, additive: kenn-io toolkit (kata among it) on PATH, pinned by `kenn.rev` |
| `.envrc.kata` | Optional: points kata at a team server from `KATA_TEAM_SERVER` and `KATA_TEAM_TOKEN` |
| `.env.local.template` | Per-user, non-secret `KEY=value` settings |
| `test-fragments.sh` | Tests `.envrc.gitignore` and `.envrc.from-template` in scratch repos |

## Gitignored (local / secret)

Never commit these; `.gitignore.template` ignores every row.

| File | Created from |
|---|---|
| `.envrc.secrets` | `.envrc.secrets.template`, on first load |
| `.envrc.user` | `.envrc.user.template`, on first load; edit it to pick `.envrc.user.{flake,editable}` instead of uv |
| `.env.local` | Copy of `.env.local.template` |
| `.env.secrets.*` | sops-encrypted secret bundles |
| `.env.catalog.*` | Catalog backend connection config |
| `.env.user.*` | Per-developer dotenv overrides |
| `/.gitignore.local` | Your own ignore rules, appended to the generated `.gitignore` |

## Path conventions

- **`$direnv_root`** — exported by the root `.envrc`; points to the repo root. Use it for paths that must survive worktree copies (e.g. `$direnv_root/.venv`), and for anything a helper would otherwise resolve against PWD (e.g. `use flake "$direnv_root"`).
- Fragment-relative names (`source_env .envrc.sops`, `dotenv_if_exists .env.local`) resolve against `.envrcs/` itself, so bare names work between fragments.

Prefer these over bare relative paths like `../.venv`, which break when the evaluation directory isn't what you expect.

## How it fits together

```
.envrc (repo root)
├── watch_file pyproject.toml
├── export direnv_root, pin direnv_layout_dir to $direnv_root/.direnv
├── source_env .envrcs/.envrc.gitignore       → rebuild .gitignore
├── source_env .envrcs/.envrc.from-template   → create .envrc.{secrets,user}
├── source_env_if_exists .envrcs/.envrc.secrets
│   └── .envrc.sops → use_sops on encrypted .env files
└── source_env_if_exists .envrcs/.envrc.user
    ├── one of: .envrc.user.{uv,flake,editable}
    ├── optional: .envrc.user.kenn
    ├── dotenv_if_exists .env.local
    └── optional: .envrc.kata
```

## Git worktrees

Worktree helpers (`new-worktree`, `setup-worktree`, `cleanup-worktree`) are provided by the [devcontainer repo](https://github.com/xorq-labs/devcontainer). Clone it as a sibling and add to PATH in `.envrc.user`:

```sh
git clone git@github.com:xorq-labs/devcontainer.git ../devcontainer
# in .envrcs/.envrc.user:
PATH_add $direnv_root/../devcontainer/dev
```

### Quick start

```sh
new-worktree feat/my-feature
# creates ../xorq-feat-my-feature, copies envrc files, symlinks .gitignore, runs direnv allow
```

### Manual setup

```sh
git worktree add ../xorq-pr-123 pr-123-branch
cd ../xorq-pr-123
setup-worktree
direnv allow
```

### Clean removal

```sh
cd ../xorq-feat-my-feature
cleanup-worktree
```
