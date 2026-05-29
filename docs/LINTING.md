# Documentation linting

Run all checks at once:

```bash
just docs-lint            # internal links only
just docs-lint-external   # + external URL verification (~5 min)
```

---

## Checks

| Check | Tool | What it catches |
|-------|------|-----------------|
| Prose style | vale | Grammar, word choice, tone (Google + Xorq rules) |
| Structure | quarto render | Broken includes, bad frontmatter, missing files |
| Internal links | lychee | Broken relative links in rendered HTML |
| External URLs | lychee `--external` | Dead external links (opt-in) |
| Markdown structure | pymarkdown | Heading level skips, list formatting, bare URLs |
| Frontmatter | python-frontmatter | Missing `title:` in non-reference `.qmd` files |
| Orphan files | bash | `.qmd` files not referenced in `_quarto.yml` |

---

## Install required tools

### vale

```bash
# Linux
sudo snap install vale --classic

# macOS
brew install vale
```

> **Note:** The `.vale.ini` and `styles/` directory are already committed. No `vale sync` needed.

### lychee

Download the binary for your platform from the [lychee releases page](https://github.com/lycheeverse/lychee/releases).

```bash
# Linux (example for v0.15 — check releases page for latest)
curl -sSL https://github.com/lycheeverse/lychee/releases/latest/download/lychee-x86_64-unknown-linux-gnu.tar.gz \
  | tar xz -C ~/.local/bin

# macOS
brew install lychee
```

### quarto

Download from [quarto.org/docs/get-started](https://quarto.org/docs/get-started/).

### pymarkdown and python-frontmatter

Both are in the `docs` dependency group:

```bash
uv sync --group docs
```

Or individually:

```bash
pip install pymarkdownlnt python-frontmatter
```

---

## Editor integration

### Vale in VS Code

1. Install the [Vale VSCode extension](https://marketplace.visualstudio.com/items?itemName=ChrisChinchilla.vale-vscode) (by Chris Chinchilla).
2. Open Settings (`Ctrl+,`), search **Vale CLI: Config**, set to `.vale.ini`.
3. Set **Vale CLI: Min Alert Level** to `suggestion`.
4. Reload VS Code.

Vale suggestions appear as squiggly underlines in any `.qmd` or `.md` file under `docs/`.

---

## CI

`docs/lint.sh` detects `CI=true` and emits [GitHub Actions annotations](https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/workflow-commands-for-github-actions#setting-an-error-message) (`::error file=...`) so errors appear inline on PR diffs.
