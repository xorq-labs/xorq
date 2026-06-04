#!/usr/bin/env bash
# docs/lint.sh — Documentation linter
#
# Checks (all run; failures collected; non-zero exit if any fail):
#   0. cli reference — regenerate CLI pages from the Click commands
#   1. vale          — prose style (Google + Xorq rules)
#   2. quartodoc     — regenerate API reference from the package
#   3. quarto render — structure validation, no code execution
#   4. lychee        — broken internal links against rendered _site/
#   5. lychee        — broken external URLs (opt-in via --external)
#   6. pymarkdown    — markdown structure (heading levels, list syntax, etc.)
#   7. frontmatter   — every non-reference .qmd has a title: field
#   8. orphans       — every .qmd is referenced in _quarto.yml
#
# Usage:
#   bash docs/lint.sh              # all checks, internal links only
#   bash docs/lint.sh --external   # also verify external URLs (~5 min)
#
# Required tools:
#   vale        prose style linter      snap install vale --classic  (Linux)
#                                       brew install vale            (macOS)
#   lychee      link checker            https://github.com/lycheeverse/lychee/releases
#   quarto      build + structure       https://quarto.org/docs/get-started/
#   quartodoc   API reference gen        pip install quartodoc
#   pymarkdown  markdown structure      pip install pymarkdownlnt
#   python3     frontmatter parsing     pip install python-frontmatter
#
# CI: set CI=true; errors are emitted as GitHub Actions ::error annotations.

set -uo pipefail

DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Vale reads .vale.ini from the repo root (moved there for vale-action), so vale
# must run from there with a docs/ path; DOCS_DIR is <repo>/docs.
REPO_ROOT="$(dirname "$DOCS_DIR")"
CHECK_EXTERNAL=false
FAILURES=0

for arg in "$@"; do
    case "$arg" in
        --external) CHECK_EXTERNAL=true ;;
        *) printf 'Unknown argument: %s\n' "$arg" >&2; exit 1 ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────

in_ci()   { [[ "${CI:-}" == "true" ]]; }

section() { printf '\n━━━ %s ━━━\n' "$1"; }

pass()    { printf '  ✓ %s\n' "$1"; }

fail()    { printf '  ✗ %s\n' "$1"; FAILURES=$((FAILURES + 1)); }

# Emit a GitHub Actions inline annotation when running in CI.
gha_error() {
    # gha_error <file> <line> <message>
    in_ci && printf '::error file=%s,line=%s::%s\n' "$1" "$2" "$3"
}

require_tool() {
    local cmd="$1" hint="$2"
    command -v "$cmd" &>/dev/null && return
    printf 'ERROR: %s not found.\n  Install: %s\n  See docs/LINTING.md for details.\n' "$cmd" "$hint"
    exit 1
}

# ── Tool presence checks (fail hard) ─────────────────────────────────────────

[[ "${SKIP_VALE:-}"   == "1" ]] || require_tool vale   "snap install vale --classic  (Linux)  |  brew install vale  (macOS)"
[[ "${SKIP_LYCHEE:-}" == "1" ]] || require_tool lychee "https://github.com/lycheeverse/lychee/releases"
require_tool quarto     "https://quarto.org/docs/get-started/"
require_tool quartodoc  "pip install quartodoc"
require_tool pymarkdown "pip install pymarkdownlnt"
python3 -c "import frontmatter" 2>/dev/null || {
    printf 'ERROR: python-frontmatter not installed.\n  Install: pip install python-frontmatter\n'
    exit 1
}

cd "$DOCS_DIR"

# ── 0. CLI reference generation ───────────────────────────────────────────────

# api_reference/cli/*.qmd are generated from the Click commands (gitignored,
# like reference/). Generate before the prose and structure checks so vale
# lints the docstring-sourced pages and render/orphan checks see them.
section "CLI reference — generate from Click commands"
if python3 "$DOCS_DIR/generate_cli_reference.py" 2>&1; then
    pass "generate_cli_reference.py"
else
    fail "generate_cli_reference.py: failed"
    in_ci && printf '::error file=docs/generate_cli_reference.py,line=1::CLI reference generation failed\n'
fi

# ── 1. Vale — prose style ─────────────────────────────────────────────────────

section "Vale — prose style"
if [[ "${SKIP_VALE:-}" == "1" ]]; then
    pass "vale: skipped (handled by vale-cli/vale-action in CI)"
else
    vale_out=$( (cd "$REPO_ROOT" && vale docs/) 2>&1) && vale_exit=0 || vale_exit=$?
    printf '%s\n' "$vale_out"
    if [[ $vale_exit -eq 0 ]]; then
        pass "vale"
    else
        fail "vale: style errors found (see output above)"
    fi
fi

# ── 2. Quartodoc build — regenerate API reference ────────────────────────────

# reference/*.qmd are generated from the package via quartodoc (config lives in
# the `quartodoc:` block of _quarto.yml). Rebuild before rendering so the API
# reference reflects the current code and stale pages from renamed/removed
# symbols don't linger.
section "Quartodoc build — API reference"
if quartodoc build 2>&1; then
    pass "quartodoc build"
else
    fail "quartodoc build: failed"
    in_ci && printf '::error file=_quarto.yml,line=1::quartodoc build failed\n'
fi

# ── 3. Quarto render — structure validation ───────────────────────────────────

section "Quarto render — structure (--no-execute)"
if quarto render --no-execute 2>&1; then
    pass "quarto render --no-execute"
else
    fail "quarto render: build failed"
    in_ci && printf '::error file=_quarto.yml,line=1::quarto render --no-execute failed\n'
fi

# ── 4. Lychee — internal links ────────────────────────────────────────────────

section "Lychee — internal links"
# _site/reference/ is auto-generated by quartodoc. Its pages carry unresolved
# interlinks (raw `.qmd` targets, backtick-encoded type refs, and `.html#member`
# anchors for upstream ibis objects) because the interlinks filter + objects.inv
# inventory aren't wired up. Those are a quartodoc-config concern, not authored
# link rot, so exclude the generated tree here — hand-written docs are still checked.
if [[ "${SKIP_LYCHEE:-}" == "1" ]]; then
    pass "lychee: skipped (handled by lycheeverse/lychee-action in CI)"
elif [[ ! -d "_site" ]]; then
    fail "lychee: _site/ not found — run quarto render first (check 2 may have failed)"
else
    if lychee --offline --include-fragments --no-progress \
        --exclude-path '_site/reference' '_site/**/*.html' 2>&1; then
        pass "lychee: no broken internal links"
    else
        fail "lychee: broken internal links found"
    fi
fi

# ── 5. Lychee — external URLs (opt-in, local only) ───────────────────────────

if [[ "$CHECK_EXTERNAL" == "true" ]]; then
    section "Lychee — external URLs"
    if [[ "${SKIP_LYCHEE:-}" == "1" ]]; then
        pass "lychee --external: skipped (handled by lycheeverse/lychee-action in CI)"
    elif [[ ! -d "_site" ]]; then
        fail "lychee --external: _site/ not found"
    else
        if lychee \
            --exclude 'localhost' \
            --exclude '127\.0\.0\.1' \
            --exclude-path '_site/reference' \
            --timeout 20 \
            --retry-wait-time 2 \
            --max-retries 3 \
            --no-progress \
            '_site/**/*.html' 2>&1; then
            pass "lychee: no broken external URLs"
        else
            fail "lychee: broken external URLs found"
        fi
    fi
fi

# ── 6. pymarkdown — markdown structure ───────────────────────────────────────

section "pymarkdown — markdown structure"
# Rules disabled for Quarto compatibility:
#   MD013  line-length     — prose doesn't need hard line limits
#   MD025  single-title    — Quarto titles live in YAML frontmatter, not an h1
#   MD033  no-inline-html  — Quarto callout blocks (:::) render as divs
#   MD034  no-bare-urls    — covered by Vale Google.URIs rule
#   MD041  first-heading   — Quarto files open with --- YAML frontmatter
#   MD046  code-block-style— Quarto uses ```{python} fenced syntax
DISABLE="MD013,MD025,MD033,MD034,MD041,MD046"
_pm_files=()
while IFS= read -r f; do _pm_files+=("$f"); done < <(find . \( -name "*.qmd" -o -name "*.md" \) \
    -not -path "./_site/*" \
    -not -path "./.quarto/*" \
    -not -path "./reference/*" \
    -not -path "./adr/*" \
    -not -path "./__pycache__/*" \
    | sort)
if [[ ${#_pm_files[@]} -eq 0 ]]; then
    pass "pymarkdown: no files to scan"
elif pymarkdown --disable-rules "$DISABLE" scan "${_pm_files[@]}" 2>&1; then
    pass "pymarkdown"
else
    fail "pymarkdown: markdown structure issues found (see output above)"
fi

# ── 7. Frontmatter — title presence ──────────────────────────────────────────

section "Frontmatter — title: present in non-reference .qmd files"
missing_titles=()
qmd_files=()
while IFS= read -r f; do
    qmd_files+=("$f")
done < <(find . -name "*.qmd" \
    -not -path "./_site/*" \
    -not -path "./.quarto/*" \
    -not -path "./reference/*" \
    -not -path "./adr/*" \
    -not -path "*/_*/*" \
    -not -path "./index.qmd" \
    -not -path "./__pycache__/*" \
    | sort)

# Check every file in one python3 process (one spawn, not one-per-file). The
# script prints the path of each file missing a title:; a parse error counts as
# missing, matching the prior per-file behavior.
if [[ ${#qmd_files[@]} -gt 0 ]]; then
    while IFS= read -r f; do
        missing_titles+=("$f")
        gha_error "docs/${f#./}" "1" "Missing 'title:' in YAML frontmatter"
    done < <(python3 - "${qmd_files[@]}" <<'PYEOF'
import sys
try:
    import frontmatter
except Exception as e:
    print(f"frontmatter import error: {e}", file=sys.stderr)
    sys.exit(1)
for path in sys.argv[1:]:
    try:
        post = frontmatter.load(path)
        if "title" not in post.metadata:
            print(path)
    except Exception as e:
        print(f"frontmatter parse error in {path}: {e}", file=sys.stderr)
        print(path)
PYEOF
)
fi

if [[ ${#missing_titles[@]} -eq 0 ]]; then
    pass "all non-reference .qmd files have title frontmatter"
else
    fail "${#missing_titles[@]} file(s) missing 'title:' frontmatter"
    printf '    %s\n' "${missing_titles[@]}"
fi

# ── 8. Orphan files — not referenced in nav, sidebars, or any .qmd link ──────

section "Orphan files — .qmd not referenced in nav, sidebars, or any .qmd link"
orphans=()
while IFS= read -r f; do
    rel="${f#./}"
    base="$(basename "$rel")"
    dir="$(dirname "$rel")"
    # Escape regex metachars ('.') so the path is matched literally below.
    rel_esc="${rel//./\\.}"
    base_esc="${base//./\\.}"

    # Reachable if the full path appears in any config/sidebar (.yml) or .qmd.
    # Catches _quarto.yml nav, generated reference/_sidebar.yml, and links that
    # spell out the directory (e.g. ../how_to/page.qmd).
    #
    # Match $rel only at a path boundary — a non-path char (or line start),
    # optionally followed by ./ or ../ prefixes. This stops a short root-level
    # name like 'index.qmd' from being counted as referenced just because some
    # nested path (e.g. 'reference/index.qmd') contains it as a substring.
    if grep -rqE "(^|[^[:alnum:]_.-])(\.\.?/)*${rel_esc}" --include="*.yml" --include="*.qmd" . 2>/dev/null; then
        continue
    fi

    # Reachable via a relative link from a sibling .qmd in the same directory
    # (e.g. index.qmd linking [Overview](overview.qmd) — only the basename appears).
    # Anchor the basename to a markdown/HTML link opener (" or () with an
    # optional ./ so prose mentions or longer names (my_overview.qmd) don't match.
    sibling_link=""
    while IFS= read -r sib; do
        [[ "$sib" == "./$rel" || "$sib" == "$rel" ]] && continue
        sibling_link=1; break
    done < <(grep -rlE "[\"(](\./)?${base_esc}" "$dir" --include="*.qmd" 2>/dev/null)
    [[ -n "$sibling_link" ]] && continue

    orphans+=("$rel")
    gha_error "docs/$rel" "1" "File not referenced in nav, sidebars, or any .qmd link (orphan)"
done < <(find . -name "*.qmd" \
    -not -path "./_site/*" \
    -not -path "./.quarto/*" \
    -not -path "./adr/*" \
    -not -path "*/_*/*" \
    -not -path "./__pycache__/*" \
    -not -path "./api_reference/cli/*" \
    | sort)
# api_reference/cli/ is excluded above: those pages are generated (check 0)
# and reach the nav via the `auto:` globs in _quarto.yml, which this textual
# scan can't see; the generator itself errors on any command missing from
# its curated nav config.

if [[ ${#orphans[@]} -eq 0 ]]; then
    pass "no orphaned .qmd files"
else
    fail "${#orphans[@]} .qmd file(s) not referenced in nav, sidebars, or any .qmd link"
    printf '    %s\n' "${orphans[@]}"
fi

# ── Summary ───────────────────────────────────────────────────────────────────

printf '\n%s\n' "════════════════════════════════════════"
if [[ $FAILURES -eq 0 ]]; then
    printf '  All checks passed.\n'
    exit 0
else
    printf '  %d check(s) failed.\n' "$FAILURES"
    in_ci && printf '::error::Documentation lint: %d check(s) failed\n' "$FAILURES"
    exit 1
fi
