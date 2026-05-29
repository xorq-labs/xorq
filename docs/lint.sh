#!/usr/bin/env bash
# docs/lint.sh — Documentation linter
#
# Checks (all run; failures collected; non-zero exit if any fail):
#   1. vale          — prose style (Google + Xorq rules)
#   2. quarto render — structure validation, no code execution
#   3. lychee        — broken internal links against rendered _site/
#   4. lychee        — broken external URLs (opt-in via --external)
#   5. pymarkdown    — markdown structure (heading levels, list syntax, etc.)
#   6. frontmatter   — every non-reference .qmd has a title: field
#   7. orphans       — every .qmd is referenced in _quarto.yml
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
#   pymarkdown  markdown structure      pip install pymarkdownlnt
#   python3     frontmatter parsing     pip install python-frontmatter
#
# CI: set CI=true; errors are emitted as GitHub Actions ::error annotations.

set -uo pipefail

DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

require_tool vale       "snap install vale --classic  (Linux)  |  brew install vale  (macOS)"
require_tool lychee     "https://github.com/lycheeverse/lychee/releases"
require_tool quarto     "https://quarto.org/docs/get-started/"
require_tool pymarkdown "pip install pymarkdownlnt"
python3 -c "import frontmatter" 2>/dev/null || {
    printf 'ERROR: python-frontmatter not installed.\n  Install: pip install python-frontmatter\n'
    exit 1
}

cd "$DOCS_DIR"

# ── 1. Vale — prose style ─────────────────────────────────────────────────────

section "Vale — prose style"
vale_out=$(vale --glob='!adr/**' . 2>&1) && vale_exit=0 || vale_exit=$?
printf '%s\n' "$vale_out"
if [[ $vale_exit -eq 0 ]]; then
    pass "vale"
else
    fail "vale: style errors found (see output above)"
    if in_ci; then
        vale --output=JSON . 2>/dev/null \
        | python3 -c "
import sys, json
data = json.load(sys.stdin)
for path, alerts in data.items():
    for a in alerts:
        col = a['Span'][0] if a.get('Span') else 1
        print(f'::error file={path},line={a[\"Line\"]},col={col}::{a[\"Message\"]} [{a[\"Check\"]}]')
" 2>/dev/null || true
    fi
fi

# ── 2. Quarto render — structure validation ───────────────────────────────────

section "Quarto render — structure (--no-execute)"
if quarto render --no-execute . 2>&1; then
    pass "quarto render --no-execute"
else
    fail "quarto render: build failed"
    in_ci && printf '::error file=_quarto.yml,line=1::quarto render --no-execute failed\n'
fi

# ── 3. Lychee — internal links ────────────────────────────────────────────────

section "Lychee — internal links"
if [[ ! -d "_site" ]]; then
    fail "lychee: _site/ not found — run quarto render first (check 2 may have failed)"
else
    if lychee --offline --include-fragments --no-progress '_site/**/*.html' 2>&1; then
        pass "lychee: no broken internal links"
    else
        fail "lychee: broken internal links found"
    fi
fi

# ── 4. Lychee — external URLs (opt-in) ───────────────────────────────────────

if [[ "$CHECK_EXTERNAL" == "true" ]]; then
    section "Lychee — external URLs"
    if [[ ! -d "_site" ]]; then
        fail "lychee --external: _site/ not found"
    else
        if lychee \
            --exclude 'localhost' \
            --exclude '127\.0\.0\.1' \
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

# ── 5. pymarkdown — markdown structure ───────────────────────────────────────

section "pymarkdown — markdown structure"
# Rules disabled for Quarto compatibility:
#   MD013  line-length     — prose doesn't need hard line limits
#   MD025  single-title    — Quarto titles live in YAML frontmatter, not an h1
#   MD033  no-inline-html  — Quarto callout blocks (:::) render as divs
#   MD034  no-bare-urls    — covered by Vale Google.URIs rule
#   MD041  first-heading   — Quarto files open with --- YAML frontmatter
#   MD046  code-block-style— Quarto uses ```{python} fenced syntax
DISABLE="MD013,MD025,MD033,MD034,MD041,MD046"
mapfile -t _pm_files < <(find . \( -name "*.qmd" -o -name "*.md" \) \
    -not -path "./_site/*" \
    -not -path "./.quarto/*" \
    -not -path "./reference/*" \
    -not -path "./adr/*" \
    -not -path "./__pycache__/*" \
    | sort)
if [[ ${#_pm_files[@]} -gt 0 ]] && \
   pymarkdown --disable-rules "$DISABLE" scan "${_pm_files[@]}" 2>&1; then
    pass "pymarkdown"
else
    fail "pymarkdown: markdown structure issues found (see output above)"
fi

# ── 6. Frontmatter — title presence ──────────────────────────────────────────

section "Frontmatter — title: present in non-reference .qmd files"
missing_titles=()
while IFS= read -r f; do
    has_title=$(python3 - "$f" <<'PYEOF'
import sys
try:
    import frontmatter
    post = frontmatter.load(sys.argv[1])
    print("yes" if "title" in post.metadata else "no")
except Exception:
    print("no")
PYEOF
)
    if [[ "$has_title" == "no" ]]; then
        missing_titles+=("$f")
        gha_error "docs/${f#./}" "1" "Missing 'title:' in YAML frontmatter"
    fi
done < <(find . -name "*.qmd" \
    -not -path "./_site/*" \
    -not -path "./.quarto/*" \
    -not -path "./reference/*" \
    -not -path "./adr/*" \
    -not -path "./__pycache__/*" \
    | sort)

if [[ ${#missing_titles[@]} -eq 0 ]]; then
    pass "all non-reference .qmd files have title frontmatter"
else
    fail "${#missing_titles[@]} file(s) missing 'title:' frontmatter"
    printf '    %s\n' "${missing_titles[@]}"
fi

# ── 7. Orphan files — not referenced in _quarto.yml ──────────────────────────

section "Orphan files — .qmd not in _quarto.yml"
orphans=()
while IFS= read -r f; do
    rel="${f#./}"
    if ! grep -qF "$rel" _quarto.yml 2>/dev/null; then
        orphans+=("$rel")
        gha_error "docs/$rel" "1" "File not referenced in _quarto.yml (orphan)"
    fi
done < <(find . -name "*.qmd" \
    -not -path "./_site/*" \
    -not -path "./.quarto/*" \
    -not -path "./adr/*" \
    -not -path "./__pycache__/*" \
    | sort)

if [[ ${#orphans[@]} -eq 0 ]]; then
    pass "no orphaned .qmd files"
else
    fail "${#orphans[@]} .qmd file(s) not referenced in _quarto.yml"
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
