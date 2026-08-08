# list justfile recipes
default:
    just --list

# clean untracked files
clean:
    git clean -fdx -e 'ci/ibis-testing-data'

# format code
fmt:
    black .
    blackdoc .
    ruff --fix .

# lint code
lint:
    black -q . --check
    ruff .

# download testing data
download-data owner="ibis-project" repo="testing-data" rev="master":
    #!/usr/bin/env bash
    outdir="{{ justfile_directory() }}/ci/ibis-testing-data"
    rm -rf "$outdir"
    url="https://github.com/{{ owner }}/{{ repo }}"

    args=("$url")
    if [ "{{ rev }}" = "master" ]; then
        args+=("--depth" "1")
    fi

    args+=("$outdir")
    git clone "${args[@]}"

    if [ "{{ rev }}" != "master" ]; then
        git -C "${outdir}" checkout "{{ rev }}"
    fi

# start backends using docker compose; no arguments starts all backends
up *backends:
    docker compose up --build --wait {{ backends }}

# start backends in CI: no rebuild, pull only missing images
up-ci *backends:
    docker compose up --wait --pull missing {{ backends }}

# scaffold a new ADR; `just adr-rename` numbers it once the pull request exists
adr-new slug:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{ justfile_directory() }}"
    dest="docs/adr/XXXX-{{ slug }}.md"
    if [ -e "$dest" ]; then
        echo "$dest already exists" >&2
        exit 1
    fi
    cp docs/adr/template.md "$dest"
    echo "created $dest"
    echo "next: write it, open the pull request, then run 'just adr-rename'"

# give the in-flight ADR the number of this branch's pull request
# pass a slug to pick one when several ADRs are in flight at once
adr-rename slug="":
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{ justfile_directory() }}"
    shopt -s nullglob
    if [ -n "{{ slug }}" ]; then
        src="docs/adr/XXXX-{{ slug }}.md"
        if [ ! -e "$src" ]; then
            echo "$src does not exist" >&2
            exit 1
        fi
    else
        placeholders=(docs/adr/XXXX-*.md)
        if [ "${#placeholders[@]}" -eq 0 ]; then
            echo "no docs/adr/XXXX-*.md to rename" >&2
            exit 1
        fi
        if [ "${#placeholders[@]}" -gt 1 ]; then
            echo "more than one unnumbered ADR:" >&2
            printf '  %s\n' "${placeholders[@]}" >&2
            echo "pass the one this pull request adds: just adr-rename <slug>" >&2
            exit 1
        fi
        src="${placeholders[0]}"
    fi
    if ! pr="$(gh pr view --json number --jq .number 2>/dev/null)"; then
        echo "no pull request found for this branch; open one first" >&2
        exit 1
    fi
    stem="${src#docs/adr/XXXX-}"
    dest="docs/adr/${pr}-${stem}"
    git mv "$src" "$dest"
    python3 -c 'import pathlib, sys; p = pathlib.Path(sys.argv[1]); p.write_text(p.read_text().replace("# ADR-XXXX:", "# ADR-" + sys.argv[2] + ":", 1))' "$dest" "$pr"
    echo "renamed to $dest"
    # Settle named references on the short numeric form now that one exists.
    # Best-effort by design: both forms resolve forever, so a reference this
    # misses stays valid rather than breaking the build.
    python3 scripts/adr_sweep_refs.py "${stem%.md}" "$pr"
    python3 scripts/adr_check.py --base main --pr "$pr"

# check ADR numbering and cross-references
adr-check *args:
    python3 scripts/adr_check.py {{ args }}

# generate API documentation
docs-apigen *args:
    cd docs && uv run --no-sync quartodoc interlinks
    uv run --no-sync quartodoc build {{ args }} --config docs/_quarto.yml
    uv run --no-sync python docs/generate_cli_reference.py
    uv run --no-sync python docs/generate_llms_txt.py

# build documentation
docs-render:
    uv run --no-sync quarto render docs

# lint documentation (vale, lychee, quarto, pymarkdown, frontmatter, orphans)
docs-lint *args:
    uv run --no-sync bash docs/lint.sh {{ args }}

# lint documentation and check external URLs (slow, ~5 min)
docs-lint-external:
    uv run --no-sync bash docs/lint.sh --external

# deploy docs to netlify
docs-deploy:
    uv run --no-sync quarto publish --no-prompt --no-browser --no-render netlify docs

# run the entire docs build pipeline
docs-build-all:
    just docs-apigen --verbose
    just docs-render
