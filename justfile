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

# generate API documentation
docs-apigen *args:
    cd docs && uv run --no-sync quartodoc interlinks
    uv run --no-sync quartodoc build {{ args }} --config docs/_quarto.yml

# build documentation
docs-render:
    uv run --no-sync quarto render docs

# deploy docs to netlify
docs-deploy:
    uv run --no-sync quarto publish --no-prompt --no-browser --no-render netlify docs

# run the entire docs build pipeline
docs-build-all:
    just docs-apigen --verbose
    just docs-render
