{ pkgs, python, black, blackdoc, ruff }: let

  letsql-pytest = pkgs.writeShellScriptBin "letsql-pytest" ''
    set -eux

    # see https://docs.pytest.org/en/latest/explanation/pythonpath.html#import-mode-importlib
    ${python}/bin/python -m pytest --import-mode=importlib
  '';

  letsql-clean = pkgs.writeShellScriptBin "letsql-clean" ''
    set -eux

    ${pkgs.git}/bin/git clean --force -dx --exclude 'ci/ibis-testing-data'
  '';

  letsql-fmt = pkgs.writeShellScriptBin "letsql-fmt" ''
    set -eux

    ${black}/bin/black .
    ${blackdoc}/bin/blackdoc .
    ${ruff}/bin/ruff --fix .
  '';

  letsql-lint = pkgs.writeShellScriptBin "letsql-lint" ''
    set -eux

    ${black}/bin/black --quiet --check .
    ${ruff}/bin/ruff .
  '';

  letsql-download-data = pkgs.writeShellScriptBin "letsql-download-data" ''
    set -eux

    owner=''${1:-ibis-project}
    repo=''${1:-testing-data}
    rev=''${1:-master}

    repo_dir=$(realpath $(${pkgs.git}/bin/git rev-parse --git-dir)/..)

    outdir=$repo_dir/ci/ibis-testing-data
    rm -rf "$outdir"
    url="https://github.com/$owner/$repo"

    args=("$url")
    if [ "$rev" = "master" ]; then
        args+=("--depth" "1")
    fi

    args+=("$outdir")
    ${pkgs.git}/bin/git clone "''${args[@]}"

    if [ "$rev" != "master" ]; then
        ${pkgs.git}/bin/git -C "''${outdir}" checkout "$rev"
    fi
  '';

  letsql-ensure-download-data = pkgs.writeShellScriptBin "letsql-ensure-download-data" ''
    repo_dir=$(realpath $(git rev-parse --git-dir)/..)
    if [ ! -d "$repo_dir/ci/ibis-testing-data" ]; then
      ${letsql-download-data}/bin/letsql-download-data
    fi
  '';

  letsql-docker-compose-up = pkgs.writeShellScriptBin "letsql-docker-compose-up" ''
    set -eux

    backends=''${@}
    docker compose up --build --wait ''${backends[@]}
  '';

  letsql-commands = {
    inherit letsql-pytest letsql-clean letsql-fmt letsql-lint letsql-ensure-download-data letsql-docker-compose-up;
  };
  letsql-commands-star = pkgs.buildEnv {
    name = "letsql-commands-star";
    paths = builtins.attrValues letsql-commands-star;
  };
in {
  inherit letsql-commands letsql-commands-star;
}
