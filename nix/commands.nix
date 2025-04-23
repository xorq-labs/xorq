{ pkgs, python }:
let

  xorq-kill-lsof-grep-port = pkgs.writeShellScriptBin "xorq-kill-lsof-grep-port" ''
    set -eux

    port=$1
    pids=($(lsof -i4@localhost | grep "$port" | awk '{print $2}'))
    if [ "''${#pids[@]}" -ne "0" ]; then
      kill "''${pids[@]}"
    fi
  '';

  xorq-gh-config-set-browser-false = pkgs.writeShellScriptBin "xorq-gh-config-set-browser-false" ''
    ${pkgs.gh}/bin/gh config set browser false
  '';

  letsql-pytest = pkgs.writeShellScriptBin "letsql-pytest" ''
    set -eux

    # see https://docs.pytest.org/en/latest/explanation/pythonpath.html#import-mode-importlib
    required_dir="$(git rev-parse --show-toplevel)/python/xorq"

    case $PWD/ in
      "$required_dir"*) true;;
      *) echo "must run from inside $required_dir"; exit 1
    esac

    ${python}/bin/python -m pytest --import-mode=importlib "''${@}"
  '';

  letsql-fmt = pkgs.writeShellScriptBin "letsql-fmt" ''
    set -eux

    ${python}/bin/python -m black .
    ${python}/bin/python -m blackdoc .
    ${python}/bin/python -m ruff --fix .
  '';

  letsql-lint = pkgs.writeShellScriptBin "letsql-lint" ''
    set -eux

    ${python}/bin/python -m black --quiet --check .
    ${python}/bin/python -m ruff .
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
    git_dir=$(git rev-parse --git-dir 2>/dev/null) || exit
    repo_dir=$(realpath "$git_dir/..")
    if [ "$(dirname "$repo_dir")" = "xorq" ] && [ ! -d "$repo_dir/ci/ibis-testing-data" ]; then
      ${letsql-download-data}/bin/letsql-download-data
    fi
  '';

  letsql-docker-compose-up = pkgs.writeShellScriptBin "letsql-docker-compose-up" ''
    set -eux

    backends=''${@}
    ${pkgs.docker-compose}/bin/docker-compose up --build --wait ''${backends[@]}
  '';

  letsql-newgrp-docker-compose-up = pkgs.writeShellScriptBin "letsql-newgrp-docker-compose-up" ''
    set -eux

    newgrp docker <<<"${letsql-docker-compose-up}/bin/letsql-docker-compose-up ''${@}"
  '';

  letsql-git-fetch-origin-pull = pkgs.writeShellScriptBin "letsql-git-fetch-origin-pull" ''
    set -eux

    PR=$1
    branchname="origin-pr-$PR"
    git fetch origin pull/$PR/head:$branchname
  '';

  letsql-git-config-blame-ignore-revs = pkgs.writeShellScriptBin "letsql-git-config-blame-ignore-revs" ''
    set -eux

    # https://black.readthedocs.io/en/stable/guides/introducing_black_to_your_project.html#avoiding-ruining-git-blame
    ignore_revs_file=''${1:-.git-blame-ignore-revs}
    ${pkgs.git}/bin/git config blame.ignoreRevsFile "$ignore_revs_file"
  '';

  letsql-maturin-build = pkgs.writeShellScriptBin "letsql-maturin-build" ''
    set -eux
    repo_dir=$(git rev-parse --show-toplevel)
    cd "$repo_dir"
    ${python}/bin/maturin build --release
  '';

  xorq-docker-run-otel-collector = pkgs.writeShellScriptBin "xorq-docker-run-otel-collector" ''
    set -eux

    image_name=otel/opentelemetry-collector-contrib:latest
    yaml_host_path=${../docker/otel/otel-collector-config.yaml}
    yaml_container_path=/etc/otel-collector-config.yaml
    logs_host_path=~/.local/share/xorq/logs/otel-logs
    logs_container_path=/otel-logs

    mkdir --mode=777 --parents "$logs_host_path"

    ${pkgs.docker}/bin/docker run \
      --publish "$OTEL_COLLECTOR_PORT_GRPC:$OTEL_COLLECTOR_PORT_GRPC" \
      --publish "$OTEL_COLLECTOR_PORT_HTTP:$OTEL_COLLECTOR_PORT_HTTP" \
      --env "GRAFANA_CLOUD_INSTANCE_ID=$GRAFANA_CLOUD_INSTANCE_ID" \
      --env "GRAFANA_CLOUD_API_KEY=$GRAFANA_CLOUD_API_KEY" \
      --env "GRAFANA_CLOUD_OTLP_ENDPOINT=$GRAFANA_CLOUD_OTLP_ENDPOINT" \
      --env "OTEL_COLLECTOR_PORT_GRPC=$OTEL_COLLECTOR_PORT_GRPC" \
      --env "OTEL_COLLECTOR_PORT_HTTP=$OTEL_COLLECTOR_PORT_HTTP" \
      --env "logs_container_path=$logs_container_path" \
      --volume "$yaml_host_path:$yaml_container_path" \
      --volume "$logs_host_path:$logs_container_path" \
      "$image_name" \
      --config="$yaml_container_path"
  '';

  xorq-docker-exec-otel-print-initial-config = pkgs.writeShellScriptBin "xorq-docker-exec-otel-print-initial-config" ''
    set -eux

    container=$1
    yaml_container_path=/etc/otel-collector-config.yaml
    ${pkgs.docker}/bin/docker exec \
      --interactive --tty \
      "$container" \
      /otelcol-contrib print-initial-config \
      --config "$yaml_container_path" \
      --feature-gates otelcol.printInitialConfig
  '';

  letsql-commands = {
    inherit
      xorq-kill-lsof-grep-port
      letsql-pytest
      letsql-fmt
      letsql-lint
      letsql-ensure-download-data
      letsql-docker-compose-up
      letsql-newgrp-docker-compose-up
      letsql-git-fetch-origin-pull
      letsql-git-config-blame-ignore-revs
      letsql-maturin-build
      xorq-gh-config-set-browser-false
      xorq-docker-run-otel-collector xorq-docker-exec-otel-print-initial-config
      ;
  };

  letsql-commands-star = pkgs.buildEnv {
    name = "letsql-commands-star";
    paths = builtins.attrValues letsql-commands;
  };
in
{
  inherit letsql-commands letsql-commands-star;
}
