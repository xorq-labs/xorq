{ pkgs, python }:
let

  cachix-cache = "xorq-labs";

  xorq-cachix-use = pkgs.writeShellScriptBin "xorq-cachix-use" ''
    ${pkgs.cachix}/bin/cachix use ${cachix-cache}
  '';

  xorq-cachix-push = pkgs.writeShellScriptBin "xorq-cachix-push" ''
    ${pkgs.nix}/bin/nix build .#devShells.x86_64-linux.default --print-out-paths --no-link | \
      ${pkgs.cachix}/bin/cachix push ${cachix-cache}
  '';

  xorq-fmt = pkgs.writeShellScriptBin "xorq-fmt" ''
    set -eux

    ${python}/bin/python -m black .
    ${python}/bin/python -m blackdoc .
    ${python}/bin/python -m ruff --fix .
  '';

  xorq-lint = pkgs.writeShellScriptBin "xorq-lint" ''
    set -eux

    ${python}/bin/python -m black --quiet --check .
    ${python}/bin/python -m ruff .
  '';

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

  xorq-git-config-blame-ignore-revs = pkgs.writeShellScriptBin "xorq-git-config-blame-ignore-revs" ''
    set -eux

    # https://black.readthedocs.io/en/stable/guides/introducing_black_to_your_project.html#avoiding-ruining-git-blame
    ignore_revs_file=''${1:-.git-blame-ignore-revs}
    ${pkgs.git}/bin/git config blame.ignoreRevsFile "$ignore_revs_file"
  '';

  xorq-download-data = pkgs.writeShellScriptBin "xorq-download-data" ''
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

  xorq-ensure-download-data = pkgs.writeShellScriptBin "xorq-ensure-download-data" ''
    git_dir=$(git rev-parse --git-dir 2>/dev/null) || exit
    repo_dir=$(realpath "$git_dir/..")
    if [ "$(dirname "$repo_dir")" = "xorq" ] && [ ! -d "$repo_dir/ci/ibis-testing-data" ]; then
      ${xorq-download-data}/bin/xorq-download-data
    fi
  '';

  xorq-install-docker = pkgs.writeShellScriptBin "xorq-install-docker" ''
    set -eux

    # https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository
    # Add Docker's official GPG key:
    sudo apt-get update
    sudo apt-get install ca-certificates
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo ${pkgs.curl}/bin/curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # Add the repository to Apt sources:
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "''${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update

    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  '';

  xorq-docker-compose-up = pkgs.writeShellScriptBin "xorq-docker-compose-up" ''
    set -eux

    backends=''${@}
    ${pkgs.docker-compose}/bin/docker-compose up --build --wait ''${backends[@]}
  '';

  xorq-newgrp-docker-compose-up = pkgs.writeShellScriptBin "xorq-newgrp-docker-compose-up" ''
    set -eux

    newgrp docker <<<"${xorq-docker-compose-up}/bin/xorq-docker-compose-up ''${@}"
  '';

  xorq-docker-run-otel-collector = pkgs.writeShellScriptBin "xorq-docker-run-otel-collector" ''
    set -eux

    image_name=otel/opentelemetry-collector-contrib:latest
    yaml_host_path=${../docker/otel/otel-collector-config.yaml}
    yaml_container_path=/etc/otel-collector-config.yaml

    logs_host_path=''${OTEL_HOST_LOG_DIR:-~/.local/share/xorq/logs/otel-logs}
    logs_container_path=''${OTEL_COLLECTOR_CONTAINER_LOG_DIR:-/otel-logs}

    mkdir --mode=777 --parents "$logs_host_path"

    ${pkgs.docker}/bin/docker run \
      --publish "$OTEL_COLLECTOR_PORT_GRPC:$OTEL_COLLECTOR_PORT_GRPC" \
      --publish "$OTEL_COLLECTOR_PORT_HTTP:$OTEL_COLLECTOR_PORT_HTTP" \
      --env "GRAFANA_CLOUD_INSTANCE_ID=$GRAFANA_CLOUD_INSTANCE_ID" \
      --env "GRAFANA_CLOUD_API_KEY=$GRAFANA_CLOUD_API_KEY" \
      --env "GRAFANA_CLOUD_OTLP_ENDPOINT=$GRAFANA_CLOUD_OTLP_ENDPOINT" \
      --env "OTEL_COLLECTOR_PORT_GRPC=$OTEL_COLLECTOR_PORT_GRPC" \
      --env "OTEL_COLLECTOR_PORT_HTTP=$OTEL_COLLECTOR_PORT_HTTP" \
      --env "OTEL_LOG_FILE_NAME=$OTEL_LOG_FILE_NAME" \
      --env "OTEL_COLLECTOR_CONTAINER_LOG_DIR=$OTEL_COLLECTOR_CONTAINER_LOG_DIR" \
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

  xorq-commands = {
    inherit
      xorq-cachix-use xorq-cachix-push
      xorq-fmt
      xorq-lint
      xorq-kill-lsof-grep-port
      xorq-gh-config-set-browser-false
      xorq-git-config-blame-ignore-revs
      xorq-ensure-download-data
      xorq-install-docker xorq-docker-compose-up xorq-newgrp-docker-compose-up
      xorq-docker-run-otel-collector xorq-docker-exec-otel-print-initial-config
      ;
  };

  xorq-commands-star = pkgs.buildEnv {
    name = "xorq-commands-star";
    paths = builtins.attrValues xorq-commands;
  };
in
{
  inherit xorq-commands xorq-commands-star;
}
