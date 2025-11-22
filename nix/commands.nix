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
    if [ "$(basename "$repo_dir")" = "xorq" ] && [ ! -d "$repo_dir/ci/ibis-testing-data" ]; then
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

  xorq-sudo-usermod-aG-docker = pkgs.writeShellScriptBin "xorq-sudo-usermod-aG-docker" ''
    set -eux

    user=''${1:-$USER}
    sudo usermod --append --groups docker "$user"
  '';

  xorq-colima-start = pkgs.writeShellScriptBin "xorq-colima-start" ''
    # ${pkgs.docker}
    ${pkgs.colima}/bin/colima start
  '';

  xorq-docker = pkgs.writeShellScriptBin "xorq-docker" ''
    set -eux

    ${pkgs.docker}/bin/docker "''${@}"
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

    yaml_file=otel-collector-config.yaml
    #
    yaml_container_path=/etc/otel-collector-config
    yaml_container_file=$yaml_container_path/$yaml_file
    # macos can't use the permissions on files in the nix store or its own default tmp dirs
    yaml_host_path=$(mktemp --directory -p $HOME)
    yaml_host_file=$yaml_host_path/$yaml_file

    cp ${../docker/otel/otel-collector-config.yaml} "$yaml_host_file"
    chmod 644 "$yaml_host_file"
    chmod 755 "$yaml_host_path"

    # Set up log directory paths
    logs_host_path=''${OTEL_HOST_LOG_DIR:-~/.local/share/xorq/logs/otel-logs}
    logs_container_path=''${OTEL_COLLECTOR_CONTAINER_LOG_DIR:-/otel-logs}

    mkdir --mode=777 --parents "$logs_host_path"

    ${pkgs.docker}/bin/docker run \
      --publish "$OTEL_COLLECTOR_PORT_GRPC:$OTEL_COLLECTOR_PORT_GRPC" \
      --publish "$OTEL_COLLECTOR_PORT_HTTP:$OTEL_COLLECTOR_PORT_HTTP" \
      --env "GRAFANA_CLOUD_INSTANCE_ID=$GRAFANA_CLOUD_INSTANCE_ID" \
      --env "PROMETHEUS_GRAFANA_USERNAME=$PROMETHEUS_GRAFANA_USERNAME" \
      --env "PROMETHEUS_GRAFANA_ENDPOINT=$PROMETHEUS_GRAFANA_ENDPOINT" \
      --env "PROMETHEUS_SCRAPE_URL=$PROMETHEUS_SCRAPE_URL" \
      --env "GRAFANA_CLOUD_API_KEY=$GRAFANA_CLOUD_API_KEY" \
      --env "GRAFANA_CLOUD_OTLP_ENDPOINT=$GRAFANA_CLOUD_OTLP_ENDPOINT" \
      --env "OTEL_COLLECTOR_PORT_GRPC=$OTEL_COLLECTOR_PORT_GRPC" \
      --env "OTEL_COLLECTOR_PORT_HTTP=$OTEL_COLLECTOR_PORT_HTTP" \
      --env "OTEL_LOG_FILE_NAME=$OTEL_LOG_FILE_NAME" \
      --env "OTEL_COLLECTOR_CONTAINER_LOG_DIR=$logs_container_path" \
      --volume "$yaml_host_path:$yaml_container_path" \
      --volume "$logs_host_path:$logs_container_path" \
      "$image_name" \
      --config="$yaml_container_file"
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

  xorq-psql = pkgs.writeShellScriptBin "xorq-psql" ''
    set -eux

    ${pkgs.postgresql}/bin/psql "''${@}"
  '';

  xorq-sops-ssh-to-age = pkgs.writeShellScriptBin "xorq-sops-ssh-to-age" ''
    set -eux

    private_key=''${1:-$HOME/.ssh/id_rsa}
    target=''${2:-$HOME/.config/sops/age/keys.txt}

    if [ ! -f "$private_key" ]; then
      echo "$private_key" does not exist
      exit 1
    fi
    mkdir -p "$(dirname "$target")" || {
      echo cannot mkdir for "$target"
      exit 1
    }

    ${pkgs.ssh-to-age}/bin/ssh-to-age \
      -private-key \
      -i "$private_key" \
      -o "$target"
  '';

  xorq-sops-ssh-to-age-public = pkgs.writeShellScriptBin "xorq-sops-ssh-to-age-public" ''
    set -eux

    public_key_path=''${1:-$HOME/.ssh/id_rsa.pub}
    target=''${2:--}


    if [ ! -f "$public_key_path" ]; then
      echo "$public_key_path" does not exist
      exit 1
    fi
    mkdir -p "$(dirname "$target")" || {
      echo cannot mkdir for "$target"
      exit 1
    }

    ${pkgs.ssh-to-age}/bin/ssh-to-age \
      -i "$public_key_path" \
      -o "$target"
  '';

  xorq-sops-add-age-key = pkgs.writeShellScriptBin "xorq-sops-add-age-key" ''
    set -eux

    public_key_path=''${1:-$HOME/.ssh/id_rsa.pub}
    sops_path=''${2:-$HOME/.sops.yaml}
    age_key=$(${xorq-sops-ssh-to-age-public}/bin/xorq-sops-ssh-to-age-public "$public_key_path" /dev/stdout)
    script=$(cat <<EOF
    from yaml import safe_dump, safe_load
    from pathlib import Path

    sops_path = Path("$sops_path")
    dct = safe_load(sops_path.read_text()) if sops_path.exists() else {}
    creation_rules = [{"age": "$age_key"}] + dct.get("creation_rules", [])
    dct = dct | {"creation_rules": creation_rules}
    sops_path.write_text(safe_dump(dct))
    EOF)
    ${python}/bin/python -c "$script"
  '';

  xorq-sops-encrypt-dotenv = pkgs.writeShellScriptBin "xorq-sops-encrypt-dotenv" ''
    set -eux

    path=$1
    target=''${2:-$path.sops-encrypted}
    ${pkgs.sops}/bin/sops encrypt --input-type dotenv --output-type dotenv $path >$target
  '';

  xorq-sops-decrypt-dotenv = pkgs.writeShellScriptBin "xorq-sops-decrypt-dotenv" ''
    set -eux
    path=$1
    ${pkgs.sops}/bin/sops decrypt --input-type dotenv --output-type dotenv $path
  '';

  xorq-sops-concatenate-dotenv = pkgs.writeShellScriptBin "xorq-sops-concatenate-dotenv" ''
    set -eux

    sops encrypt --input-type dotenv --output-type dotenv <(
      for file in "$@"; do
        ${pkgs.sops}/bin/sops decrypt --input-type dotenv --output-type dotenv "$file"
      done
    )
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
      xorq-install-docker xorq-sudo-usermod-aG-docker xorq-docker-compose-up xorq-newgrp-docker-compose-up
      xorq-colima-start
      xorq-docker xorq-docker-run-otel-collector xorq-docker-exec-otel-print-initial-config
      xorq-psql
      xorq-sops-encrypt-dotenv xorq-sops-decrypt-dotenv xorq-sops-concatenate-dotenv
      xorq-sops-ssh-to-age xorq-sops-ssh-to-age-public xorq-sops-add-age-key
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
