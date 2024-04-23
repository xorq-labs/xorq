with (import <nixpkgs> {});
let
  this-dir = toString ./.;
  venv-path = toString ./venv;
  python-bin = "${pkgs.python310}/bin/python";
  pb-ver = "25.1";
  protoc-target = "~/.local";
  cargo-target = "~/.cargo";

  ensure-cargo = pkgs.writeShellScriptBin "ensure-cargo" ''
    set -eux

    target=${cargo-target}
    if [ ! -d "$target" ]; then
      ${pkgs.bash}/bin/bash <(
        ${pkgs.curl}/bin/curl \
          --proto '=https' \
          --tlsv1.2 -sSf https://sh.rustup.rs \
        ) \
        --no-modify-path \
        -y

        if [ ! -d "$target" ]; then
          echo "$target doesn't exist after running rustup install"
          exit 1
        fi
    fi
  '';

  ensure-protoc = pkgs.writeShellScriptBin "ensure-protoc" ''
    set -eux

    pb_ver=''${1:-${pb-ver}}
    target=''${2:-${protoc-target}}

    echo $target
    if [ ! -f "$target/bin/protoc" ]; then
      zip_name="protoc-$pb_ver-linux-x86_64.zip"
      PB_REL="https://github.com/protocolbuffers/protobuf/releases"
      curl -LO "$PB_REL/download/v$pb_ver/$zip_name"
      unzip "$zip_name" -d "$target"
      rm "$zip_name"
    fi
    echo $target
  '';

  ensure-pre-commit-installed = pkgs.writeShellScriptBin "ensure-pre-commit-installed" ''
    test -x ${this-dir}/.git/hooks/pre-commit || {
      cd ${this-dir} && \
        ${pkgs.pre-commit}/bin/pre-commit install
    }
  '';

  make-venv = pkgs.writeShellScriptBin "make-venv" ''
    ${python-bin} -m venv --without-pip "${venv-path}"
    source ${venv-path}/bin/activate
    python <(${pkgs.curl}/bin/curl https://bootstrap.pypa.io/get-pip.py)
    pip install poetry
    cd ${this-dir}
    poetry check
    poetry install --no-root
  '';

  letsql-ensure-setup = pkgs.writeShellScriptBin "letsql-ensure-setup" ''
    set -eux
    ${ensure-cargo}/bin/ensure-cargo
    ${ensure-protoc}/bin/ensure-protoc
    ${ensure-pre-commit-installed}/bin/ensure-pre-commit-installed
    if [ ! -d ${venv-path} ]; then
      ${make-venv}/bin/make-venv
    fi
  '';

  letsql-maturin-develop = pkgs.writeShellScriptBin "letsql-maturin-develop" ''
    source ${venv-path}/bin/activate
    cd ${this-dir}
    maturin develop
  '';

  letsql-docker-compose-up = pkgs.writeShellScriptBin "letsql-docker-compose-up" ''
    set -eux
    docker compose up
    # docker compose up --build --force-recreate --no-deps
  '';

  letsql-docker-compose-down-volumes = pkgs.writeShellScriptBin "letsql-docker-compose-down-volumes" ''
    set -eux
    docker compose down --volumes
  '';

  letsql-sudo-usermod-append-docker = pkgs.writeShellScriptBin "letsql-usermod-append-docker" ''
    set -eux
    sudo usermod --append --groups docker $USER
  '';

  letsql-newgrp-docker = pkgs.writeShellScriptBin "letsql-newgrp-docker" ''
    set -eux
    newgrp docker
  '';

in pkgs.mkShell {
  shellHook = ''
    # work around poetry keyring issue
    export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
    export PATH=${protoc-target}/bin:${cargo-target}/bin:$PATH

    ${letsql-ensure-setup}/bin/letsql-ensure-setup
    source ${venv-path}/bin/activate
    python -c 'import letsql' >/dev/null 2>/dev/null || {
      # unclear why we can't run this inside shell hook, but it always fails on the first run
      echo "initial population required, run
        letsql-maturin-develop"
    }
  '';
  packages = with pkgs; [
    letsql-ensure-setup
    letsql-maturin-develop

    letsql-docker-compose-up
    letsql-docker-compose-down-volumes
    letsql-sudo-usermod-append-docker
    letsql-newgrp-docker

    pre-commit
    just
  ];
  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
}
