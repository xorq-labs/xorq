let
  flake = builtins.getFlake (toString ./.);
  inherit (flake.lib.${builtins.currentSystem}) pkgs mkToolsPackages poetryOverrides;
  inherit (flake.inputs.poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryEnv overrides;

  python = pkgs.python310;

  editableApp = mkPoetryEnv {
    projectDir = ./.;
    inherit python;
    preferWheels = true;
    groups = [ "dev" "test" "docs" ];
    editablePackageSources = {
      letsql = ./python;
    };
    overrides = overrides.withDefaults poetryOverrides;
  };
  toolsPackages = mkToolsPackages editableApp;
  shellHook = ''
    set -eu

    repo_dir=$(git rev-parse --show-toplevel)
    if [ "$(basename "$repo_dir")" != "letsql" ]; then
      echo "not in letsql, exiting"
      exit 1
    fi
    case $(uname) in
      Darwin) suffix=dylib ;;
      *)      suffix=so    ;;
    esac
    source=$repo_dir/target/debug/maturin/libletsql.$suffix
    target=$repo_dir/python/letsql/_internal.abi3.so
    if [ ! -e "$source" ]; then
      ${toolsPackages}/bin/maturin build
    fi
    if [ ! -L "$target" -o "$(realpath "$source")" != "$(realpath "$target")" ]; then
      rm -f "$target"
      ln -s "$source" "$target"
    fi
  '';
in pkgs.mkShell {
  packages = [
    editableApp
    toolsPackages
    pkgs.poetry
  ] ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
    pkgs.libiconv
  ];
  inherit shellHook;
}
