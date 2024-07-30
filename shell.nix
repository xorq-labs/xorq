let
  flake = builtins.getFlake (toString ./.);
  inherit (flake.lib.${builtins.currentSystem}) pkgs mkToolsPackages;
  inherit (flake.inputs.poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryEnv;

  python = pkgs.python310;

  editableApp = mkPoetryEnv {
    projectDir = ./.;
    inherit python;
    preferWheels = true;
    groups = [ "dev" "test" "docs" ];
    editablePackageSources = {
      letsql = ./python;
    };
  };
  toolsPackages = mkToolsPackages editableApp;
in pkgs.mkShell {
  packages = [
    editableApp
    toolsPackages
    pkgs.poetry
  ];
  shellHook = ''
    repo_dir=$(git rev-parse --show-toplevel)
    if [ "$(basename "$repo_dir")" != "letsql" ]; then
      echo "not in letsql, exiting"
      exit 1
    fi
    source=$repo_dir/target/debug/maturin/libletsql.so
    target=$repo_dir/python/letsql/_internal.abi3.so
    if [ ! -f "$target" ]; then
      maturin build
      ln -s "$source" "$target"
    fi
    if [ "$(realpath "$source")" != "$(realpath "$target")" ]; then
      rm "$target"
      ln -s "$source" "$target"
    fi
  '';
}
