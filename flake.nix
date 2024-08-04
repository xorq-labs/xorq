{
  description = "Application packaged using poetry2nix";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    rust-overlay.url = "github:oxalica/rust-overlay";
    crane = {
      url = "github:ipetkov/crane";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix, rust-overlay, crane }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            (import rust-overlay)
            poetry2nix.overlays.default
          ];
        };
        mkLETSQL = (import ./nix/letsql.nix { inherit system pkgs poetry2nix crane; }) ./.;
        mkCommands = python: import ./nix/commands.nix {
          inherit pkgs python;
        };
        mkShellHook = python: let
          commands = mkCommands python;
        in ''
          export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
          ${commands.letsql-commands.letsql-ensure-download-data}/bin/letsql-ensure-download-data
        '';
        mkToolsPackages = python: let
          commands = mkCommands python;
          letsql = mkLETSQL python;
        in pkgs.buildEnv {
          name = "tools-${python.pythonVersion}";
          paths = [
            letsql.toolchain
            pkgs.poetry
            python
            commands.letsql-commands-star
          ];
        };
        mkToolsShell = python: let
          toolsPackages = mkToolsPackages python;
          shellHook = mkShellHook python;
        in pkgs.mkShell {
          packages = [
            toolsPackages
            pkgs.maturin
          ];
          inherit shellHook;
        };
        mkDevShell = python: let
          toolsPackages = mkToolsPackages python;
          letsql = mkLETSQL python;
          shellHook = mkShellHook python;
        in pkgs.mkShell {
          packages = [
            letsql.appFromWheel
            toolsPackages
          ];
          inherit shellHook;
        };
        letsql310 = mkLETSQL pkgs.python310;
        letsql311 = mkLETSQL pkgs.python311;
        letsql312 = mkLETSQL pkgs.python312;
        toolsPackages310 = mkToolsPackages pkgs.python310;
        toolsPackages311 = mkToolsPackages pkgs.python311;
        toolsPackages312 = mkToolsPackages pkgs.python312;
        tools310 = mkToolsShell pkgs.python310;
        tools311 = mkToolsShell pkgs.python311;
        tools312 = mkToolsShell pkgs.python312;
        dev310 = mkDevShell pkgs.python310;
        dev311 = mkDevShell pkgs.python311;
        dev312 = mkDevShell pkgs.python312;
      in
      {
        packages = {
          app310 = letsql310.app;
          appFromWheel310 = letsql310.appFromWheel;
          app311 = letsql311.app;
          appFromWheel311 = letsql311.appFromWheel;
          app312 = letsql312.app;
          appFromWheel312 = letsql312.appFromWheel;
          inherit toolsPackages310 toolsPackages311 toolsPackages312;
          #
          app = self.packages.${system}.app310;
          appFromWheel = self.packages.${system}.appFromWheel310;
          toolsPackages = self.packages.${system}.toolsPackages310;
          default = self.packages.${system}.appFromWheel;
        };
        lib = {
          inherit (letsql310) poetryOverrides maturinOverride;
          inherit mkLETSQL mkCommands mkShellHook mkToolsPackages mkDevShell;
          inherit pkgs;
        };
        devShells = {
          inherit tools310 tools311 tools312;
          tools = self.devShells.${system}.tools310;
          inherit dev310 dev311 dev312;
          dev = self.devShells.${system}.dev310;
          default = self.devShells.${system}.dev;
        };
      });
}
