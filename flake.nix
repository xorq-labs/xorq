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
        #
        mkShellApp = drv: name: bin-name: pkgs.writeShellApplication {
          inherit name;
          runtimeInputs = [ ];
          text = pkgs.lib.optionalString pkgs.stdenv.isDarwin ''
            export PINS_DATA_DIR="$HOME/Library/Application Support/pins-py"
            export PINS_CACHE_DIR=$HOME/Library/Caches/pins-py
          '' + ''
            bash <(
              cat <(grep ^declare ${drv} | grep -v "SSL_CERT_FILE=") <(echo ${bin-name} "''${@}")
            )
          '';
        };
        letsql-ipython-310 = mkShellApp dev310 "letsql-ipython-310" "ipython";
        letsql-ipython-311 = mkShellApp dev311 "letsql-ipython-311" "ipython";
        letsql-ipython-312 = mkShellApp dev312 "letsql-ipython-312" "ipython";
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
          inherit letsql-ipython-310 letsql-ipython-311 letsql-ipython-312;
          #
          app = self.packages.${system}.app310;
          appFromWheel = self.packages.${system}.appFromWheel310;
          toolsPackages = self.packages.${system}.toolsPackages310;
          letsql = self.packages.${system}.letsql-ipython-310;
          #
          default = self.packages.${system}.letsql;
        };
        lib = {
          inherit (letsql310) poetryOverrides maturinOverride toolchain;
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
