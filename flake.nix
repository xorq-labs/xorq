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

        toolchainFile = ./rust-toolchain.toml;
        cargoToml = ./Cargo.toml;
        lockFile = ./Cargo.lock;
        pythonVersion = "310";
        rustSrc = with pkgs.lib.fileset; toSource {
          root = ./.;
          fileset = unions [
            cargoToml
            lockFile
            ./pyproject.toml
            ./README.md
            ./LICENSE
            (fileFilter (file: file.hasExt "rs") ./src)
          ];
        };
        pySrc = with pkgs.lib.fileset; toSource {
          root = ./.;
          fileset = unions [
            cargoToml
            lockFile
            ./pyproject.toml
            ./poetry.lock
            ./README.md
            ./LICENSE
            (fileFilter (file: file.hasExt "py") ./python)
            (fileFilter (file: file.hasExt "sql") ./python)
            (fileFilter (file: file.hasExt "rs") ./src)
          ];
        };

        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };
        inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryApplication overrides;
        toolchain = pkgs.rust-bin.fromRustupToolchainFile toolchainFile;
        craneLib = (crane.mkLib pkgs).overrideToolchain toolchain;
        python' = pkgs."python${pythonVersion}";
        wheelName = let
          inherit (craneLib.crateNameFromCargoToml { inherit cargoToml; }) pname version;
          wheelTail = {
            x86_64-linux = "linux_x86_64";
            aarch64-darwin = "macosx_11_0_arm64";
            # aarch64-linux = "";
            # x86_64-darwin = "";
          }.${system};
        in "${pname}-${version}-cp38-abi3-${wheelTail}.whl";

        commands = let
          get-first-pname-drv = pname: builtins.elemAt (builtins.filter (drv: drv.pname == pname) myappFromWheel.requiredPythonModules) 0;
          black = get-first-pname-drv "black";
          blackdoc = get-first-pname-drv "blackdoc";
          ruff = get-first-pname-drv "ruff";
        in import ./nix/commands.nix {
          inherit pkgs black blackdoc ruff;
          inherit (myappFromWheel) python;
        };
        inherit (commands) letsql-commands;

        poetryOverrides = final: prev: {
          ibis-framework = prev.ibis-framework.overridePythonAttrs (old: {
            buildInputs = (old.buildInputs or [ ]) ++ [ prev.poetry-dynamic-versioning ];
          });
          xgboost = prev.xgboost.overridePythonAttrs (old: {
          } // pkgs.lib.attrsets.optionalAttrs pkgs.stdenv.isDarwin {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ prev.cmake ];
            cmakeDir = "../cpp_src";
            preBuild = ''
              cd ..
            '';
          });
        };
        maturinOverride = old: with pkgs.rustPlatform; {
          cargoDeps = importCargoLock {
            inherit lockFile;
          };
          nativeBuildInputs = (old.nativeBuildInputs or []) ++ [
            cargoSetupHook
            maturinBuildHook
          ];
        };
        buildPhaseCargoCommand = ''
          ${pkgs.maturin}/bin/maturin build \
            --offline \
            --target-dir target \
            --manylinux off \
            --strip \
            --release
        '';

        commonCraneArgs = {
          src = rustSrc;
          strictDeps = true;
          nativeBuildInputs = [
            python'
          ];
          buildInputs = pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.libiconv
            python'
          ];
        };
        crateWheelDeps = craneLib.buildPackage (commonCraneArgs // {
          pname = "crateWheel-deps";
          src = rustSrc;
          doInstallCargoArtifacts = true;
          inherit buildPhaseCargoCommand;
          installPhaseCommand = "mkdir -p $out";
        });
        crateWheel = craneLib.buildPackage (commonCraneArgs // {
          cargoArtifacts = crateWheelDeps;
          src = pySrc;
          inherit buildPhaseCargoCommand;
          installPhaseCommand = ''
            ls target/wheels/*
            mkdir -p $out
            cp target/wheels/*.whl $out/
          '';
        });

        commonPoetryArgs = {
          projectDir = ./.;
          src = pySrc;
          preferWheels = true;
          python = python';
          groups = [ "dev" "test" "docs" ];
        };
        myapp = (mkPoetryApplication (commonPoetryArgs // {
          buildInputs = pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.libiconv
          ];
        })).overridePythonAttrs maturinOverride;
        myappFromWheel = (mkPoetryApplication (commonPoetryArgs // {
          src = "${crateWheel}/${wheelName}";
          overrides = overrides.withDefaults poetryOverrides;
        })).override (_old: {
          format = "wheel";
        });

        toolsPackages = pkgs.buildEnv {
          name = "tools";
          paths = [
            toolchain
            pkgs.maturin
            pkgs.poetry
            python'
          ] ++ (builtins.attrValues letsql-commands);
        };
        shellHook = ''
          export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
          ${letsql-commands.letsql-ensure-download-data}/bin/letsql-ensure-download-data
        '';

      in
      {
        packages = {
          inherit crateWheelDeps crateWheel myapp myappFromWheel toolsPackages;
          default = self.packages.${system}.myapp;
        };

        devShells = {
          dev = pkgs.mkShell {
            packages = [
              self.packages.${system}.myapp
              toolsPackages
            ];
          };
          devFromWheel = pkgs.mkShell {
            packages = [
              self.packages.${system}.myappFromWheel
              toolsPackages
            ];
            inherit shellHook;
          };
          inputs = pkgs.mkShell {
            inputsFrom = [ self.packages.${system}.myapp ];
            packages = toolsPackages;
          };
          tools = pkgs.mkShell {
            packages = [
              toolsPackages
            ];
          };
          default = self.devShells.${system}.devFromWheel;
        };
      });
}
