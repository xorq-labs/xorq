{ system, pkgs, poetry2nix, crane }:
let
  mkLETSQL = src: python: with pkgs.lib.path; let
    # how do we ensure pkgs comes with rust-overlay overlay?
    cargoToml = append src "Cargo.toml";
    rustSrcSet = with pkgs.lib.fileset; unions [
      cargoToml
      (append src "Cargo.lock")
      (append src "pyproject.toml")
      (append src "README.md")
      (append src "LICENSE")
      (fileFilter (file: file.hasExt "rs") (append src "src"))
    ];
    pySrcSet = with pkgs.lib.fileset; unions [
      rustSrcSet
      (append src "poetry.lock")
      (fileFilter (file: file.hasExt "py") (append src "python"))
      (fileFilter (file: file.hasExt "sql") (append src "python"))
    ];
    rustSrc = with pkgs.lib.fileset; toSource {
      root = src;
      fileset = rustSrcSet;
    };
    pySrc = with pkgs.lib.fileset; toSource {
      root = src;
      fileset = unions [ rustSrcSet pySrcSet ];
    };
    toolchain = pkgs.rust-bin.fromRustupToolchainFile (append src "rust-toolchain.toml");
    craneLib = (crane.mkLib pkgs).overrideToolchain toolchain;
    wheelName = let
      inherit (craneLib.crateNameFromCargoToml { inherit cargoToml; }) pname version;
      wheelTail = {
        x86_64-linux = "linux_x86_64";
        aarch64-darwin = "macosx_11_0_arm64";
        # aarch64-linux = "";
        # x86_64-darwin = "";
      }.${system};
    in "${pname}-${version}-cp38-abi3-${wheelTail}.whl";
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
        python
      ];
      buildInputs = pkgs.lib.optionals pkgs.stdenv.isDarwin [
        pkgs.libiconv
        python
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
    crateWheelSrc = "${crateWheel}/${wheelName}";

    inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryApplication overrides;
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
        lockFile = "${old.src}/Cargo.lock";
        outputHashes = {
          "gbdt-0.1.3" = "sha256-oEiCWgbu9FBJt+5ceGTt/OCqbx1UI24dXOUEFy0hZoU=";
        };
      };
      nativeBuildInputs = (old.nativeBuildInputs or []) ++ [
        cargoSetupHook
        maturinBuildHook
      ];
    };
    commonPoetryArgs = {
      projectDir = src;
      preferWheels = true;
      inherit python;
      groups = [ "dev" "test" "docs" ];
    };
    app = (mkPoetryApplication (commonPoetryArgs // {
      buildInputs = pkgs.lib.optionals pkgs.stdenv.isDarwin [
        pkgs.libiconv
      ];
    })).overridePythonAttrs maturinOverride;
    appFromWheel = (mkPoetryApplication (commonPoetryArgs // {
      src = crateWheelSrc;
      overrides = overrides.withDefaults poetryOverrides;
    })).override {
      format = "wheel";
    };
  in {
    inherit toolchain;
    inherit rustSrcSet pySrcSet rustSrc pySrc crateWheelSrc;
    inherit poetryOverrides maturinOverride;
    inherit app appFromWheel;
  };
in mkLETSQL
