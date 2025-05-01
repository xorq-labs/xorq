{
  system,
  pkgs,
  crane,
  src,
  toolchain,
}:
let
  mkCrateWheelSrc =
    { python }:
    let
      inherit (pkgs.lib.path) append;
      inherit (pkgs.lib.fileset) unions toSource fileFilter;
      cargoToml = append src "Cargo.toml";
      rustSrcSet = unions [
        cargoToml
        (append src "Cargo.lock")
        (append src "pyproject.toml")
        (append src "README.md")
        (append src "LICENSE")
        (append src "python/xorq/internal.py")
        (append src "python/xorq/env_templates")
        (fileFilter (file: file.hasExt "rs") (append src "src"))
      ];
      pySrcSet = unions [
        rustSrcSet
        (fileFilter (file: file.hasExt "py") (append src "python"))
        (fileFilter (file: file.hasExt "sql") (append src "python"))
        (append src "python/xorq/env_templates")
      ];
      rustSrc = toSource {
        root = src;
        fileset = rustSrcSet;
      };
      pySrc = toSource {
        root = src;
        fileset = unions [
          rustSrcSet
          pySrcSet
        ];
      };
      craneLib = (crane.mkLib pkgs).overrideToolchain (_: toolchain);
      wheelName =
        let
          inherit (craneLib.crateNameFromCargoToml { inherit cargoToml; }) pname version;
          wheelTail =
            {
              x86_64-linux = "linux_x86_64";
              aarch64-darwin = "macosx_11_0_arm64";
              # aarch64-linux = "";
              # x86_64-darwin = "";
            }
            .${system};
        in
        "${pname}-${version}-cp38-abi3-${wheelTail}.whl";
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
        nativeBuildInputs =
          [
            python
            pkgs.pkg-config
          ]
          ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.darwin.apple_sdk.frameworks.Security
          ];
        buildInputs = pkgs.lib.optionals pkgs.stdenv.isDarwin [
          pkgs.libiconv
          python
        ];
        PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
      };
      crateWheelDeps = craneLib.buildPackage (
        commonCraneArgs
        // {
          pname = "crateWheel-deps";
          src = rustSrc;
          doNotPostBuildInstallCargoBinaries = true;
          inherit buildPhaseCargoCommand;
          doInstallCargoArtifacts = true;
          installPhaseCommand = "mkdir -p $out";
        }
      );
      crateWheel = craneLib.buildPackage (
        commonCraneArgs
        // {
          cargoArtifacts = crateWheelDeps;
          src = pySrc;
          doNotPostBuildInstallCargoBinaries = true;
          inherit buildPhaseCargoCommand;
          installPhaseCommand = ''
            ls target/wheels/*
            mkdir -p $out
            cp target/wheels/*.whl $out/
          '';
        }
      );
      crateWheelSrc = "${crateWheel}/${wheelName}";
    in
    crateWheelSrc;
  usePyprojectWheelHook =
    old: pythonSet:
    (builtins.filter
      # all the hooks have the same name and we fail if we have the previous one
      (drv: drv.name != "pyproject-hook")
      (old.nativeBuildInputs or [ ])
    )
    ++ [ pythonSet.pyprojectWheelHook ];
  mkLetsqlCrateWheelSrcOverride = python: pythonSet: old: {
    src = mkCrateWheelSrc { inherit python; };
    format = "wheel";
    nativeBuildInputs = usePyprojectWheelHook old pythonSet;
  };
  mkPyprojectOverrides-wheel = python: pythonSet: final: prev: {
    xorq = prev.xorq.overrideAttrs (mkLetsqlCrateWheelSrcOverride python pythonSet);
  };
in
{
  inherit
    mkCrateWheelSrc
    mkLetsqlCrateWheelSrcOverride
    mkPyprojectOverrides-wheel
    usePyprojectWheelHook
    ;
}
