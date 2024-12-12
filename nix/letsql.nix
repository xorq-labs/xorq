{
  system,
  pkgs,
  pyproject-nix,
  uv2nix,
  pyproject-build-systems,
  crane,
  src,
}:
let
  mkLETSQL =
    python:
    let
      inherit (pkgs.lib) nameValuePair;
      inherit (pkgs.lib.path) append;
      compose = pkgs.lib.trivial.flip pkgs.lib.trivial.pipe;

      injectProjectVersion =
        version: project:
        pkgs.lib.attrsets.updateManyAttrsByPath [
          {
            path = [
              "pyproject"
              "project"
            ];
            update = old: { inherit version; } // old;
          }
        ] project;
      addNativeBuildInputs =
        drvs:
        (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ drvs;
        });
      addResolved =
        final: names:
        (old: {
          nativeBuildInputs =
            (old.nativeBuildInputs or [ ])
            ++ final.resolveBuildSystem (
              pkgs.lib.listToAttrs (map (name: pkgs.lib.nameValuePair name [ ]) names)
            );
        });
      mkEditableScopeOverride =
        final: project: root:
        final.callPackage (
          {
            python,
            stdenv,
            pyprojectHook,
            resolveBuildSystem,
            pythonPkgsBuildHost,
          }:
          stdenv.mkDerivation (
            pyproject-nix.build.lib.renderers.mkDerivationEditable
              {
                inherit project root;
                environ = pyproject-nix.lib.pep508.mkEnviron python;
              }
              {
                inherit
                  python
                  pyprojectHook
                  resolveBuildSystem
                  pythonPkgsBuildHost
                  ;
              }
          )
        ) { };

      toolchain = pkgs.rust-bin.fromRustupToolchainFile (append src "rust-toolchain.toml");
      mkCrateWheelSrc = import ./crate-wheel.nix {
        inherit
          system
          pkgs
          crane
          src
          toolchain
          ;
      };
      crateWheelSrc = mkCrateWheelSrc { inherit python; };
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = src; };
      overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };
      letsqlEditableOverride = final: _prev: {
        # necessary because uv2nix wants to read project for itself and we can't inject version there
        letsql = let
          project = injectProjectVersion "0.1.10" (
            pyproject-nix.lib.project.loadUVPyproject { projectRoot = src; }
          );
        in mkEditableScopeOverride final project "$REPO_ROOT/python";
      };
      letsqlCrateWheelSrcOverride = old: {
        src = crateWheelSrc;
        format = "wheel";
        nativeBuildInputs =
          (builtins.filter
            # all the hooks have the same name and we fail if we have the previous one
            (drv: drv.name != "pyproject-hook")
            (old.nativeBuildInputs or [ ])
          )
          ++ [ pythonSet.pyprojectWheelHook ];
      };

      darwinPyprojectOverrides = final: prev: {
        scipy = prev.scipy.overrideAttrs (compose [
          (addResolved final [
            "meson-python"
            "ninja"
            "cython"
            "numpy"
            "pybind11"
            "pythran"
          ])
          (addNativeBuildInputs [
            pkgs.gfortran
            pkgs.cmake
            pkgs.xsimd
            pkgs.pkg-config
            pkgs.openblas
            pkgs.meson
          ])
        ]);
        xgboost = prev.xgboost.overrideAttrs (compose [
          (addNativeBuildInputs [ pkgs.cmake ])
          (addResolved final [ "hatchling" ])
        ]);
        scikit-learn = prev.scikit-learn.overrideAttrs (
          addResolved final [
            "meson-python"
            "ninja"
            "cython"
            "numpy"
            "scipy"
          ]
        );
        duckdb = prev.duckdb.overrideAttrs (addNativeBuildInputs [
          prev.setuptools
          prev.pybind11
        ]);
        pyarrow = prev.pyarrow.overrideAttrs (addNativeBuildInputs [
          prev.setuptools
          prev.cython
          pkgs.cmake
          prev.numpy
          pkgs.pkg-config
          pkgs.arrow-cpp
        ]);
        google-crc32c = prev.google-crc32c.overrideAttrs (addNativeBuildInputs [ prev.setuptools ]);
        psycopg2-binary = prev.psycopg2-binary.overrideAttrs (addNativeBuildInputs [
          prev.setuptools
          pkgs.postgresql
          pkgs.openssl
        ]);
      };
      pyprojectOverrides = final: prev: {
        letsql = prev.letsql.overrideAttrs letsqlCrateWheelSrcOverride;
        cityhash = prev.cityhash.overrideAttrs (
          addResolved final (if python.pythonAtLeast "3.12" then [ "setuptools" ] else [ ])
        );
      };
      pythonSet =
        # Use base package set from pyproject.nix builders
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (
            pkgs.lib.composeManyExtensions (
              [
                pyproject-build-systems.overlays.default
                overlay
                pyprojectOverrides
              ]
              ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [ darwinPyprojectOverrides ]
            )
          );

      # # issue is that this doesn't exist: project.pyproject.project.version
      # virtualenv-editable =
      #   let
      #   editableOverlay = workspace.mkEditablePyprojectOverlay {
      #     root = "$REPO_ROOT";
      #   };
      #   editablePythonSet = pythonSet.overrideScope editableOverlay;
      #   in editablePythonSet.mkVirtualEnv "letsql-dev-env" workspace.deps.all;
      virtualenv-editable = (pythonSet.overrideScope letsqlEditableOverride).mkVirtualEnv "letsql-dev-env" workspace.deps.all;
      virtualenv = pythonSet.mkVirtualEnv "letsql-env" workspace.deps.all;

      editableShellHook = ''
        # Undo dependency propagation by nixpkgs.
        unset PYTHONPATH
        # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
        export REPO_ROOT=$(git rev-parse --show-toplevel)
      '';
      maybeMaturinBuildHook = ''
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
        source=$repo_dir/target/release/maturin/libletsql.$suffix
        target=$repo_dir/python/letsql/_internal.abi3.so
        if [ ! -e "$source" ]; then
          maturin build --release
        fi
        if [ ! -L "$target" -o "$(realpath "$source")" != "$(realpath "$target")" ]; then
          rm -f "$target"
          ln -s "$source" "$target"
        fi
      '';

      inherit
        (import ./commands.nix {
          inherit pkgs;
          python = virtualenv-editable;
        })
        letsql-commands-star
        ;
      toolsPackages = [
        pkgs.uv
        toolchain
        letsql-commands-star
      ];
      shell = pkgs.mkShell {
        packages = [
          virtualenv
        ] ++ toolsPackages;
        shellHook = ''
          unset PYTHONPATH
        '';
      };
      editableShell = pkgs.mkShell {
        packages = [
          virtualenv-editable
        ] ++ toolsPackages;
        shellHook = pkgs.lib.strings.concatStrings [
          editableShellHook
          "\n"
          maybeMaturinBuildHook
        ];
      };

    in
    {
      inherit
        editableShellHook
        maybeMaturinBuildHook
        crateWheelSrc
        virtualenv
        virtualenv-editable
        toolchain
        letsql-commands-star
        toolsPackages
        shell
        editableShell
        letsqlCrateWheelSrcOverride
        ;
    };
in
mkLETSQL
