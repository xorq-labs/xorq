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
      toolchain = pkgs.rust-bin.fromRustupToolchainFile (append src "rust-toolchain.toml");
      crateWheelLib = import ./crate-wheel.nix {
        inherit
          system
          pkgs
          crane
          src
          toolchain
          ;
      };
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = src; };
      wheelOverlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };
      editableOverlay = workspace.mkEditablePyprojectOverlay {
        root = "$REPO_ROOT";
      };
      pyprojectOverrides-base = final: prev: {
        cityhash = prev.cityhash.overrideAttrs (
          addResolved final (if python.pythonAtLeast "3.12" then [ "setuptools" ] else [ ])
        );
      };
      pyprojectOverrides-wheel = crateWheelLib.mkPyprojectOverrides-wheel python pythonSet-base;
      pyprojectOverrides-editable = final: prev: {
        xorq = prev.xorq.overrideAttrs (old: {
          patches = (old.patches or [ ]) ++ [
            ./pyproject.build-system.diff
          ];
          nativeBuildInputs =
            (old.nativeBuildInputs or [ ])
            ++ final.resolveBuildSystem {
              setuptools = [ ];
            };
        });
      };
      pythonSet-base =
        # Use base package set from pyproject.nix builders
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (
            pkgs.lib.composeManyExtensions (
              [
                pyproject-build-systems.overlays.default
                wheelOverlay
                pyprojectOverrides-base
              ]
              ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [ darwinPyprojectOverrides ]
            )
          );
      pyprojectOverrides-pypi = final: prev: {
        xorq = prev.xorq.overrideAttrs (old: {
          src = pkgs.fetchurl {
            url = "https://files.pythonhosted.org/packages/82/7c/4f6fa35ac8bc4aeef68f94c91f2001a799aa6210456c2d806d3574f7ea1f/letsql-0.1.12-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
            sha256 = "sha256-1SSYEzLzPYt1dE1q4s7sEbVRA6Sc0j3/VSWx1Q0kGRk=";
          };
          format = "wheel";
          nativeBuildInputs = crateWheelLib.usePyprojectWheelHook old pythonSet-base;
        });
      };
      overridePythonSet =
        overrides: pythonSet-base.overrideScope (pkgs.lib.composeManyExtensions overrides);
      pythonSet-editable = overridePythonSet [
        pyprojectOverrides-editable
        editableOverlay
      ];
      pythonSet-wheel = overridePythonSet [ pyprojectOverrides-wheel ];
      pythonSet-pypi = overridePythonSet [ pyprojectOverrides-pypi ];
      virtualenv-editable = pythonSet-editable.mkVirtualEnv "xorq" workspace.deps.all;
      virtualenv = pythonSet-wheel.mkVirtualEnv "xorq" workspace.deps.all;
      virtualenv-pypi = pythonSet-pypi.mkVirtualEnv "xorq" workspace.deps.all;

      editableShellHook = ''
        # Undo dependency propagation by nixpkgs.
        unset PYTHONPATH
        # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
        export REPO_ROOT=$(git rev-parse --show-toplevel)
      '';
      maybeMaturinBuildHook = ''
        set -eu

        repo_dir=$(git rev-parse --show-toplevel)
        if [ "$(basename "$repo_dir")" != "xorq" ]; then
          echo "not in xorq, exiting"
          exit 1
        fi
        case $(uname) in
          Darwin) suffix=dylib ;;
          *)      suffix=so    ;;
        esac
        source=$repo_dir/target/release/maturin/libletsql.$suffix
        target=$repo_dir/python/xorq/_internal.abi3.so

        if [ -e "$target" ]; then
          for other in $(find src -name '*rs'); do
            if [ "$target" -ot "$other" ]; then
              rm -f "$target"
              break
            fi
          done
        fi

        if [ ! -e "$source" -o ! -e "$target" ]; then
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
        pkgs.gh
      ];
      shell = pkgs.mkShell {
        packages = [
          virtualenv
        ] ++ toolsPackages;
        shellHook = ''
          unset PYTHONPATH
        '';
      };
      pypiShell = pkgs.mkShell {
        packages = [
          virtualenv-pypi
        ];
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
        pythonSet-base
        pythonSet-editable
        pythonSet-wheel
        virtualenv
        virtualenv-editable
        virtualenv-pypi
        editableShellHook
        maybeMaturinBuildHook
        toolchain
        letsql-commands-star
        toolsPackages
        shell
        editableShell
        pypiShell
        ;
    };
in
mkLETSQL
