{
  pkgs,
  pyproject-nix,
  uv2nix,
  pyproject-build-systems,
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
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = src; };
      wheelOverlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };
      editableOverlay = workspace.mkEditablePyprojectOverlay {
        root = "$REPO_ROOT";
      };

      virtualenv-pure-pypi =
        let
        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./pure-pypi; };
        wheelOverlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };
        # pyprojectOverrides-base
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
        virtualenv = pythonSet-base.mkVirtualEnv "xorq" workspace.deps.all;
        in virtualenv;

      pyprojectOverrides-base = final: prev: {
        cityhash = prev.cityhash.overrideAttrs (
          addResolved final (if python.pythonAtLeast "3.12" then [ "setuptools" ] else [ ])
        );
      };
      pyprojectOverrides-editable = final: prev: {
        xorq = prev.xorq.overrideAttrs (old: {
          nativeBuildInputs =
            (old.nativeBuildInputs or [ ])
            ++ final.resolveBuildSystem {
              editables = [ ];
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
      overridePythonSet =
        overrides: pythonSet-base.overrideScope (pkgs.lib.composeManyExtensions overrides);
      pythonSet-editable = overridePythonSet [
        pyprojectOverrides-editable
        editableOverlay
      ];
      virtualenv-editable = pythonSet-editable.mkVirtualEnv "xorq" workspace.deps.all;
      virtualenv = pythonSet-base.mkVirtualEnv "xorq" workspace.deps.all;

      editableShellHook = ''
        # Undo dependency propagation by nixpkgs.
        unset PYTHONPATH
        # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
        export REPO_ROOT=$(git rev-parse --show-toplevel)
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
      editableShell = pkgs.mkShell {
        packages = [
          virtualenv-editable
        ] ++ toolsPackages;
        shellHook = editableShellHook;
      };

    in
    {
      inherit
        pythonSet-base
        pythonSet-editable
        virtualenv
        virtualenv-editable
        editableShellHook
        letsql-commands-star
        toolsPackages
        shell
        virtualenv-pure-pypi
        editableShell
        ;
    };
in
mkLETSQL
