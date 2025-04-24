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

        grpcio = prev.grpcio.overrideAttrs (compose [
          (addBuildInputs [ pkgs.zlib pkgs.openssl pkgs.c-ares ])
          (addNativeBuildInputs [ pkgs.pkg-config pkgs.cmake ])
        
          (old: {
             NIX_CFLAGS_COMPILE = (old.NIX_CFLAGS_COMPILE or "") +
               " -DTARGET_OS_OSX=1 -D_DARWIN_C_SOURCE" +
               " -I${pkgs.zlib.dev}/include" +
               " -I${pkgs.openssl.dev}/include" +
               " -I${pkgs.c-ares.dev}/include";
             
             NIX_LDFLAGS = (old.NIX_LDFLAGS or "") +
               " -L${pkgs.zlib.out}/lib -lz" +
               " -L${pkgs.openssl.out}/lib -lssl -lcrypto" +
               " -L${pkgs.c-ares.out}/lib -lcares";
        
            GRPC_PYTHON_BUILD_SYSTEM_OPENSSL = "1";
            GRPC_PYTHON_BUILD_SYSTEM_ZLIB    = "1";
            GRPC_PYTHON_BUILD_SYSTEM_CARES   = "1";
        
            preBuild = ''
              export PYTHONPATH=${final.setuptools}/${python.sitePackages}:$PYTHONPATH
            '';
          })
        ]);

      };
      addNativeBuildInputs =
        drvs:
        (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ drvs;
        });
      addBuildInputs =
        drvs:
        old: {
          buildInputs = (old.buildInputs or []) ++ drvs;
        };
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
      overridePythonSet =
        overrides: pythonSet-base.overrideScope (pkgs.lib.composeManyExtensions overrides);
      pythonSet-editable = overridePythonSet [
        pyprojectOverrides-editable
        editableOverlay
      ];
      pythonSet-wheel = overridePythonSet [ pyprojectOverrides-wheel ];
      virtualenv-editable = pythonSet-editable.mkVirtualEnv "xorq" workspace.deps.all;
      virtualenv = pythonSet-wheel.mkVirtualEnv "xorq" workspace.deps.all;
      virtualenv-default = pythonSet-wheel.mkVirtualEnv "xorq-default" workspace.deps.default;

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
      defaultShell = pkgs.mkShell {
        packages = [
          virtualenv-default
        ] ++ toolsPackages;
        shellHook = ''
          unset PYTHONPATH
        '';
      };
      shell = pkgs.mkShell {
        packages = [
          virtualenv
        ] ++ toolsPackages;
        shellHook = ''
          unset PYTHONPATH
        '';
        env = {
          UV_NO_SYNC = "1";
          UV_PYTHON = "${virtualenv}/bin/python";
          UV_PYTHON_DOWNLOADS = "never";
        };
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
        env = {
          UV_NO_SYNC = "1";
          UV_PYTHON = "${virtualenv-editable}/bin/python";
          UV_PYTHON_DOWNLOADS = "never";
        };
      };

    in
    {
      inherit
        pythonSet-base
        pythonSet-editable
        pythonSet-wheel
        virtualenv
        virtualenv-editable
        virtualenv-default
        editableShellHook
        maybeMaturinBuildHook
        toolchain
        letsql-commands-star
        toolsPackages
        shell
        virtualenv-pure-pypi
        editableShell
        defaultShell
        ;
    };
in
mkLETSQL
