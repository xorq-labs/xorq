{
  system,
  pkgs,
  pyproject-nix,
  uv2nix,
  pyproject-build-systems,
  src,
}:
let
  mkXorq =
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
          prev.setuptools-scm
          prev.pybind11
        ]);
        pyarrow = let
          arrow-testing = pkgs.fetchFromGitHub {
            name = "arrow-testing";
            owner = "apache";
            repo = "arrow-testing";
            rev = "d2a13712303498963395318a4eb42872e66aead7";
            hash = "sha256-IkiCbuy0bWyClPZ4ZEdkEP7jFYLhM7RCuNLd6Lazd4o=";
          };
          parquet-testing = pkgs.fetchFromGitHub {
            name = "parquet-testing";
            owner = "apache";
            repo = "parquet-testing";
            rev = "18d17540097fca7c40be3d42c167e6bfad90763c";
            hash= "sha256-gKEQc2RKpVp39RmuZbIeIXAwiAXDHGnLXF6VQuJtnRA=";
          };
          version = "21.0.0";
          arrow-cpp = pkgs.arrow-cpp.overrideAttrs (old: {
            inherit version;
            src = pkgs.fetchFromGitHub {
              owner = "apache";
              repo = "arrow";
              rev = "apache-arrow-${version}";
              hash = "sha256-6RFa4GTNgjsHSX5LYp4t6p8ynmmr7Nuotj9C7mTmvlM=";
            };
            PARQUET_TEST_DATA = pkgs.lib.optionalString old.doInstallCheck "${parquet-testing}/data";
            ARROW_TEST_DATA = pkgs.lib.optionalString old.doInstallCheck "${arrow-testing}/data";
            # Disable mimalloc allocator to avoid missing header on Darwin (why?)
            cmakeFlags = (old.cmakeFlags or []) ++ [ "-DARROW_MIMALLOC=OFF" ];
          });
        in (prev.pyarrow.overrideAttrs (compose [
          (addBuildInputs [
            pkgs.pkg-config
            arrow-cpp
          ])
          (addNativeBuildInputs [
            python
            pkgs.cmake
            pkgs.pkg-config
            arrow-cpp
            prev.pyprojectBuildHook
            prev.pyprojectWheelHook
          ])
          (addResolved final [
            "setuptools"
            "cython"
            "numpy"
          ])
        ])).overrideAttrs (_: {
            preBuild = ''
              cd ..
            '';
        })
        ;
        google-crc32c = prev.google-crc32c.overrideAttrs (addNativeBuildInputs [ prev.setuptools ]);
        psycopg2-binary = prev.psycopg2-binary.overrideAttrs (addNativeBuildInputs [
          prev.setuptools
          pkgs.postgresql.pg_config
          pkgs.openssl
        ]);

        grpcio = prev.grpcio.overrideAttrs (compose [
          (addNativeBuildInputs [
            pkgs.pkg-config pkgs.cmake
          ])
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
              unset AR
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
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = src; };
      wheelOverlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };
      editableOverlay = workspace.mkEditablePyprojectOverlay {
        root = "$REPO_ROOT";
      };

      pyprojectOverrides-base = final: prev: {
        cityhash = prev.cityhash.overrideAttrs (
          addResolved final (if python.pythonAtLeast "3.12" then [ "setuptools" ] else [ ])
        );
        hash-cache = prev.hash-cache.overrideAttrs (addResolved final [ "hatchling" ]);
        xorq-hash-cache = prev.xorq-hash-cache.overrideAttrs (addResolved final [ "hatchling" ]);
        xorq-feature-utils = prev.xorq-feature-utils.overrideAttrs (addResolved final [ "hatchling" ]);
        xorq-weather-lib = prev.xorq-weather-lib.overrideAttrs (addResolved final [ "hatchling" ]);
        pyiceberg = prev.pyiceberg.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
            prev.poetry-core
          ];
        });
        psycopg-c = (prev.psycopg-c.overrideAttrs (addResolved final [
          "setuptools"
        ])).overrideAttrs(addNativeBuildInputs [ pkgs.postgresql.pg_config ]);
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
      virtualenv-default = pythonSet-base.mkVirtualEnv "xorq-default" workspace.deps.default;
      virtualenv-all = pythonSet-base.mkVirtualEnv "xorq-all" workspace.deps.all;
      virtualenv-editable = pythonSet-editable.mkVirtualEnv "xorq-editable" workspace.deps.all;

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
        xorq-commands-star
        ;
      toolsPackages = [
        pkgs.uv
        xorq-commands-star
        pkgs.gh
        pkgs.graphviz
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
          virtualenv-all
        ] ++ toolsPackages;
        shellHook = ''
          unset PYTHONPATH
        '';
        env = {
          UV_NO_SYNC = "1";
          UV_PYTHON = "${virtualenv-all}/bin/python";
          UV_PYTHON_DOWNLOADS = "never";
        };
      };
      editableShell = pkgs.mkShell {
        packages = [
          virtualenv-editable
        ] ++ toolsPackages;
        shellHook = editableShellHook;
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
        virtualenv-default
        virtualenv-all
        virtualenv-editable
        editableShellHook
        xorq-commands-star
        toolsPackages
        defaultShell
        shell
        editableShell
        ;
    };
in
mkXorq
