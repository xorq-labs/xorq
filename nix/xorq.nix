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
        # scipy, pyarrow, etc. now resolve to prebuilt macosx wheels (see wheelStdenv),
        # so their darwin source-build overrides are no longer needed.
        # xgboost's prebuilt macosx wheel ships libxgboost.dylib linked against
        # @rpath/libomp.dylib (Homebrew's OpenMP); add an rpath to nixpkgs' libomp so
        # it resolves inside the nix store.
        xgboost = prev.xgboost.overrideAttrs (old: {
          buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.llvmPackages.openmp ];
          postInstall = (old.postInstall or "") + ''
            install_name_tool -add_rpath ${pkgs.llvmPackages.openmp}/lib \
              "$out/${python.sitePackages}/xgboost/lib/libxgboost.dylib"
          '';
        });
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
        boring-semantic-layer = prev.boring-semantic-layer.overrideAttrs (addResolved final [ "hatchling" ]);
        pyroaring = prev.pyroaring.overrideAttrs (addResolved final [ "setuptools" "cython" ]);
        pyiceberg = prev.pyiceberg.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
            prev.poetry-core
          ];
        });
        thrift = prev.thrift.overrideAttrs (compose [
          (addResolved final [ "setuptools" ])
          (old: {
            preBuild = ''
              _shim=$TMPDIR/distutils-shim
              mkdir -p "$_shim"
              cat > "$_shim/sitecustomize.py" << 'SITE'
import _distutils_hack
_distutils_hack.add_shim()
SITE
              export PYTHONPATH=${final.setuptools}/${python.sitePackages}:$_shim:''${PYTHONPATH:-}
            '';
          })
        ]);
        psycopg-c = (prev.psycopg-c.overrideAttrs (addResolved final [
          "setuptools"
        ])).overrideAttrs(addNativeBuildInputs [ pkgs.postgresql.pg_config ]);
        # git-annex PyPI package is a platform-specific binary wheel that
        # uv2nix cannot resolve.  Stub it out and provide the real binary
        # via pkgs.git-annex in the shell packages instead.
        git-annex = python.pkgs.buildPythonPackage {
          pname = "git-annex";
          version = "0.0.0";
          src = pkgs.emptyDirectory;
          format = "other";
          installPhase = ''
            mkdir -p $out/${python.sitePackages}
          '';
        };
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
      # On darwin, nixpkgs reports a low SDK version (11.3), which makes uv2nix's
      # wheel selector (pypa.selectWheels reads stdenv.targetPlatform.darwinSdkVersion)
      # reject scipy/pyarrow/etc. prebuilt wheels tagged macosx_12+, forcing slow
      # source builds. Report a higher SDK *for wheel selection only* via a metadata
      # `//` override (mkDerivation/toolchain are untouched). selectWheels still picks
      # the most-portable compatible wheel (e.g. macosx_12_0), so this does not assume
      # a newer macOS at runtime.
      wheelStdenv =
        if pkgs.stdenv.isDarwin then
          pkgs.stdenv
          // {
            hostPlatform = pkgs.stdenv.hostPlatform // { darwinSdkVersion = "14.0"; };
            targetPlatform = pkgs.stdenv.targetPlatform // { darwinSdkVersion = "14.0"; };
            buildPlatform = pkgs.stdenv.buildPlatform // { darwinSdkVersion = "14.0"; };
          }
        else
          pkgs.stdenv;
      pythonSet-base =
        # Use base package set from pyproject.nix builders
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
          stdenv = wheelStdenv;
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
        pkgs.git-annex
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
          UV_TOOL_RUN_LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
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
          UV_TOOL_RUN_LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
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
