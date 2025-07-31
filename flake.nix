{
  description = "A modern data processing library focused on composability, portability, and performance.";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs = {
        pyproject-nix.follows = "pyproject-nix";
        uv2nix.follows = "uv2nix";
        nixpkgs.follows = "nixpkgs";
      };
    };
    nix-utils = {
      url = "github:xorq-labs/nix-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      nix-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pythonDarwinHotfix = old: {
          # https://github.com/NixOS/nixpkgs/pull/390454
          preConfigure = old.preConfigure + (
            pkgs.lib.optionalString
            (system == "aarch64-darwin")
            ''
              # Fix _ctypes module compilation
              export NIX_CFLAGS_COMPILE+=" -DUSING_APPLE_OS_LIBFFI=1"
            ''
          );
        };
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            (_final: prev: {
              python310 = prev.python310.overrideAttrs pythonDarwinHotfix;
              python311 = prev.python311.overrideAttrs pythonDarwinHotfix;
            })
          ];
        };
        inherit (nix-utils.lib.${system}.utils) drvToApp;

        src = ./.;
        mkXorq = import ./nix/xorq.nix {
          inherit
            system
            pkgs
            pyproject-nix
            uv2nix
            pyproject-build-systems
            src
            ;
        };
        xorq-310 = mkXorq pkgs.python310;
        xorq-311 = mkXorq pkgs.python311;
        xorq-312 = mkXorq pkgs.python312;
        xorq-313 = mkXorq pkgs.python313;
      in
      {
        formatter = pkgs.nixfmt-rfc-style;
        apps = {
          python-310-default = drvToApp {
            drv = xorq-310.virtualenv-default;
            name = "python";
          };
          python-311-default = drvToApp {
            drv = xorq-311.virtualenv-default;
            name = "python";
          };
          python-312-default = drvToApp {
            drv = xorq-312.virtualenv-default;
            name = "python";
          };
          python-313-default = drvToApp {
            drv = xorq-313.virtualenv-default;
            name = "python";
          };
          ipython-310-all = drvToApp {
            drv = xorq-310.virtualenv-all;
            name = "ipython";
          };
          ipython-311-all = drvToApp {
            drv = xorq-311.virtualenv-all;
            name = "ipython";
          };
          ipython-312-all = drvToApp {
            drv = xorq-312.virtualenv-all;
            name = "ipython";
          };
          ipython-313-all = drvToApp {
            drv = xorq-313.virtualenv-all;
            name = "ipython";
          };
          default = self.apps.${system}.python-312-default;
          xorq = drvToApp {
            drv = xorq-312.virtualenv-all;
            name = "xorq";
          };
        };
        lib = {
          inherit
            pkgs
            src
            mkXorq
            xorq-310
            xorq-311
            xorq-312
            ;
        };
        devShells = {
          impure = pkgs.mkShell {
            packages = [
              pkgs.python310
              pkgs.uv
              pkgs.gh
            ];
            env =
              {
                # Prevent uv from managing Python downloads
                UV_PYTHON_DOWNLOADS = "never";
                # Force uv to use nixpkgs Python interpreter
                UV_PYTHON = pkgs.python310.interpreter;
              }
              // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
                # Python libraries often load native shared objects using dlopen(3).
                # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
                LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1;
              };
            shellHook = ''
              unset PYTHONPATH
            '';
          };
          uv = pkgs.mkShell {
            packages = [
              pkgs.python310
              pkgs.uv
            ];
            shellHook = ''
              unset PYTHONPATH
            '';
          };
          virtualenv-310 = xorq-310.shell;
          virtualenv-default-313 = xorq-313.defaultShell;
          virtualenv-editable-310 = xorq-310.editableShell;
          virtualenv-311 = xorq-311.shell;
          virtualenv-editable-311 = xorq-311.editableShell;
          virtualenv-312 = xorq-312.shell;
          virtualenv-editable-312 = xorq-312.editableShell;
          virtualenv-313 = xorq-313.shell;
          virtualenv-editable-313 = xorq-313.editableShell;
          default = self.devShells.${system}.virtualenv-313;
        };
      }
    );
}
