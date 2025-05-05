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
    rust-overlay.url = "github:oxalica/rust-overlay";
    crane = {
      url = "github:ipetkov/crane";
    };
    nix-utils = {
      url = "github:letsql/nix-utils";
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
      rust-overlay,
      crane,
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
            (import rust-overlay)
            (_final: prev: {
              python310 = prev.python310.overrideAttrs pythonDarwinHotfix;
              python311 = prev.python311.overrideAttrs pythonDarwinHotfix;
            })
          ];
        };
        inherit (nix-utils.lib.${system}.utils) drvToApp;

        src = ./.;
        mkLETSQL = import ./nix/letsql.nix {
          inherit
            system
            pkgs
            pyproject-nix
            uv2nix
            pyproject-build-systems
            crane
            src
            ;
        };
        letsql-310 = mkLETSQL pkgs.python310;
        letsql-311 = mkLETSQL pkgs.python311;
        letsql-312 = mkLETSQL pkgs.python312;
      in
      {
        formatter = pkgs.nixfmt-rfc-style;
        apps = {
          ipython-310 = drvToApp {
            drv = letsql-310.virtualenv;
            name = "ipython";
          };
          ipython-311 = drvToApp {
            drv = letsql-311.virtualenv;
            name = "ipython";
          };
          ipython-312 = drvToApp {
            drv = letsql-312.virtualenv;
            name = "ipython";
          };
          default = self.apps.${system}.ipython-312;
        };
        lib = {
          inherit
            pkgs
            src
            mkLETSQL
            letsql-310
            letsql-311
            letsql-312
            ;
        };
        devShells = {
          impure = pkgs.mkShell {
            packages = [
              pkgs.python310
              letsql-310.toolchain
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
              letsql-310.toolchain
            ];
            shellHook = ''
              unset PYTHONPATH
            '';
          };
          virtualenv-310 = letsql-310.shell;
          virtualenv-default-310 = letsql-310.defaultShell;
          virtualenv-editable-310 = letsql-310.editableShell;
          virtualenv-311 = letsql-311.shell;
          virtualenv-editable-311 = letsql-311.editableShell;
          virtualenv-312 = letsql-312.shell;
          virtualenv-editable-312 = letsql-312.editableShell;
          default = self.devShells.${system}.virtualenv-310;
        };
      }
    );
}
