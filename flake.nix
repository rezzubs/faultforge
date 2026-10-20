{
  description = "FaultForge development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        # NOTE: the project's own Python is deliberately left out. We can let uv
        # manage it which is fine because we already require nix-ld on NixOS.
        packages = with pkgs; [
          cargo
          cargo-nextest
          clippy
          just
          ruff
          rust-analyzer
          rustc
          rustfmt
          ty
          uv

          # Not used to build or run faultforge itself (see NOTE above) - only
          # gives pyo3-ffi's build script a Python to find for `cargo
          # clippy`/`test` on the `bindings` crate. Being nix-native, it's
          # correctly RPATH-linked, so cargo's test binaries can find libpython
          # at runtime without any LD_LIBRARY_PATH help.
          python314
        ];

        # Attributes not recognized by mkShell (packages, shellHook, etc.)
        # are exported as environment variables in the shell.
        RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";
      };

      formatter.${system} = pkgs.nixfmt-tree;
    };
}
