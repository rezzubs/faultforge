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
          # Linker for the extension module builds (see MATURIN_PEP517_ARGS).
          zig

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

        # Without a uv-managed Python installed, uv would otherwise pick up
        # `python314` from above. Being nix-native, it doesn't go through
        # nix-ld or the system library paths, so wheels that need e.g.
        # libstdc++ fail to load.
        UV_PYTHON_PREFERENCE = "only-managed";

        # The extension module is loaded by uv's Python, which uses the host's
        # glibc rather than nix's. Linking with the nix toolchain would tie the
        # module to nix's (newer) glibc, so it fails to load on hosts with an
        # older one. Instead, link with zig against a fixed glibc baseline,
        # same as the manylinux wheels it's loaded alongside. 2_28 matches
        # torch's own requirement. Applies to the builds uv runs through
        # maturin's PEP 517 backend.
        MATURIN_PEP517_ARGS = "--zig --compatibility manylinux_2_28";
      };

      formatter.${system} = pkgs.nixfmt-tree;
    };
}
