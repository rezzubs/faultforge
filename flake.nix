{
  description = "FaultForge development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {nixpkgs, ...}: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {inherit system;};

    # Libraries that prebuilt (non-nix-built) binaries need but won't find
    # on NixOS.
    #
    # Used for
    # - LD_LIBRARY_PATH - dlopen()-ing a wheel's compiled extension (even for
    #   nix-native binaries).
    # - NIX_LD_LIBRARY_PATH - nix-ld resolving a prebuilt binary's own
    #   dependencies at startup.
    #
    # Same libraries are needed in both cases, so both variables share
    # this one list. Requires `programs.nix-ld.enable` in your NixOS system
    # configuration.
    foreignLibraryPath = pkgs.lib.makeLibraryPath [
      # libstdc++ for torch/numpy
      # libc for cPython binaries downloaded by `uv`.
      pkgs.stdenv.cc.cc.lib
      # for numpy
      pkgs.zlib
      # matplotlib's compiled `_c_internal_utils` extension dlopen()s these to
      # probe for a usable display (`display_is_valid()`); without them the
      # dlopen silently fails, matplotlib assumes headless, and it falls back
      # to the non-interactive Agg backend even when a real X11/Wayland
      # session is running.
      pkgs.libX11
      pkgs.wayland
    ];
  in {
    devShells.${system}.default = pkgs.mkShell {
      # NOTE: the project's own Python is deliberately left out. We can let uv
      # manage it which is fine because we already require nix-ld on NixOS.
      packages = with pkgs; [
        cargo
        cargo-nextest
        clippy
        just
        # Not used to build or run faultforge itself (see NOTE above) - only
        # gives pyo3-ffi's build script a Python to find for `cargo
        # clippy`/`test` on the `bindings` crate. Being nix-native, it's
        # correctly RPATH-linked, so cargo's test binaries can find libpython
        # at runtime without any LD_LIBRARY_PATH help.
        python314
        rust-analyzer
        rustc
        rustfmt
        uv
      ];

      # Attributes not recognized by mkShell (packages, shellHook, etc.)
      # are exported as environment variables in the shell.
      RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";

      LD_LIBRARY_PATH = foreignLibraryPath;
      NIX_LD_LIBRARY_PATH = foreignLibraryPath;
    };

    formatter.${system} = pkgs.alejandra;
  };
}
