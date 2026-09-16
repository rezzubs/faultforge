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
    ];
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = [
        # Rust
        pkgs.cargo
        pkgs.rustc
        pkgs.clippy
        pkgs.rustfmt
        pkgs.rust-analyzer
        pkgs.cargo-nextest
        # NOTE: python is deliberately left out. We can let uv manage it which
        # is fine because we already require nix-ld on NixOS.
        pkgs.uv
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
