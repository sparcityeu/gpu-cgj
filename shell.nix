# Development environment
# You can enter it through `nix develop` or (legacy) `nix-shell`
{pkgs ? (import ./nixpkgs.nix) {}}: {
  default = pkgs.mkShell {
    nativeBuildInputs = with pkgs; [
      cudatoolkit
      # Run CUDA executables with `steam-run` so they can find libcuda.so
      # https://stackoverflow.com/questions/3253257/cuda-driver-version-is-insufficient-for-cuda-runtime-version
      steam-run

      ligra-utils
    ];
  };
}
