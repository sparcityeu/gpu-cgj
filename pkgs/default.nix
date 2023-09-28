# Custom packages, that can be defined similarly to ones from nixpkgs
# You can build them using `nix build .#example` or (legacy) `nix-build -A example`
{pkgs ? (import ../nixpkgs.nix) {}}: {
  # example = pkgs.callPackage ./example { };
  ligra-apps = pkgs.callPackage ./ligra/apps.nix { };
  ligra-utils = pkgs.callPackage ./ligra/utils.nix { };
  krill = pkgs.callPackage ./krill { };
}
