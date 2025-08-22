{
  description = "Python 3.11 devShell for direnv";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";

  outputs = { self, nixpkgs }:
  let
  system = "x86_64-linux"; # change to aarch64-darwin if on Apple Silicon
  pkgs = import nixpkgs { inherit system; };
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = [
        pkgs.python311
        pkgs.python311Packages.pip
        pkgs.python311Packages.virtualenv
      ];
      shellHook = ''
        echo "🐍 Using Python $(${pkgs.python311}/bin/python --version)"
      '';
    };
  };
}
