{
  description = "Python experiments";

  nixConfig = {
    extra-substituters = "https://twesterhout-chapel.cachix.org";
    extra-trusted-public-keys = "twesterhout-chapel.cachix.org-1:bs5PQPqy21+rP2KJl+O40/eFVzdsTe6m7ZTiOEE7PaI=";
  };

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nix-filter.url = "github:numtide/nix-filter";
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };
    lattice-symmetries = {
      url = "github:twesterhout/lattice-symmetries-haskell";
    };
  };

  outputs = inputs: inputs.flake-utils.lib.eachDefaultSystem (system:
    with builtins;
    let
      pkgs = import inputs.nixpkgs { inherit system; };

      lattice-symmetries = inputs.lattice-symmetries.packages.${system}.python;
      my-python = pkgs.python3.withPackages (ps: with ps; [
        bitarray
        igraph
        jsonlines
        jupyter
        loguru
        matplotlib
        more-itertools
        numpy
        pandas
        pyyaml
        scikit-learn
        scipy
        seaborn
        sympy
        torch
      ]);
    in
    {
      devShells.default =
        pkgs.mkShell {
          packages = [ ];
          buildInputs = [ lattice-symmetries ];
          nativeBuildInputs = with pkgs; [
            my-python
            # lsp support for Python
            python3Packages.black
            nodePackages.pyright
            # Nix stuff
            nil
            nixpkgs-fmt
          ];
          shellHook = ''
            export PROMPT_COMMAND=""
            export PS1='🐍 Python ${pkgs.python3.version} \w $ '
            export LS_PATH=${lattice-symmetries}
          '';
        };
      formatter = pkgs.nixpkgs-fmt;
    });
}
