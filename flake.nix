{
  description = "Python experiments";

  nixConfig = {
    extra-substituters = "https://twesterhout-chapel.cachix.org";
    extra-trusted-public-keys = "twesterhout-chapel.cachix.org-1:bs5PQPqy21+rP2KJl+O40/eFVzdsTe6m7ZTiOEE7PaI=";
  };

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    lattice-symmetries = {
      url = "github:twesterhout/lattice-symmetries";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };
  };

  outputs = inputs:
    let
      pkgs-for = system: import inputs.nixpkgs {
        inherit system;
        overlays = [ inputs.lattice-symmetries.overlays.default ];
      };

      # Our Python dependencies
      my-python-packages = ps: with ps; [
        bitarray
        igraph
        jsonlines
        jupyter
        fire
        lattice-symmetries
        loguru
        matplotlib
        more-itertools
        numpy
        pandas
        pytest
        pyyaml
        scikit-learn
        scipy
        seaborn
        sympy
        torch
	      torchmetrics
        tqdm
      ];
    in
    {
      devShells = inputs.flake-utils.lib.eachDefaultSystemMap (system: with (pkgs-for system); {
        default = mkShell {
          nativeBuildInputs = [
            (python3.withPackages my-python-packages)
            # LSP support for Python
            python3Packages.black
            nodePackages.pyright
            # Nix stuff
            nil
            nixpkgs-fmt
            # direnv
            direnv
          ];
          shellHook = ''
            export PROMPT_COMMAND=""
            export PS1='🐍 Python ${python3.version} \w $ '
            export LS_PATH=${lattice-symmetries.python}
          '';
        };
      });
    };
}
