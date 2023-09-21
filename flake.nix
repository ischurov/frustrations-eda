{
  description = "Python experiments";

  nixConfig = {
    extra-substituters = "https://twesterhout-chapel.cachix.org";
    extra-trusted-public-keys = "twesterhout-chapel.cachix.org-1:bs5PQPqy21+rP2KJl+O40/eFVzdsTe6m7ZTiOEE7PaI=";
  };

  inputs = rec {
    nixpkgs.follows = "lattice-symmetries/nixpkgs";
    flake-utils.follows = "lattice-symmetries/flake-utils";
    lattice-symmetries.url = "github:twesterhout/lattice-symmetries/nikita";
    ising-glass-annealer = {
      url = "github:twesterhout/ising-glass-annealer";
      inputs.nixpkgs.follows = "lattice-symmetries/nixpkgs";
      inputs.flake-utils.follows = "lattice-symmetries/flake-utils";
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
        overlays = [ inputs.lattice-symmetries.overlays.default inputs.ising-glass-annealer.overlays.default ];
      };

      # Our Python dependencies
      my-python-packages = ps: with ps; [
        bitarray
        igraph
        ising-glass-annealer
        jsonlines
        jupyter
        fire
        lattice-symmetries
        loguru
        matplotlib
        more-itertools
        numpy
        pandas
        plotnine
        pytest
        pyyaml
        scikit-learn
        scipy
        seaborn
        snakeviz
        sympy
        torch
        torchmetrics
        tqdm
        (
          buildPythonPackage rec {
            pname = "combinadics";
            version = "0.0.3";
            src = fetchPypi {
              inherit pname version;
              sha256 = "sha256-CFbtDgcbrFKEYknegVRSUZbc+jS0OCGN53ZYBAUAFD4=";
            };
            doCheck = false;
          }
        )
      ];
    in
    {
      devShells = inputs.flake-utils.lib.eachDefaultSystemMap (system: with (pkgs-for system); {
        default = mkShell {
          nativeBuildInputs = [
            ffmpeg
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
