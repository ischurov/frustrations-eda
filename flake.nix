{
  description = "Python experiments";

  nixConfig = {
    extra-substituters = "https://twesterhout-chapel.cachix.org";
    extra-trusted-public-keys =
      "twesterhout-chapel.cachix.org-1:bs5PQPqy21+rP2KJl+O40/eFVzdsTe6m7ZTiOEE7PaI=";
  };

  inputs = {
    nixpkgs.follows = "lattice-symmetries/nixpkgs";
    flake-utils.follows = "lattice-symmetries/flake-utils";
    lattice-symmetries.url = "github:twesterhout/lattice-symmetries/nikita";
    nix-on-the-cluster.url = "github:twesterhout/nix-on-the-cluster";
    ising-glass-annealer = {
      url = "github:twesterhout/ising-glass-annealer";
      inputs.nixpkgs.follows = "lattice-symmetries/nixpkgs";
      inputs.flake-utils.follows = "lattice-symmetries/flake-utils";
    };
  };

  outputs = inputs:
    let
      torch-bin-overlay = (final: prev: {
        pythonPackagesExtensions = (prev.pythonPackagesExtensions or [ ]) ++ [
          (py-final: py-prev: {
            torch = py-final.torch-bin.overrideAttrs (attrs: {
              # torchvision expects torch to have these attributes
              passthru = attrs.passthru // {
                cudaSupport = final.config.cudaSupport;
                cudaPackages = final.cudaPackages;
                cudaCapabilities =
                  if final.config.cudaSupport then [ "7.0+PTX" ] else [ ];
              };
            });
            torchvision = py-final.torchvision-bin;
            combinadics = py-final.callPackage ./nix/combinadics.nix { };
            HolisticTraceAnalysis =
              py-final.callPackage ./nix/holistic-trace-analysis.nix { };
	    lattice-symmetries = py-prev.lattice-symmetries.overridePythonAttrs (attrs: { doCheck = false; checkPhase = "true"; });
          })
        ];
      });

      pkgs-for = system:
        import inputs.nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
          config.nvidia.acceptLicense = true;
          overlays = [
            inputs.lattice-symmetries.overlays.default
            inputs.ising-glass-annealer.overlays.default
            inputs.nix-on-the-cluster.overlays.lilo
            torch-bin-overlay
          ];
        };

      # Our Python dependencies
      my-python-packages = ps:
        with ps; [
          HolisticTraceAnalysis
          bitarray
          combinadics
          fire
          gitpython
          igraph
          ipympl
          ising-glass-annealer
          jsonlines
          jupyter
          lattice-symmetries
          loguru
          matplotlib
          more-itertools
          numpy
          pandas
          plotly
          # plotnine
          pytest
          pyyaml
          scikit-learn
          scipy
          seaborn
          snakeviz
          sympy
          tensorboard
          torch
          torchmetrics
          torchvision
          tqdm
        ];
    in {
      packages = inputs.flake-utils.lib.eachDefaultSystemMap (system:
        with (pkgs-for system); {
          default = singularity-tools.buildImage {
            name = "frustrations-eda";
            contents = [ (python3.withPackages my-python-packages) coreutils ];
            diskSize = 20480;
            memSize = 5120;
          };
        });
      devShells = inputs.flake-utils.lib.eachDefaultSystemMap (system:
        with (pkgs-for system); {
          default = mkShell {
            nativeBuildInputs = [
              ffmpeg
              (python3.withPackages my-python-packages)
              # LSP support for Python
              python3Packages.black
              py-spy
              nodePackages.pyright
              # Nix stuff
              nil
              nixpkgs-fmt
              nixfmt
              # direnv
              direnv
              nvtop
              nvtop-nvidia
            ];
            shellHook = ''
                            export PROMPT_COMMAND=""
                            export PS1='🐍 Python ${python3.version} \w $ '
                            export LS_PATH=${lattice-symmetries.python}
              	            export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
            '';
            # LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH

          };
        });
    };
}
