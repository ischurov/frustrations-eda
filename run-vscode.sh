#!/bin/bash

#SBATCH --job-name="vscode"
#SBATCH --time=12:00:00     # walltime
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -p tcm
#SBATCH --exclusive

export PATH="/home/ischurov/.local/bin:$PATH"
cd /home/ischurov/tcm10/frustrations-eda

nix develop .#default --command code tunnel
