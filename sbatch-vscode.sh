#!/bin/bash

#SBATCH --job-name="vscode"
#SBATCH --time=12:00:00     # walltime
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -p tcm
#SBATCH --exclusive
#SBATCH --exclude cn[90,92,94]

set -x

export PATH="/home/ischurov/.local/bin:$PATH"
cd /home/ischurov/tcm10/frustrations-eda

nix develop .#default --command code tunnel --accept-server-license-terms 
