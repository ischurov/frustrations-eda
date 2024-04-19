#!/bin/bash

#SBATCH --job-name="nix-develop"
#SBATCH --time=12:00:00     # walltime
#SBATCH -n1
#S-BATCH -c192
#SBATCH --exclusive
#SBATCH -p genoa

set -x

# export PATH="/home/ischurov/.local/bin:$PATH"
cd /home/ishchurov/frustrations-eda

nix develop .#default --command echo OK
nix build


