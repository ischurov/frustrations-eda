#!/bin/bash

#SBATCH --job-name="vmc_vs_lbfgs"
#SBATCH --time=1-00:00:00     # walltime
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -p tcm
#SBATCH --exclusive
set -x
export PATH="/home/ischurov/.local/bin:$PATH"
cd /home/ischurov/tcm10/frustrations-eda

nix develop .#default --command python ./vmc_vs_lbfgs_2023_08_09_kagome24.py
