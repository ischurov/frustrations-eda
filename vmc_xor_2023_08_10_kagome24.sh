#!/bin/bash

#SBATCH --job-name="vmc_xor"
#SBATCH --time=1-00:00:00     # walltime
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -p tcm
#SBATCH --array=0-1%2
#SBATCH --exclusive
set -x
export PATH="/home/ischurov/.local/bin:$PATH"
cd /home/ischurov/tcm10/frustrations-eda

nix develop .#default --command python ./vmc_xor_2023_08_10_kagome24.py $SLURM_ARRAY_TASK_ID
