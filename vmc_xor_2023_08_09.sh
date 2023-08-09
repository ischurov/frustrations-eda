#!/bin/bash

#SBATCH --job-name="vmc_xor"
#SBATCH --time=1-00:00:00     # walltime
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -p tcm
#SBATCH --exclusive
#SBATCH --array=0-40%2
#SBATCH --exclude=cn90,cn91,cn41,cn42,cn39,cn24,cn19
set -x
export PATH="/home/ischurov/.local/bin:$PATH"
cd /home/ischurov/tcm10/frustrations-eda

nix develop .#default --command python ./vmc_xor_2023_08_09.py $SLURM_ARRAY_TASK_ID
