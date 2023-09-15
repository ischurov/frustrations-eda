#!/bin/bash

#SBATCH --job-name="swo_lanczos"
#SBATCH --time=2-00:00:00     # walltime
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -p tcm
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --array=1-11%1

export PATH="/home/ischurov/.local/bin:$PATH"
cd /home/ischurov/tcm10/frustrations-eda

nix develop .#default --command python ./swo_lanczos_2023_09_09.py $SLURM_ARRAY_TASK_ID
