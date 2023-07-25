#!/bin/bash

#SBATCH --job-name="fourier_xor_shuffle_2023_07_25"
#SBATCH --time=2-00:00:00     # walltime
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -p tcm
#SBATCH --exclusive
#SBATCH --array=0-7%4

export PATH="/home/ischurov/.local/bin:$PATH"
cd /home/ischurov/tcm10/frustrations-eda

nix develop .#default --command python ./fourier_xor_shuffle_2023_07_25.py --task_id $SLURM_ARRAY_TASK_ID $@
