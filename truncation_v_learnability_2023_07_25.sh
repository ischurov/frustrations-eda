#!/bin/bash

#SBATCH --job-name="truncation_v_learnability_2023_07_25"
#SBATCH --time=12:00:00     # walltime
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -p tcm
#SBATCH --exclusive
#SBATCH --array=0-20%5

export PATH="/home/ischurov/.local/bin:$PATH"
cd /home/ischurov/tcm10/frustrations-eda

nix develop .#default --command python ./truncation_v_learnability_2023_07_25.py --task_id $SLURM_ARRAY_TASK_ID $@
