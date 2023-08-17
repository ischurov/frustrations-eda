#!/bin/bash

#SBATCH --job-name="truncation_v_learnability"
#SBATCH --time=2-00:00:00     # walltime
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -p tcm
#SBATCH --exclusive
#SBATCH --array=0-20%2
#SBATCH --exclude=cn90,cn91,cn41,cn42,cn24,cn19

export PATH="/home/ischurov/.local/bin:$PATH"
cd /home/ischurov/tcm10/frustrations-eda

nix develop .#default --command python ./truncation_v_learnability_2023_08_08.py --task_id $SLURM_ARRAY_TASK_ID
