#!/bin/bash

#SBATCH --job-name="swo_validation"
#SBATCH --time=2-00:00:00     # walltime
#SBATCH -N 1
#SBATCH -p tcm
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --array=27

export PATH="/home/ischurov/.local/bin:$PATH"
cd /home/ischurov/tcm10/frustrations-eda

nix develop .#default --command python ./swo_validation_2023_09_14.py $SLURM_ARRAY_TASK_ID
