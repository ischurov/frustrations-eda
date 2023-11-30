#!/bin/bash

#SBATCH --job-name="groundstates"
#SBATCH --time=2-00:00:00     # walltime
#SBATCH -N 1
#SBATCH -p tcm
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --exclude=cn[90,20,25]

export PATH="/home/ischurov/.local/bin:$PATH"
cd /home/ischurov/tcm10/frustrations-eda

nix develop .#default --command python ./groundstates_36_spins.py $SLURM_ARRAY_TASK_ID
