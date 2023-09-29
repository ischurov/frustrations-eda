#!/bin/bash

#SBATCH --job-name="fourier_paper_reproduction"
#SBATCH --time=2-00:00:00     # walltime
#SBATCH -N 1
#SBATCH -p tcm
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --array=0

export PATH="/home/ischurov/.local/bin:$PATH"
cd /home/ischurov/tcm10/frustrations-eda

nix develop .#default --command python ./fourier_paper_reproduction_2023_09_28.py $SLURM_ARRAY_TASK_ID
