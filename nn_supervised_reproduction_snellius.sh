#!/bin/bash

#SBATCH --job-name="nn_supervised_reproduction"
#SBATCH --time=11:59:59     # walltime
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -c 18

export PATH="/home/ischurov/.local/bin:$PATH"
cd /home/ischurov/tcm10/frustrations-eda

nix develop .#default --command python ./nn_supervised_reproduction.py $SLURM_ARRAY_TASK_ID
