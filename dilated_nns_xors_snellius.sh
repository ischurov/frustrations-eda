#!/bin/bash

#SBATCH --job-name="dilated_nns_xors"
#SBATCH --time=11:59:59     # walltime
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -c 18

export PATH="/home/ishchurov/.local/bin:$PATH"
cd /home/ishchurov/tcm10/frustrations-eda

nix develop .#default --command python ./dilated_nns_xors.py $SLURM_ARRAY_TASK_ID
