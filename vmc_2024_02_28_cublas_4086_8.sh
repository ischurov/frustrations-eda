#!/bin/bash

#SBATCH --job-name="vmc"
#SBATCH --time=00:59:59     # walltime
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -c 18

export PATH="/home/ishchurov/.local/bin:$PATH"

cd /home/ishchurov/frustrations-eda

apptainer exec --nv singularity-image-frustrations-eda.img  \
          sh -c 'export CUBLAS_WORKSPACE_CONFIG=:4096:8; python ./vmc_2024_02_28.py $SLURM_ARRAY_TASK_ID'

