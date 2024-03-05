#!/bin/bash

#SBATCH --job-name="vmc_ising"
#SBATCH --time=00:59:59     # walltime
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -c 18

export PATH="/home/ishchurov/.local/bin:$PATH"

cd /home/ishchurov/frustrations-eda

apptainer exec --nv singularity-image-frustrations-eda.img  \
          python ./vmc_ising.py $SLURM_ARRAY_TASK_ID

