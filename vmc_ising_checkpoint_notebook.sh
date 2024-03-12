#!/bin/bash

#SBATCH --job-name="vmc_ising"
#SBATCH --time=00:59:59     # walltime
#SBATCH -n 1
#SBATCH -p genoa
#SBATCH -c 16

export PATH="/home/ishchurov/.local/bin:$PATH"

cd /home/ishchurov/frustrations-eda

apptainer exec --nv singularity-image-frustrations-eda.img  \
          python ./vmc_ising_checkpoint_notebook.py

