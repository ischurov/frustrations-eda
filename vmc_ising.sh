#!/bin/bash

#SBATCH --job-name="vmc_ising"
#SBATCH --time=00:59:59     # walltime
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -c 18

export PATH="/home/ishchurov/.local/bin:$PATH"

cd /home/ishchurov/frustrations-eda

apptainer exec --bind /usr/lib64:/usr/lib64 \
	       --bind /sw/arch/RHEL8/EB_production/2023/software/CUDA/12.1.1/lib/:/sw/arch/RHEL8/EB_production/2023/software/CUDA/12.1.1/lib/ \
	       singularity-image-frustrations-eda.img sh -c 'LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH python ./vmc_ising.py $SLURM_ARRAY_TASK_ID'

