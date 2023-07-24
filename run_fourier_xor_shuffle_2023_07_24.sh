#!/bin/bash

#SBATCH --job-name="fourier_xor_shuffle_2023_07_24"
#SBATCH --time=3-00:00:00     # walltime
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -p tcm
#SBATCH --exclusive

export PATH="/home/ischurov/.local/bin:$PATH"
cd /home/ischurov/tcm10/frustrations-eda

nix develop .#default --command python ./fourier_xor_shuffle_2023_07_24.py $@
