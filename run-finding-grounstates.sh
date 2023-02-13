#!/bin/bash

#SBATCH --job-name="groundstate"
#SBATCH --time=23:00:00     # walltime
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -p tcm
#SBATCH --exclusive

source /opt/jupyter-conda/etc/profile.d/conda.sh 

cd /home/ischurov/tcm10/frustrations-eda
conda activate latsym2
python ./finding_groundstates_2023_02_13.py $@

