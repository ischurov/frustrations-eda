#!/bin/bash

#SBATCH --job-name="complexity"
#SBATCH --time=23:59:59     # walltime
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -p tcm
#SBATCH --exclusive

source /opt/jupyter-conda/etc/profile.d/conda.sh 

cd /home/ischurov/tcm10/frustrations-eda
conda activate latsym2
python ./boolean-analysis-complexity.py $@

