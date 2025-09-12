#!/bin/bash
#
#SBATCH --job-name='sign_systems'
#SBATCH -N1
#SBATCH --mem 0
#SBATCH -p tcm
#SBATCH --time=2-0:00:00
#SBATCH --output=/vol/tcm11/kravchenko/to_scratch/frustrations-eda-main/paper_final_epochs/logs/slurm-%j.out
#SBATCH --error=/vol/tcm11/kravchenko/to_scratch/frustrations-eda-main/paper_final_epochs/logs/slurm-%j.err


PYTHON=python


${PYTHON} train_for_paper_epochs.py square truncated
${PYTHON} train_for_paper_epochs.py square flat
${PYTHON} train_for_paper_epochs.py square nosign
 
