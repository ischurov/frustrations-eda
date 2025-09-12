#!/bin/bash
#
#SBATCH --job-name='xors'
#SBATCH -N1
#SBATCH --mem 0
#SBATCH -p tcm
#SBATCH --time=2-0:00:00
#SBATCH --output=/vol/tcm11/kravchenko/to_scratch/analysislogs/slurm-%j.out
#SBATCH --error=/vol/tcm11/kravchenko/to_scratch/analysislogs/slurm-%j.err

export PATH="/vol/tcm11/kravchenko/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH=/vol/tcm11/kravchenko/miniconda3/envs/ls/lib
source /vol/tcm11/kravchenko/miniconda3/bin/activate ls


PYTHON=python

${PYTHON} train_for_paper_xors.py triangle truncated
${PYTHON} train_for_paper_xors.py square truncated
${PYTHON} train_for_paper_xors.py kagome truncated

${PYTHON} train_for_paper_xors.py triangle flat
${PYTHON} train_for_paper_xors.py square flat
${PYTHON} train_for_paper_xors.py kagome flat

${PYTHON} train_for_paper_xors.py kagome nosign
${PYTHON} train_for_paper_xors.py triangle nosign
${PYTHON} train_for_paper_xors.py square nosign
 
