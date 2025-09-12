#!/bin/bash
#
#SBATCH --job-name='xors'
#SBATCH -N1
#SBATCH --mem 0
#SBATCH -p tcm
#SBATCH --time=2-0:00:00
#SBATCH --output=/vol/tcm11/kravchenko/to_scratch/frustrations-eda-main/analysis/logs/slurm-%j.out
#SBATCH --error=/vol/tcm11/kravchenko/to_scratch/frustrations-eda-main/analysis/logs/slurm-%j.err

export PATH="/vol/tcm11/kravchenko/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH=/vol/tcm11/kravchenko/miniconda3/envs/ls/lib
source /vol/tcm11/kravchenko/miniconda3/bin/activate ls


PYTHON="/vol/tcm11/kravchenko/miniconda3/envs/ls/bin/python"

${PYTHON} plot_for_paper_coef_distribution_triangle.py 0.4
${PYTHON} plot_for_paper_coef_distribution_triangle.py 0.8
${PYTHON} plot_for_paper_coef_distribution_triangle.py 0.9
${PYTHON} plot_for_paper_coef_distribution_triangle.py 0.91
${PYTHON} plot_for_paper_coef_distribution_triangle.py 0.92
${PYTHON} plot_for_paper_coef_distribution_triangle.py 0.93
${PYTHON} plot_for_paper_coef_distribution_triangle.py 0.94
${PYTHON} plot_for_paper_coef_distribution_triangle.py 0.95
${PYTHON} plot_for_paper_coef_distribution_triangle.py 1
${PYTHON} plot_for_paper_coef_distribution_triangle.py 1.25

${PYTHON} plot_for_paper_coef_distribution.py

${PYTHON} combine_coef_triangle.py
