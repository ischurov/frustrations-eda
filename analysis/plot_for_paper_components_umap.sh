#!/bin/bash
#
#SBATCH --job-name='xors'
#SBATCH -N1
#SBATCH --mem 0
#SBATCH -p tcm
#SBATCH --time=2-0:00:00
#SBATCH --output=/vol/tcm11/kravchenko/to_scratch/frustrations-eda-main/results_polished_umap/logs/slurm-%j.out
#SBATCH --error=/vol/tcm11/kravchenko/to_scratch/frustrations-eda-main/results_polished_umap/logs/slurm-%j.err

export PATH="/vol/tcm11/kravchenko/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH=/vol/tcm11/kravchenko/miniconda3/envs/ls/lib
source /vol/tcm11/kravchenko/miniconda3/bin/activate ls


PYTHON="/vol/tcm11/kravchenko/miniconda3/envs/ls/bin/python"

${PYTHON} plot_for_paper_components_umap.py 50
${PYTHON} combine_umap.py 50

${PYTHON} plot_for_paper_components_umap.py 100
${PYTHON} combine_umap.py 100

${PYTHON} plot_for_paper_components_umap.py 200
${PYTHON} combine_umap.py 200


${PYTHON} plot_for_paper_components_umap.py 300
${PYTHON} combine_umap.py 300


${PYTHON} plot_for_paper_components_umap.py 400
${PYTHON} combine_umap.py 400


${PYTHON} plot_for_paper_components_umap.py 500
${PYTHON} combine_umap.py 500


${PYTHON} plot_for_paper_components_umap.py 1000
${PYTHON} combine_umap.py 1000


${PYTHON} plot_for_paper_components_umap.py 5000
${PYTHON} combine_umap.py 5000


${PYTHON} plot_for_paper_components_umap.py 10000
${PYTHON} combine_umap.py 10000


${PYTHON} plot_for_paper_components_umap.py 20000
${PYTHON} combine_umap.py 20000


${PYTHON} plot_for_paper_components_umap.py 30000
${PYTHON} combine_umap.py 30000
