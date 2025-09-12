
PYTHON=python

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
