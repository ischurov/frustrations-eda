
PYTHON=python

mkdir "umap"

${PYTHON} plot_for_paper_umap.py 50
${PYTHON} combine_umap_lattice.py 50

${PYTHON} plot_for_paper_umap.py 100
${PYTHON} combine_umap_lattice.py 100

${PYTHON} plot_for_paper_umap.py 200
${PYTHON} combine_umap_lattice.py 200


${PYTHON} plot_for_paper_umap.py 300
${PYTHON} combine_umap_lattice.py 300


${PYTHON} plot_for_paper_umap.py 400
${PYTHON} combine_umap_lattice.py 400


${PYTHON} plot_for_paper_umap.py 500
${PYTHON} combine_umap_lattice.py 500


${PYTHON} plot_for_paper_umap.py 1000
${PYTHON} combine_umap_lattice.py 1000


${PYTHON} plot_for_paper_umap.py 5000
${PYTHON} combine_umap_lattice.py 5000


${PYTHON} plot_for_paper_umap.py 10000
${PYTHON} combine_umap_lattice.py 10000


${PYTHON} plot_for_paper_umap.py 20000
${PYTHON} combine_umap_lattice.py 20000


${PYTHON} plot_for_paper_umap.py 30000
${PYTHON} combine_umap_lattice.py 30000

${PYTHON} plot_for_paper_components_umap.py 30000

${PYTHON} plot_for_paper_components_umap.py 10000
