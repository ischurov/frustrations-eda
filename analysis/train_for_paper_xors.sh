

PYTHON=python

mkdir paper_final_xors

${PYTHON} train_for_paper_xors.py triangle truncated
${PYTHON} train_for_paper_xors.py square truncated
${PYTHON} train_for_paper_xors.py kagome truncated

${PYTHON} train_for_paper_xors.py triangle flat
${PYTHON} train_for_paper_xors.py square flat
${PYTHON} train_for_paper_xors.py kagome flat

${PYTHON} train_for_paper_xors.py kagome nosign
${PYTHON} train_for_paper_xors.py triangle nosign
${PYTHON} train_for_paper_xors.py square nosign
 
